"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""
import sys
import json
import logging
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from fenitop.fem import form_fem
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import Communicator, XDMFTimeSeries


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def _configure_logger(log_path, file_prefix, comm):
    logger = logging.getLogger(f"fenitop_{file_prefix}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    if comm.rank == 0:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = FlushFileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def topopt(fem, opt):
    """Main function for topology optimization."""

    # Initialization
    comm = MPI.COMM_WORLD
    output_dir = Path(opt.get("output_folder", "results"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    opt["output_folder"] = str(output_dir)

    file_prefix = opt.get("output_prefix", "fenitop")
    if comm.rank == 0:
        for pattern in [
            f"{file_prefix}_*.xdmf",
            f"{file_prefix}_*.h5",
            f"{file_prefix}_summary.json",
            f"{file_prefix}_run.log",
        ]:
            for stale_path in output_dir.glob(pattern):
                stale_path.unlink(missing_ok=True)
    comm.barrier()

    logger = _configure_logger(output_dir / f"{file_prefix}_run.log", file_prefix, comm)
    if comm.rank == 0:
        logger.info("Starting topology optimization with output directory %s", output_dir)

    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_phys_field)
    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])

    # Initialize XDMF time series writer
    path = opt["output_folder"]
    density_saver = XDMFTimeSeries(
        fem["mesh"], f"{path}/{file_prefix}_density_history.xdmf", "density")
    displacement_saver = XDMFTimeSeries(
        fem["mesh"], f"{path}/{file_prefix}_displacement_history.xdmf", "displacement")

    num_consts = 1 if opt["opt_compliance"] else 2
    num_elems = rho_field.x.petsc_vec.array.size
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    # Apply passive zones
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.x.petsc_vec.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    initial_entry = {
        "iteration": 0,
        "state": "initial",
        "beta": 1.0,
        "compliance": None,
        "volume": None,
        "objective": None,
        "change": None,
        "grayness": None,
        "initial_density": float(opt["vol_frac"]),
    }
    if comm.rank == 0:
        logger.info("history %s", json.dumps(initial_entry, sort_keys=True))

    density_saver.save(rho_phys_field, 0.0)  # Save initial state
    displacement_saver.save(u_field, 0.0)
    if comm.rank == 0:
        logger.info("Saved initial design to XDMF time series.")
    save_interval = max(1, opt["max_iter"] // opt["output_interval"])  # Save ~20 time steps during optimization
    final_saved = False  # Track if final iteration was already saved

    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1

        # Density filter and Heaviside projection
        density_filter.forward()
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)

        # Solve FEM
        linear_problem.solve_fem()

        # Compute function values and sensitivities
        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)
        if opt["opt_compliance"]:
            g_vec = np.array([V_value-opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
        else:
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])

        # Update the design variables
        rho_values = rho_field.x.petsc_vec.array.copy()

        # print(rho_values.shape)
        if opt["opt_compliance"] and opt["use_oc"]:
            rho_new, change = optimality_criteria(
                rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(
                num_consts, num_elems, opt_iter, rho_values, rho_min, rho_max,
                rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"],
                logger=logger)
            rho_old2 = rho_old1.copy()
            rho_old1 = rho_values.copy()
        rho_field.x.petsc_vec.array[:] = rho_new.copy()

        # Save to time series at regular intervals
        saved_this_iter = False
        if opt_iter % save_interval == 0:
            density_saver.save(rho_phys_field, float(opt_iter))
            displacement_saver.save(u_field, float(opt_iter))
            saved_this_iter = True
            if comm.rank == 0:
                logger.info("Saved iteration %d to XDMF time series.", opt_iter)

        # Output the histories
        opt_time = time.perf_counter() - opt_start_time
        grayness = comm.allreduce(np.sum(np.abs(rho_new - 0.5)), op=MPI.SUM) / comm.allreduce(rho_new.size, op=MPI.SUM)
        history_entry = {
            "iteration": int(opt_iter),
            "state": "iterate",
            "beta": float(beta),
            "compliance": float(C_value),
            "volume": float(V_value),
            "objective": float(U_value),
            "change": float(change),
            "grayness": float(grayness),
            "time_seconds": float(opt_time),
        }
        if comm.rank == 0:
            logger.info("history %s", json.dumps(history_entry, sort_keys=True))
            logger.info(
                "iter=%d time=%.3f beta=%.3f compliance=%.6e volume=%.6f objective=%.6e change=%.6e grayness=%.6f",
                opt_iter,
                opt_time,
                beta,
                C_value,
                V_value,
                U_value,
                change,
                grayness,
            )
            
        # Store whether we saved this iteration for final check
        final_saved = saved_this_iter

    # Save final result to time series only if we didn't already save it
    if not final_saved:
        density_saver.save(rho_phys_field, float(opt_iter))
        displacement_saver.save(u_field, float(opt_iter))
        if comm.rank == 0:
            logger.info("Saved final design (iteration %d) to XDMF time series.", opt_iter)
    else:
        if comm.rank == 0:
            logger.info("Final design already saved at iteration %d.", opt_iter)

    values = S_comm.gather(rho_phys_field.x.petsc_vec)
    summary = None
    if comm.rank == 0 and values is not None:
        summary = {
            "iterations": int(opt_iter),
            "final_compliance": float(C_value),
            "final_volume": float(V_value),
            "final_objective": float(U_value),
            "grayness": float(grayness),
            "output_folder": str(output_dir),
            "output_prefix": file_prefix,
            "initial_density": float(opt["vol_frac"]),
        }
        summary_path = output_dir / f"{file_prefix}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Wrote summary to %s", summary_path)

    return {
        "density_values_serial": values,
        "mesh_serial": fem.get("mesh_serial"),
        "summary": summary,
        "converged_raw": {"opt_iter": int(opt_iter), "change": float(change)},
    }
