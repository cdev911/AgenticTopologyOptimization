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
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.fem import Function

from fenitop.fem import form_fem
from fenitop.optimize import mma_optimizer, optimality_criteria
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.utility import Communicator


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def topopt(fem, opt):
    """Main function for topology optimization."""

    comm = MPI.COMM_WORLD
    output_dir = Path(opt.get("output_folder", "results"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    opt["output_folder"] = str(output_dir)

    file_prefix = opt.get("output_prefix", "fenitop")
    for pattern in [f"{file_prefix}_*.xdmf", f"{file_prefix}_*.npy", f"{file_prefix}_results.json", f"{file_prefix}_*.h5", f"{file_prefix}_run.log"]:
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink(missing_ok=True)
    log_path = output_dir / f"{file_prefix}_run.log"
    logger = logging.getLogger(f"fenitop_{file_prefix}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = FlushFileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    if comm.rank == 0:
        logger.info("Starting topology optimization with output directory %s", output_dir)

    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_phys_field)
    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])

    num_consts = 1 if opt["opt_compliance"] else 2
    num_elems = rho_field.x.petsc_vec.array.size
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    initial_density = float(opt.get("initial_density", opt.get("vol_frac", 0.5)))
    rho_ini = np.full(num_elems, initial_density)
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.x.petsc_vec.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    history_records = []
    density_snapshots = []
    field_snapshots = []
    initial_entry = {
        "iteration": 0,
        "state": "initial",
        "beta": 1.0,
        "compliance": None,
        "volume": None,
        "objective": None,
        "change": None,
        "grayness": None,
        "initial_density": initial_density,
    }
    history_records.append(initial_entry)
    density_snapshot = S_comm.gather(rho_phys_field.x.petsc_vec)
    if comm.rank == 0 and density_snapshot is not None:
        density_snapshots.append(np.asarray(density_snapshot, dtype=float))
    if comm.rank == 0:
        logger.info("history %s", json.dumps(initial_entry, sort_keys=True))

    opt_iter, beta, change = 0, 1, 2 * opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1

        density_filter.forward()
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)

        linear_problem.solve_fem()

        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)
        if comm.rank == 0:
            field_snapshots.append({
                "iteration": int(opt_iter),
                "density": np.asarray(rho_phys_field.x.petsc_vec.array, dtype=float).copy(),
                "displacement": np.asarray(u_field.x.petsc_vec.array, dtype=float).copy(),
                "adjoint": np.asarray(lambda_field.x.petsc_vec.array, dtype=float).copy(),
                "dCdrho": np.asarray(dCdrho, dtype=float).copy(),
                "dVdrho": np.asarray(dVdrho, dtype=float).copy(),
                "dUdrho": None if dUdrho is None else np.asarray(dUdrho, dtype=float).copy(),
            })
        if opt["opt_compliance"]:
            g_vec = np.array([V_value - opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
        else:
            g_vec = np.array([V_value - opt["vol_frac"], C_value - opt["compliance_bound"]])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])

        rho_values = rho_field.x.petsc_vec.array.copy()
        if opt["opt_compliance"] and opt["use_oc"]:
            rho_new, change = optimality_criteria(
                rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(
                num_consts, num_elems, opt_iter, rho_values, rho_min, rho_max,
                rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"])
            rho_old2 = rho_old1.copy()
            rho_old1 = rho_values.copy()
        rho_field.x.petsc_vec.array[:] = rho_new.copy()

        grayness = comm.allreduce(np.sum(np.abs(rho_new - 0.5)), op=MPI.SUM) / comm.allreduce(rho_new.size, op=MPI.SUM)

        opt_time = time.perf_counter() - opt_start_time
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
        history_records.append(history_entry)
        density_snapshot = S_comm.gather(rho_phys_field.x.petsc_vec)
        if comm.rank == 0 and density_snapshot is not None:
            density_snapshots.append(np.asarray(density_snapshot, dtype=float))
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

    values = S_comm.gather(rho_phys_field.x.petsc_vec)
    if comm.rank == 0 and values is not None:
        summary = {
            "iterations": int(opt_iter),
            "final_compliance": float(C_value),
            "final_volume": float(V_value),
            "final_objective": float(U_value),
            "grayness": float(grayness),
            "output_folder": str(output_dir),
            "output_prefix": file_prefix,
            "initial_density": initial_density,
        }
        with open(output_dir / f"{file_prefix}_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        xdmf_path = output_dir / f"{file_prefix}_fields.xdmf"
        mesh = fem["mesh"]
        with XDMFFile(comm, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(mesh)
            def write_scalar_field(name, values, space, time_value):
                field = Function(space)
                field.x.array[:] = values
                field.name = name
                xdmf.write_function(field, time_value)

            def write_vector_field(name, values, space, time_value):
                field = Function(space)
                field.x.array[:] = values
                field.name = name
                xdmf.write_function(field, time_value)

            write_scalar_field("density_design", np.asarray(rho_field.x.petsc_vec.array, dtype=float), rho_field.function_space, 0.0)
            write_scalar_field("density_physical", np.asarray(rho_phys_field.x.petsc_vec.array, dtype=float), rho_phys_field.function_space, 0.0)
            write_vector_field("displacement", np.asarray(u_field.x.petsc_vec.array, dtype=float), u_field.function_space, 0.0)
            write_vector_field("adjoint", np.asarray(lambda_field.x.petsc_vec.array, dtype=float), lambda_field.function_space, 0.0)
            if field_snapshots:
                latest = field_snapshots[-1]
                write_scalar_field("dCdrho", latest["dCdrho"], rho_field.function_space, float(latest["iteration"]))
                write_scalar_field("dVdrho", latest["dVdrho"], rho_field.function_space, float(latest["iteration"]))
                if latest["dUdrho"] is not None:
                    write_scalar_field("dUdrho", latest["dUdrho"], rho_field.function_space, float(latest["iteration"]))

        logger.info("Wrote consolidated XDMF artifact to %s", xdmf_path)
