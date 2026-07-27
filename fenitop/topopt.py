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
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from fenitop.fem import form_fem
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import Communicator, XDMFTimeSeries
from fenitop.numerics import require_density_bounds, require_finite


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def _atomic_write_json(path: Path, value) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _configure_logger(log_path, file_prefix, comm):
    logger = logging.getLogger(f"fenitop_{file_prefix}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    if comm.rank == 0:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = FlushFileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def _close_logger(logger):
    """Flush, close, and detach run-specific handlers deterministically."""
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def _initial_design_values(centers, opt):
    """Build the bounded design field, including passive solid/void zones."""
    num_elems = centers.shape[1]
    solid = np.asarray(opt["solid_zone"](centers), dtype=bool)
    void = np.asarray(opt["void_zone"](centers), dtype=bool)
    if solid.shape != (num_elems,) or void.shape != (num_elems,):
        raise ValueError("Passive-zone locators must return one boolean per design cell.")
    if np.any(solid & void):
        raise ValueError("Passive solid and void zones overlap.")
    configured_initial_density = float(opt["initial_density"])
    rho_ini = np.full(num_elems, configured_initial_density)
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01
    return rho_ini, rho_min, rho_max


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

    linear_problem = density_filter = sens_problem = None
    try:
        linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
        density_filter = DensityFilter(
            comm, rho_field, rho_phys_field,
            opt["filter_radius"], fem["petsc_options"],
        )
        heaviside = Heaviside(rho_phys_field)
        sens_problem = Sensitivity(
            comm, opt, linear_problem, u_field, lambda_field, rho_phys_field
        )
        S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])

        path = opt["output_folder"]
        density_saver = XDMFTimeSeries(
            fem["mesh"], f"{path}/{file_prefix}_density_history.xdmf", "density"
        )
        displacement_saver = XDMFTimeSeries(
            fem["mesh"], f"{path}/{file_prefix}_displacement_history.xdmf", "displacement"
        )

        num_consts = 1 if opt["opt_compliance"] else 2
        num_elems = rho_field.x.petsc_vec.array.size
        if not opt["use_oc"]:
            rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
            low, upp = None, None

        centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
        configured_initial_density = float(opt["initial_density"])
        rho_ini, rho_min, rho_max = _initial_design_values(centers, opt)
        rho_field.x.petsc_vec.array[:] = rho_ini
        rho_field.x.scatter_forward()
        require_density_bounds(
            "initial design density", rho_field.x.array,
            component="initialization", iteration=0, comm=comm,
        )

        def evaluate_state(iteration, beta):
            density_filter.forward(iteration=iteration)
            heaviside.forward(beta, iteration=iteration)
            linear_problem.solve_fem(iteration=iteration)
            values_, sensitivities_ = sens_problem.evaluate(iteration=iteration)
            heaviside.backward(sensitivities_, iteration=iteration)
            gradients_ = density_filter.backward(
                sensitivities_, iteration=iteration
            )
            require_finite(
                "evaluated metrics", values_,
                component="state_evaluation", iteration=iteration, comm=comm,
            )
            return values_, gradients_

        def material_metrics(iteration):
            owned = (
                rho_phys_field.function_space.dofmap.index_map.size_local
                * rho_phys_field.function_space.dofmap.index_map_bs
            )
            physical = rho_phys_field.x.array[:owned]
            require_density_bounds(
                "physical density", physical,
                component="state_metrics", iteration=iteration, comm=comm,
            )
            gray_sum = comm.allreduce(
                float(np.sum(4.0 * physical * (1.0 - physical))), op=MPI.SUM
            )
            count = comm.allreduce(int(physical.size), op=MPI.SUM)
            grayness_ = gray_sum / count
            binarization_ = 1.0 - grayness_
            require_finite(
                "material metrics", [grayness_, binarization_],
                component="state_metrics", iteration=iteration, comm=comm,
            )
            return float(grayness_), float(binarization_)

        save_interval = max(1, opt["max_iter"] // opt["output_interval"])
        opt_iter, beta, change = 0, 1.0, None
        optimizer_status = None

        # Iteration zero is a fully evaluated physical state, not an empty file placeholder.
        [C_value, V_value, U_value], gradients = evaluate_state(0, beta)
        grayness, binarization = material_metrics(0)
        initial_entry = {
            "iteration": 0,
            "state": "initial",
            "beta": float(beta),
            "compliance": float(C_value),
            "volume": float(V_value),
            "objective": float(U_value),
            "change": None,
            "grayness": grayness,
            "binarization_score": binarization,
            "initial_density": configured_initial_density,
        }
        if comm.rank == 0:
            logger.info("history %s", json.dumps(initial_entry, sort_keys=True))
        density_saver.save(rho_phys_field, 0.0)
        displacement_saver.save(u_field, 0.0)
        if comm.rank == 0:
            logger.info("Saved evaluated initial design to XDMF time series.")
        final_saved_iteration = 0

        while opt_iter < opt["max_iter"]:
            opt_start_time = time.perf_counter()
            opt_iter += 1
            dCdrho, dVdrho, dUdrho = gradients
            if opt["opt_compliance"]:
                g_vec = np.array([V_value-opt["vol_frac"]])
                dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
            else:
                g_vec = np.array([
                    V_value-opt["vol_frac"],
                    C_value-opt["compliance_bound"],
                ])
                dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])
            require_finite(
                "optimizer inputs", np.hstack([g_vec, dJdrho, dgdrho.ravel()]),
                component="optimizer", iteration=opt_iter, comm=comm,
            )

            rho_values = rho_field.x.petsc_vec.array.copy()
            if opt["opt_compliance"] and opt["use_oc"]:
                rho_new, change, optimizer_status = optimality_criteria(
                    rho_values, rho_min, rho_max, g_vec[0],
                    dJdrho, dgdrho[0], opt["move"],
                )
            else:
                rho_new, change, low, upp, optimizer_status = mma_optimizer(
                    num_consts, num_elems, opt_iter, rho_values, rho_min, rho_max,
                    rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp,
                    move=opt["move"], logger=logger,
                )
                rho_old2 = rho_old1.copy()
                rho_old1 = rho_values.copy()
            require_density_bounds(
                "optimizer density update", rho_new,
                component="optimizer", iteration=opt_iter, comm=comm,
            )
            rho_field.x.petsc_vec.array[:] = rho_new
            rho_field.x.scatter_forward()

            beta_changed = False
            if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
                beta = min(beta * 2.0, float(opt["beta_max"]))
                beta_changed = True

            # Evaluate the updated design before logging, saving, or terminating.
            [C_value, V_value, U_value], gradients = evaluate_state(opt_iter, beta)
            grayness, binarization = material_metrics(opt_iter)
            opt_time = time.perf_counter() - opt_start_time
            history_entry = {
                "iteration": int(opt_iter),
                "state": "iterate",
                "beta": float(beta),
                "compliance": float(C_value),
                "volume": float(V_value),
                "objective": float(U_value),
                "change": float(change),
                "grayness": grayness,
                "binarization_score": binarization,
                "optimizer_status": optimizer_status.as_dict(),
                "time_seconds": float(opt_time),
            }
            if comm.rank == 0:
                logger.info("history %s", json.dumps(history_entry, sort_keys=True))
                logger.info(
                    "iter=%d time=%.3f beta=%.3f compliance=%.6e volume=%.6f "
                    "objective=%.6e change=%.6e grayness=%.6f binarization=%.6f",
                    opt_iter, opt_time, beta, C_value, V_value, U_value,
                    change, grayness, binarization,
                )

            if opt_iter % save_interval == 0:
                density_saver.save(rho_phys_field, float(opt_iter))
                displacement_saver.save(u_field, float(opt_iter))
                final_saved_iteration = opt_iter
                if comm.rank == 0:
                    logger.info("Saved evaluated iteration %d to XDMF time series.", opt_iter)

            continuation_completed = beta >= float(opt["beta_max"])
            if (
                change <= opt["opt_tol"]
                and continuation_completed
                and not beta_changed
            ):
                break

        if final_saved_iteration != opt_iter:
            density_saver.save(rho_phys_field, float(opt_iter))
            displacement_saver.save(u_field, float(opt_iter))
            if comm.rank == 0:
                logger.info("Saved evaluated final design (iteration %d).", opt_iter)

        continuation_completed = beta >= float(opt["beta_max"])
        converged = bool(
            change <= opt["opt_tol"]
            and continuation_completed
            and not beta_changed
        )
        if converged:
            stop_reason = "tolerance_met"
        elif change <= opt["opt_tol"] and not continuation_completed:
            stop_reason = "continuation_incomplete"
        else:
            stop_reason = "max_iterations_reached"

        values = S_comm.gather(rho_phys_field.x.petsc_vec)
        summary = None
        if comm.rank == 0 and values is not None:
            summary = {
                "iterations": int(opt_iter),
                "final_compliance": float(C_value),
                "final_volume": float(V_value),
                "final_objective": float(U_value),
                "grayness": grayness,
                "binarization_score": binarization,
                "final_change": float(change),
                "final_beta": float(beta),
                "continuation_completed": continuation_completed,
                "converged": converged,
                "stop_reason": stop_reason,
                "optimizer_status": optimizer_status.as_dict(),
                "output_folder": str(output_dir),
                "output_prefix": file_prefix,
                "initial_density": configured_initial_density,
            }
            require_finite(
                "summary metrics",
                [
                    summary["final_compliance"], summary["final_volume"],
                    summary["final_objective"], summary["grayness"],
                    summary["binarization_score"], summary["final_change"],
                    summary["final_beta"],
                ],
                component="summary", iteration=opt_iter, comm=MPI.COMM_SELF,
            )
            summary_path = output_dir / f"{file_prefix}_summary.json"
            _atomic_write_json(summary_path, summary)
            logger.info("Wrote summary to %s", summary_path)

        return {
            "density_values_serial": values,
            "mesh_serial": fem.get("mesh_serial"),
            "summary": summary,
            "converged_raw": {
                "opt_iter": int(opt_iter),
                "change": float(change),
                "converged": converged,
                "stop_reason": stop_reason,
                "final_beta": float(beta),
                "continuation_completed": continuation_completed,
                "optimizer_status": optimizer_status.as_dict(),
            },
        }
    finally:
        if sens_problem is not None:
            sens_problem.close()
        if density_filter is not None:
            density_filter.close()
        if linear_problem is not None:
            linear_problem.close()
        _close_logger(logger)
