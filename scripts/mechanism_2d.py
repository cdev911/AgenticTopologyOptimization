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
import argparse
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fenitop.config import build_mesh, load_config, normalize_boundary_conditions
from fenitop.topopt import topopt


def build_problem(config_path=None):
    config = load_config(config_path, default_case="mechanism_2d")
    mesh_spec = config.get("mesh", {})
    mesh = build_mesh(mesh_spec, MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 0:
        mesh_serial = build_mesh(mesh_spec, MPI.COMM_SELF)
    else:
        mesh_serial = None

    fem_config = config.get("fem", {})
    dirichlet_bcs, traction_bcs = normalize_boundary_conditions(fem_config, dim=2)

    fem = {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        "young's modulus": fem_config.get("young's modulus", 100),
        "poisson's ratio": fem_config.get("poisson's ratio", 0.25),
        "disp_bc": fem_config.get("disp_bc", lambda x: np.isclose(x[0], 0)),
        "dirichlet_bcs": dirichlet_bcs,
        "traction_bcs": traction_bcs,
        "body_force": tuple(fem_config.get("body_force", [0.0, 0.0])),
        "quadrature_degree": fem_config.get("quadrature_degree", 2),
        "petsc_options": fem_config.get("petsc_options", {}),
    }

    opt = dict(config.get("opt", {}))
    opt.setdefault("output_folder", "results")
    opt.setdefault("output_prefix", "mechanism_2d")
    opt.setdefault("output_interval", 20)
    opt.setdefault("initial_density", opt.get("vol_frac", 0.5))
    return fem, opt


def main():
    parser = argparse.ArgumentParser(description="Run the 2D mechanism topology optimization example")
    parser.add_argument("--config", default=None, help="Path to a JSON configuration file")
    args = parser.parse_args()

    fem, opt = build_problem(args.config)
    topopt(fem, opt)


if __name__ == "__main__":
    main()

# Execute the code in parallel:
# mpirun -n 8 python3 scripts/mechanism_2d.py --config config/mechanism_2d.json
