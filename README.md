# FEniTop

FEniTop is a FEniCSx-based topology optimization package for 2D and 3D problems. It combines a finite-element formulation with density filtering, Heaviside projection, and optimization routines such as OC and MMA.

## Docker-based setup

The recommended way to run this project is in a container based on the stable Dolfinx image.

### Prerequisites

- Docker Engine
- Docker Compose

### Pull the base image

```bash
docker pull dolfinx/dolfinx:stable
```

If you want to target a specific release instead of the latest stable tag, you can also use:

```bash
docker pull dolfinx/dolfinx:v0.11.0
```

### Build the project image

```bash
docker compose build
```

### Run the examples

Run the 2D beam example:

```bash
docker compose run --rm fenitop python scripts/beam_2d.py --config config/beam_2d.json
```

Run the compliant mechanism example:

```bash
docker compose run --rm fenitop python scripts/mechanism_2d.py --config config/mechanism_2d.json
```

Run with MPI:

```bash
docker compose run --rm fenitop mpirun -n 2 python scripts/beam_2d.py --config config/beam_2d.json
```

## Configuration-driven runs

The example entry points now read a JSON configuration file. The repository includes example configs in the config directory:

- config/beam_2d.json
- config/mechanism_2d.json

The config file controls:

- mesh generation and boundary conditions
- constitutive law and FEM solver parameters
- objective and constraint settings
- filter parameters and optimizer settings
- output folder and output prefix

Each config also carries a parameter_guidance block with human-readable descriptions and validation rules. The loader validates the values before the run begins, so invalid settings such as negative iteration counts, decimal iteration counts, or volume fractions outside $[0, 1]$ fail fast with a clear error.

## Outputs

Each run writes FEniCSx-native XDMF time series output with HDF5 sidecar data:

- `<prefix>_density_history.xdmf` and `<prefix>_density_history.h5` contain the scalar physical density history.
- `<prefix>_displacement_history.xdmf` and `<prefix>_displacement_history.h5` contain the vector displacement history.
- `<prefix>_run.log` contains flushed per-iteration logging and structured `history` JSON records.
- `<prefix>_summary.json` contains the final compliance, volume, objective, grayness, and iteration count.

The XDMF files are intended to be opened in ParaView. Density and displacement are written as separate time series so each file has a clean, selectable field while keeping all time steps for that field in one place.

These files are written under the output folder defined in the config file, typically `results/` beneath the repository root. Generated output files are ignored by Git.

## Repository layout

- scripts/: example entry points
- fenitop/: core FEM, sensitivity, optimization, and utility modules
- config/: JSON config files for the examples

