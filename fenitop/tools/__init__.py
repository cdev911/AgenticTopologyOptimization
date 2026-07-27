"""Agent-facing tool layer: validate_config, run_topopt, analyze_results.

Deliberately empty of eager imports: fenitop.tools.config_models and
fenitop.tools.safety need no dolfinx/MPI and must stay importable on a bare
host, while fenitop.tools.run_topopt does need the full dolfinx/PETSc/MPI
stack. Importing this package itself must not force that dependency.
"""
