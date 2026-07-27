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

import numpy as np
import ufl
import basix
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (functionspace, Function, Constant,
                         dirichletbc, locate_dofs_topological)

from fenitop.utility import create_mechanism_vectors
from fenitop.utility import LinearProblem
from fenitop.boundary_resolver import resolve_boundary
from fenitop.tools.mechanical_units import (
    MechanicalUnitContext,
    resultant_to_traction,
)


def _as_bc_value(value, dim):
    if value is None:
        return np.zeros(dim, dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            return np.full(dim, float(arr[0]), dtype=float)
        if arr.size != dim:
            raise ValueError(f"Boundary-condition values must have length {dim}; got {arr.size}.")
        return arr.astype(float)
    return np.full(dim, float(value), dtype=float)


def form_fem(fem, opt):
    """Form an FEA problem."""
    # Function spaces and functions
    mesh = fem["mesh"]
    V = functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))
    S0 = functionspace(mesh, ("DG", 0))
    S = functionspace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_field = Function(V)  # Displacement field
    lambda_field = Function(V)  # Adjoint variable field
    rho_field = Function(S0)  # Density field
    rho_phys_field = Function(S)  # Physical density field

    # Material interpolation
    E0, nu = fem["young's modulus"], fem["poisson's ratio"]
    p, eps = opt["penalty"], opt["epsilon"]
    E = (eps + (1-eps)*rho_phys_field**p) * E0
    _lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  # Lame constants

    # Kinematics
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):  # 3D or plane strain
        return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1
    dirichlet_bcs = []
    for bc_spec in fem.get("dirichlet_bcs", []):
        marker = bc_spec.get("marker", fem.get("disp_bc"))
        value = bc_spec.get("value", [0.0] * dim)
        facets = (
            resolve_boundary(mesh, bc_spec["selector"]).facets
            if "selector" in bc_spec
            else locate_entities_boundary(mesh, fdim, marker)
        )
        if len(facets) == 0:
            continue
        bc_value = _as_bc_value(value, dim)
        dirichlet_bcs.append(
            dirichletbc(Constant(mesh, bc_value), locate_dofs_topological(V, fdim, facets), V)
        )
    if not dirichlet_bcs and "disp_bc" in fem:
        disp_facets = locate_entities_boundary(mesh, fdim, fem["disp_bc"])
        dirichlet_bcs.append(
            dirichletbc(Constant(mesh, np.zeros(dim, dtype=float)),
                        locate_dofs_topological(V, fdim, disp_facets), V)
        )

    tractions, facets, markers = [], [], []
    unit_spec = fem.get("unit_context", {"kind": "legacy_consistent"})
    unit_context = (
        MechanicalUnitContext(
            length_unit=unit_spec["length_unit"],
            force_unit=unit_spec["force_unit"],
            stress_unit=unit_spec["stress_unit"],
            thickness_value=unit_spec.get("thickness_value", 1.0),
        )
        if unit_spec.get("kind") == "explicit"
        else None
    )
    for marker, load_spec in enumerate(fem.get("traction_bcs", [])):
        if isinstance(load_spec, dict):
            resolved = resolve_boundary(mesh, load_spec["selector"])
            traction = tuple(load_spec["value"])
            if load_spec.get("quantity_kind") == "resultant":
                traction, integrated = resultant_to_traction(
                    traction,
                    boundary_measure=resolved.measure,
                    context=unit_context,
                )
                if not np.allclose(
                    integrated,
                    load_spec["value"],
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Integrated resultant verification failed for "
                        f"{load_spec.get('bc_id', 'boundary load')}."
                    )
            current_facets = resolved.facets
        else:
            traction, traction_bc = load_spec
            current_facets = locate_entities_boundary(
                mesh, fdim, traction_bc
            )
        tractions.append(Constant(mesh, np.asarray(traction, dtype=float)))
        facets.extend(current_facets)
        markers.extend([marker,] * len(current_facets))
    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    _, unique_indices = np.unique(facets, return_index=True)
    facets, markers = facets[unique_indices], markers[unique_indices]
    sorted_indices = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[sorted_indices], markers[sorted_indices])

    metadata = {"quadrature_degree": fem["quadrature_degree"]}
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)
    b = Constant(mesh, np.array(fem["body_force"], dtype=float))

    # Establish the equilibrium and adjoint equations
    lhs = ufl.inner(sigma(u), epsilon(v))*dx
    rhs = ufl.dot(b, v)*dx
    for marker, t in enumerate(tractions):
        rhs += ufl.dot(t, v)*ds(marker)
    if opt["opt_compliance"]:
        spring_vec = opt["l_vec"] = None
    else:
        spring_vec, opt["l_vec"] = create_mechanism_vectors(
            V, opt["in_spring"], opt["out_spring"])
    linear_problem = LinearProblem(u_field, lambda_field, lhs, rhs, opt["l_vec"],
                                   spring_vec, dirichlet_bcs, fem["petsc_options"])

    # Define optimization-related variables
    opt["f_int"] = ufl.inner(sigma(u_field), epsilon(v))*dx
    opt["compliance"] = ufl.inner(sigma(u_field), epsilon(u_field))*dx
    opt["volume"] = rho_phys_field*dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx

    return linear_problem, u_field, lambda_field, rho_field, rho_phys_field
