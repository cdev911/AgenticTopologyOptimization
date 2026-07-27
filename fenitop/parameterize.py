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
from dolfinx import la
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import create_matrix, assemble_matrix
from petsc4py import PETSc

from fenitop.numerics import check_ksp, require_density_bounds, require_finite


class DensityFilter():
    def __init__(self, comm, rho, rho_tilde, R=1.0, petsc_options=None):
        """Construct a PDE filter."""
        if petsc_options is None:
            petsc_options = {}
        # Initialization
        S0, S = rho.function_space, rho_tilde.function_space
        u0, u = ufl.TrialFunction(S0), ufl.TrialFunction(S)
        v, self.af = ufl.TestFunction(S), Function(S)

        self.rho, self.rho_tilde = rho, rho_tilde
        self.rho_tilde_wrap = self.rho_tilde.x.petsc_vec
        self.af_wrap = self.af.x.petsc_vec
        self.vec_s0, self.vec_s = rho.x.petsc_vec.copy(), rho_tilde.x.petsc_vec.copy()

        # Construct Kf and T matrices based on the Helmholtz PDE
        dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})
        Kf_expr = (R**2*ufl.dot(ufl.grad(u), ufl.grad(v)) + u*v)*dx
        T_expr = u0*v*dx
        Kf_form, T_form = form(Kf_expr), form(T_expr)
        self.Kf_mat, self.T_mat = create_matrix(Kf_form), create_matrix(T_form)

        # Construct a filtering solver
        self.solver = PETSc.KSP().create(comm)
        self.solver.setOperators(self.Kf_mat)
        prefix = f"filter_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        # Apply PETSc options
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        self.Kf_mat.setOptionsPrefix(prefix)
        self.Kf_mat.setFromOptions()

        # Assemble Kf and T matrices
        assemble_matrix(self.Kf_mat, Kf_form)
        self.Kf_mat.assemble()
        assemble_matrix(self.T_mat, T_form)
        self.T_mat.assemble()
        self.T_mat_transpose = self.T_mat.copy()
        self.T_mat_transpose.transpose()

    def forward(self, iteration=None):
        """Compute the filtered variables."""
        require_density_bounds(
            "design density", self.rho.x.array,
            component="density_filter_forward", iteration=iteration, comm=self.rho.function_space.mesh.comm,
        )
        self.T_mat.mult(self.rho.x.petsc_vec, self.vec_s)
        self.solver.solve(self.vec_s, self.rho_tilde_wrap)
        check_ksp(self.solver, component="density_filter_forward", iteration=iteration)
        self.rho_tilde.x.scatter_forward()
        require_density_bounds(
            "filtered density", self.rho_tilde.x.array,
            component="density_filter_forward", iteration=iteration,
            tolerance=1e-8, comm=self.rho.function_space.mesh.comm,
        )
        return self.rho_tilde

    def backward(self, sf_vectors, iteration=None):
        """Recover the sensitivities."""
        values = []
        for index, sf in enumerate(sf_vectors):
            if sf is not None:
                require_finite(
                    f"projected sensitivity {index}", sf.array,
                    component="density_filter_adjoint", iteration=iteration,
                    comm=self.rho.function_space.mesh.comm,
                )
                self.solver.solve(sf, self.af_wrap)
                check_ksp(
                    self.solver, component="density_filter_adjoint", iteration=iteration
                )
                self.af.x.scatter_forward()
                self.T_mat_transpose.mult(self.af.x.petsc_vec, self.vec_s0)
                result = self.vec_s0.array.copy()
                require_finite(
                    f"design sensitivity {index}", result,
                    component="density_filter_adjoint", iteration=iteration,
                    comm=self.rho.function_space.mesh.comm,
                )
                values.append(result)
            else:
                values.append(None)
        return values

    def close(self):
        """Release PETSc resources deterministically; safe to call repeatedly."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.solver.destroy()
        self.Kf_mat.destroy()
        self.T_mat.destroy()
        self.T_mat_transpose.destroy()
        self.vec_s0.destroy()
        self.vec_s.destroy()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Heaviside():
    def __init__(self, rho_phys):
        self.rho_phys = rho_phys

    def forward(self, beta, eta=0.5, iteration=None):
        denominator = np.tanh(beta*eta) + np.tanh(beta*(1-eta))
        self.drho = beta*(1-np.tanh(beta*(self.rho_phys.x.petsc_vec-eta))**2) / denominator
        self.rho_phys.x.array[:] = (
            np.tanh(beta*eta)+np.tanh(beta*(self.rho_phys.x.array-eta))) / denominator
        self.rho_phys.x.scatter_forward()
        require_density_bounds(
            "projected physical density", self.rho_phys.x.array,
            component="heaviside_projection", iteration=iteration,
            tolerance=1e-8, comm=self.rho_phys.function_space.mesh.comm,
        )

    def backward(self, vectors, iteration=None):
        require_finite(
            "projection derivative", self.drho,
            component="heaviside_projection", iteration=iteration,
            comm=self.rho_phys.function_space.mesh.comm,
        )
        for index, vector in enumerate(vectors):
            if vector is not None:
                vector.array *= self.drho
                require_finite(
                    f"projected sensitivity {index}", vector.array,
                    component="heaviside_projection", iteration=iteration,
                    comm=self.rho_phys.function_space.mesh.comm,
                )
