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
import logging

import numpy as np
import dolfinx.fem
import dolfinx.io
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector, set_bc
from petsc4py import PETSc
from scipy.spatial import cKDTree

from fenitop.numerics import check_ksp, require_finite

logger = logging.getLogger(__name__)


def create_mechanism_vectors(func_space, in_spring, out_spring):
    """Create vectors for compliant mechanism design."""
    index_map = func_space.dofmap.index_map
    block_size = func_space.dofmap.index_map_bs
    comm = func_space.mesh.comm
    size_local = index_map.size_local * block_size
    size_global = index_map.size_global * block_size

    spring_vec = PETSc.Vec().create(comm)
    spring_vec.setSizes((size_local, size_global))
    spring_vec.setFromOptions()
    spring_vec.setUp()
    l_vec = spring_vec.duplicate()

    local_range = index_map.local_range
    local_indices = np.arange(local_range[0], local_range[1]).astype(np.int32)
    local_size = np.ptp(local_range)
    local_nodes = func_space.tabulate_dof_coordinates()[:local_size]

    for n, (locator, direction, value) in enumerate([in_spring, out_spring]):
        ctrl_nodes = local_indices[locator(local_nodes.T)]
        offset = ["x", "y", "z"].index(direction)
        ctrl_dofs = ctrl_nodes * block_size + offset
        spring_vec.setValues(ctrl_dofs, [value] * ctrl_dofs.size)
        if n == 1:
            l_vec.setValues(ctrl_dofs, [1.0] * ctrl_dofs.size)
    spring_vec.assemble()
    l_vec.assemble()
    return spring_vec, l_vec


class LinearProblem:
    def __init__(self, u, lam, lhs, rhs, l_vec, spring_vec, bcs=None, petsc_options=None):
        """Initialize a linear problem."""
        if bcs is None:
            bcs = []
        if petsc_options is None:
            petsc_options = {}

        self.u, self.lam = u, lam
        self.u_wrap = self.u.x.petsc_vec
        self.lam_wrap = self.lam.x.petsc_vec
        self.lhs_form, self.rhs_form = form(lhs), form(rhs)
        self.lhs_mat = create_matrix(self.lhs_form)
        self.rhs_vec = create_vector(self.u.function_space)
        self.bcs, self.l_vec, self.spring_vec = bcs, l_vec, spring_vec

        self.solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self.solver.setOperators(self.lhs_mat)
        prefix = f"linear_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        for var in [self.lhs_mat, self.rhs_vec, self.l_vec]:
            if var is not None:
                var.setOptionsPrefix(prefix)
                var.setFromOptions()

        assemble_vector(self.rhs_vec, self.rhs_form)
        self.rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.rhs_vec, self.bcs)

    def solve_fem(self, iteration=None):
        """Solve K*x=F for FEM."""
        self.lhs_mat.zeroEntries()
        assemble_matrix(self.lhs_mat, self.lhs_form, bcs=self.bcs)
        self.lhs_mat.assemble()
        if self.spring_vec is not None:
            self.lhs_mat.setDiagonal(self.lhs_mat.getDiagonal() + self.spring_vec)
        self.solver.solve(self.rhs_vec, self.u_wrap)
        check_ksp(self.solver, component="elasticity", iteration=iteration)
        self.u.x.scatter_forward()
        require_finite(
            "displacement", self.u.x.array,
            component="elasticity", iteration=iteration,
            comm=self.u.function_space.mesh.comm,
        )

    def solve_adjoint(self, iteration=None):
        """Solve K*lambda=-L for the adjoint equation."""
        self.solver.solve(-self.l_vec, self.lam_wrap)
        check_ksp(self.solver, component="adjoint", iteration=iteration)
        self.lam.x.scatter_forward()
        require_finite(
            "adjoint field", self.lam.x.array,
            component="adjoint", iteration=iteration,
            comm=self.lam.function_space.mesh.comm,
        )

    def close(self):
        """Release PETSc resources deterministically; safe to call repeatedly."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.solver.destroy()
        self.lhs_mat.destroy()
        self.rhs_vec.destroy()
        if self.spring_vec is not None:
            self.spring_vec.destroy()
            self.l_vec.destroy()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Communicator:
    """Communicate information among different processes."""

    def __init__(self, func_space, mesh_serial, size=1):
        self.size = size
        self.comm = func_space.mesh.comm
        idx_map = func_space.dofmap.index_map

        num_local_nodes = idx_map.size_local
        num_global_nodes = idx_map.size_global
        num_nodal_dofs = func_space.dofmap.index_map_bs
        self.num_global_dofs = num_global_nodes * num_nodal_dofs

        local_nodal_range = np.asarray(idx_map.local_range, dtype=np.int32)
        local_dof_range = local_nodal_range * num_nodal_dofs
        local_nodes = func_space.tabulate_dof_coordinates()[:num_local_nodes]

        local_nodal_range_gather = self.comm.gather(local_nodal_range, root=0)
        self.local_dof_range_gather = self.comm.gather(local_dof_range, root=0)
        local_nodes_gather = self.comm.gather(local_nodes, root=0)

        element = func_space.ufl_element()
        if self.comm.rank == 0:
            func_space_serial = dolfinx.fem.functionspace(mesh_serial, element)
            nodes_serial = func_space_serial.tabulate_dof_coordinates()

            nodes_collect = np.zeros((num_global_nodes, 3))
            for r, nodes in zip(local_nodal_range_gather, local_nodes_gather):
                nodes_collect[r[0]:r[1]] = nodes
            global_to_local_nodes = compare_matrices(nodes_serial, nodes_collect)
            local_to_global_nodes = compare_matrices(nodes_collect, nodes_serial)

            def node2dof(nodes, num_nodal_dofs):
                return (np.tile(nodes, (num_nodal_dofs, 1)) * num_nodal_dofs
                        + np.arange(num_nodal_dofs).reshape(-1, 1)).ravel("F")

            global_to_local_dofs = node2dof(global_to_local_nodes, num_nodal_dofs)
            self.local_to_global_dofs = node2dof(local_to_global_nodes, num_nodal_dofs)
            self.local_to_global_dofs = (
                np.tile(self.local_to_global_dofs.reshape(-1, 1), (1, size)) * size + np.arange(size)).ravel()
        else:
            global_to_local_dofs = None
        global_to_local_dofs = self.comm.bcast(global_to_local_dofs, root=0)
        self.idx = global_to_local_dofs[local_dof_range[0]:local_dof_range[1]]

    def bcast(self, func, global_values):
        """Broadcast data from Process 0 to all the other processes."""
        if func.vector.size != global_values.size:
            raise ValueError("Mismatched sizes.")
        func.vector.array = global_values[self.idx]

    def gather(self, func):
        """Gather data to Process 0 from all the other processes."""
        if type(func) is Function:
            values_gather = self.comm.gather(func.x.array, root=0)
        elif type(func) is PETSc.Vec:
            values_gather = self.comm.gather(func.array, root=0)
        elif type(func) is np.ndarray:
            values_gather = self.comm.gather(func, root=0)
        else:
            raise TypeError("Unsupported func.")

        if self.comm.rank == 0:
            values_collect = np.zeros(self.num_global_dofs * self.size)
            for r, local_values in zip(self.local_dof_range_gather, values_gather):
                values_collect[r[0] * self.size:r[1] * self.size] = local_values
            global_values = values_collect[self.local_to_global_dofs]
        else:
            global_values = None
        return global_values


def compare_matrices(array1, array2, precision=12, k=1):
    """Find the "args" such that array1[args] == array2."""
    kd_tree = cKDTree(array1.round(precision))
    return kd_tree.query(array2.round(precision), k=k)[1]


class XDMFTimeSeries:
    def __init__(self, mesh_obj, filename="optimization_history.xdmf", field_name="density"):
        self.mesh_obj = mesh_obj
        self.filename = filename
        self.field_name = field_name
        self.initialized = False
        self.comm = mesh_obj.comm
        self._seen = set()

    def save(self, rho, time_step):
        t = float(time_step)
        if t in self._seen:
            return
        self._seen.add(t)

        mode = "w" if not self.initialized else "a"
        with dolfinx.io.XDMFFile(self.comm, self.filename, mode) as xdmf:
            if not self.initialized:
                xdmf.write_mesh(self.mesh_obj)
                self.initialized = True
                if self.comm.rank == 0:
                    logger.info("Created XDMF time series file: %s", self.filename)
            rho.name = self.field_name
            xdmf.write_function(rho, t)
