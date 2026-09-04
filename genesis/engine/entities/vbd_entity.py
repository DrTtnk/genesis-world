from pathlib import Path

import numpy as np
import quadrants as qd
import trimesh

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.repr_base import RBC
from genesis.utils.misc import tensor_to_array

from .base_entity import Entity


class VBDVisGeom(RBC):
    """A visual geom of a VBD entity, the deformable counterpart of `RigidVisGeom`.

    It carries the render mesh drawn by the visualizer, decoupled from the simulation mesh, and 'sim_verts_idx',
    the map from each render-mesh vertex to the simulated vertex standing for it (vertices co-located across visual
    geoms or duplicated by texture seams share a single simulated vertex).
    """

    def __init__(self, entity, vvert_start, vface_start, vmesh, sim_verts_idx):
        self._uid = gs.UID()
        self._entity = entity
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._vmesh = vmesh
        self._sim_verts_idx = sim_verts_idx

    def get_trimesh(self):
        """The underlying `trimesh.Trimesh` of the render mesh."""
        return self._vmesh.trimesh

    @property
    def uid(self):
        """Unique ID of the vgeom."""
        return self._uid

    @property
    def entity(self):
        """The VBD entity the vgeom belongs to."""
        return self._entity

    @property
    def vmesh(self):
        """The render mesh."""
        return self._vmesh

    @property
    def sim_verts_idx(self):
        """Map from render-mesh vertex index to the entity's simulated vertex index."""
        return self._sim_verts_idx

    @property
    def surface(self):
        """Surface object of the vgeom."""
        return self._vmesh.surface

    @property
    def uvs(self):
        """UV coordinates of the vgeom."""
        return self._vmesh.uvs

    @property
    def metadata(self):
        """Metadata of the render mesh."""
        return self._vmesh.metadata

    @property
    def n_vverts(self):
        """Number of render vertices of the vgeom."""
        return len(self._vmesh.verts)

    @property
    def n_vfaces(self):
        """Number of render faces of the vgeom."""
        return len(self._vmesh.faces)

    @property
    def vvert_start(self):
        """Starting index of the vgeom's render vertices in the VBD solver."""
        return self._vvert_start

    @property
    def vface_start(self):
        """Starting index of the vgeom's render faces in the VBD solver."""
        return self._vface_start

    @property
    def vvert_end(self):
        """Ending index of the vgeom's render vertices in the VBD solver."""
        return self._vvert_start + self.n_vverts

    @property
    def vface_end(self):
        """Ending index of the vgeom's render faces in the VBD solver."""
        return self._vface_start + self.n_vfaces


@qd.data_oriented
class VBDEntity(Entity):
    """
    A Vertex Block Descent (VBD)-based entity for deformable tetrahedral solids.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        idx,
        v_start=0,
        el_start=0,
        vvert_start=0,
        vface_start=0,
        name: str | None = None,
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        self._v_start = v_start  # offset for vertex index of elements
        self._el_start = el_start  # offset for element index
        self._vvert_start = vvert_start  # offset for render vertices
        self._vface_start = vface_start  # offset for render faces
        self._step_global_added = None
        self.sample()

        self.active = False  # This attribute is only used in forward pass.

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def instantiate(self, verts, elems):
        """
        Initialize VBD entity with given vertices and elements.

        Parameters
        ----------
        verts : np.ndarray
            Array of vertex positions with shape (n_vertices, 3).

        elems : np.ndarray
            Array of tetrahedra indexing into verts, with shape (n_elements, 4).
        """
        verts = verts.astype(gs.np_float, copy=False)
        elems = elems.astype(gs.np_int, copy=False)

        # Compose the morph pose offset (e.g. an up-axis conversion) onto the morph orientation, rotating the verts
        # about their COM (the pre-existing morph.quat convention), then translate by the body-frame offset position
        # R(morph.quat) @ offset_pos.
        morph_quat = np.array(self._morph.quat, dtype=gs.np_float)
        init_quat = gu.transform_quat_by_quat(np.array(self._morph.offset_quat, dtype=gs.np_float), morph_quat)
        R = gu.quat_to_R(init_quat)
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ R.T + verts_COM
        offset_shift = gu.transform_by_quat(np.array(self._morph.offset_pos, dtype=gs.np_float), morph_quat)
        init_positions = init_positions + offset_shift

        if not init_positions.shape[0] > 0:
            gs.raise_exception("Entity has zero vertices.")

        self.init_positions = gs.tensor(init_positions)
        self.elems = elems
        self._n_vertices = len(init_positions)
        self._n_elements = len(elems)

    def sample(self):
        """
        Build the entity's visual geoms and simulation mesh from its morph.

        Each morph sub-mesh becomes a visual geom with its own surface and UVs, while the simulation operates on a
        single welded copy of their vertices, tracked through 'VBDVisGeom.sim_verts_idx': welding and
        tetrahedralization both keep the input vertices first and in order, so these maps remain valid indices into
        the simulated vertices.
        """
        meshes = gs.Mesh.from_morph_surface(self._morph, self._surface)
        surface_verts, surface_faces, verts_maps = mu.merge_submeshes(
            [mesh.verts for mesh in meshes], [mesh.faces for mesh in meshes]
        )
        self._vgeoms = gs.List()
        vvert_start, vface_start = self._vvert_start, self._vface_start
        for mesh, verts_idx in zip(meshes, verts_maps):
            self._vgeoms.append(
                VBDVisGeom(
                    entity=self,
                    vvert_start=vvert_start,
                    vface_start=vface_start,
                    vmesh=mesh,
                    sim_verts_idx=verts_idx,
                )
            )
            vvert_start += len(mesh.verts)
            vface_start += len(mesh.faces)

        # Tetgen refinement depends on the absolute coordinates of its input. File meshes are tetrahedralized
        # untranslated so the result, and its on-disk cache, are shared across all placements of the same asset;
        # primitives keep the position baked in, as the simulated rest state is sensitive to the exact refinement.
        is_mesh_morph = isinstance(self._morph, gs.options.morphs.Mesh)
        if not is_mesh_morph:
            surface_verts = surface_verts + self._morph.pos
        surface_trimesh = trimesh.Trimesh(vertices=surface_verts, faces=surface_faces, process=False)
        verts, elems = eu.mesh_to_elements(surface_trimesh, tet_cfg=self.tet_cfg)
        if is_mesh_morph:
            verts = verts + self._morph.pos

        self.instantiate(verts, elems)

    def _add_to_solver(self, in_backward=False):
        if not in_backward:
            self._step_global_added = self._sim.cur_step_global
            gs.logger.info(
                f"Entity {self.uid} added. class: {self.__class__.__name__}, morph: {self.morph.__class__.__name__}, size: ({self.n_elements}, {self.n_vertices}), material: {self.material}."
            )

        verts_numpy = tensor_to_array(self.init_positions, dtype=gs.np_float)
        elems_np = self.elems.astype(gs.np_int, copy=False)

        p0 = verts_numpy[elems_np[:, 0]]
        p1 = verts_numpy[elems_np[:, 1]]
        p2 = verts_numpy[elems_np[:, 2]]
        p3 = verts_numpy[elems_np[:, 3]]
        Dm = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)
        total_rest_volume = np.abs(np.linalg.det(Dm)).sum() / 6.0

        gain = self.material.gain if isinstance(self.material, gs.materials.VBD.Muscle) else 0.0

        self._solver._kernel_add_elements(
            v_start=self._v_start,
            el_start=self._el_start,
            verts=verts_numpy,
            elems=elems_np,
            mass=float(self.material.rho * total_rest_volume / self.n_vertices),
            mu=self.material.mu,
            lam=self.material.lam,
            gain=gain,
            mu_forward=self.material.mu_forward,
            mu_backward=self.material.mu_backward,
            mu_lateral=self.material.mu_lateral,
        )

        for vgeom in self._vgeoms:
            # A vgeom without a texture carries no UVs; an empty array leaves its slice of the solver buffer zeroed.
            uvs = vgeom.uvs
            if uvs is None:
                uvs = np.zeros((0, 2), dtype=gs.np_float)
            self._solver._kernel_add_vverts(
                vvert_start=vgeom.vvert_start,
                vface_start=vgeom.vface_start,
                v_start=self._v_start,
                verts_idx=vgeom.sim_verts_idx,
                uvs=uvs,
                vfaces=vgeom.vmesh.faces.astype(gs.np_int, copy=False),
            )

        self.active = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic entity ops -------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        pass

    @qd.kernel
    def _kernel_get_frame(self, pos: qd.types.ndarray(), vel: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self._solver.verts[i_global, i_b].pos[j]
                vel[i_b, i_v, j] = self._solver.verts[i_global, i_b].vel[j]

    def get_state(self):
        """Positions and velocities of the entity's vertices, each of shape (B, n_vertices, 3)."""
        pos = gs.zeros((self._sim._B, self.n_vertices, 3), dtype=gs.tc_float, requires_grad=False, scene=self.scene)
        vel = gs.zeros((self._sim._B, self.n_vertices, 3), dtype=gs.tc_float, requires_grad=False, scene=self.scene)
        self._kernel_get_frame(pos, vel)
        return pos, vel

    def get_positions(self):
        """Positions of the entity's vertices, shape (B, n_vertices, 3)."""
        pos, _ = self.get_state()
        return pos

    def set_friction_frame(self, tangent):
        """
        Set the forward direction of every vertex for anisotropic floor friction.

        Parameters
        ----------
        tangent : array_like, shape (n_vertices, 3)
            Forward direction of each vertex. Its projection on the floor plane must be non-zero; it is
            normalized there. Sliding along it uses `material.mu_forward`, against it `mu_backward`,
            sideways `mu_lateral`. Default is +x for every vertex.
        """
        tangent = np.asarray(tangent, dtype=gs.np_float)
        if tangent.shape != (self.n_vertices, 3):
            gs.raise_exception(f"`tangent` should have shape ({self.n_vertices}, 3), got {tangent.shape}.")
        if np.any(np.linalg.norm(tangent[:, :2], axis=-1) < 1e-6):
            gs.raise_exception("`tangent` must have a non-zero projection on the floor plane for every vertex.")
        self._solver.set_friction_frame(self._v_start, tangent)

    def set_muscle(self, group, fiber):
        """
        Set the muscle group and fiber direction of each tetrahedron.

        Parameters
        ----------
        group : array_like, shape (n_elements,)
            Muscle group index of each tetrahedron, or -1 for a passive tetrahedron.
        fiber : array_like, shape (n_elements, 3)
            Unit fiber direction of each tetrahedron. Only used for tetrahedra with `group >= 0`.
        """
        if not isinstance(self.material, gs.materials.VBD.Muscle):
            gs.raise_exception("`set_muscle` is only supported by entities with `VBD.Muscle` material.")

        group = np.asarray(group)
        fiber = np.asarray(fiber)

        if group.shape != (self.n_elements,):
            gs.raise_exception(f"`group` should have shape ({self.n_elements},), got {group.shape}.")
        if fiber.shape != (self.n_elements, 3):
            gs.raise_exception(f"`fiber` should have shape ({self.n_elements}, 3), got {fiber.shape}.")
        if group.size > 0 and group.max() >= self.material.n_groups:
            gs.raise_exception(f"`group` has an entry >= n_groups ({self.material.n_groups}).")

        actuated = group >= 0
        fiber_norm = np.linalg.norm(fiber[actuated], axis=-1)
        if np.any(np.abs(fiber_norm - 1.0) > 1e-4):
            gs.raise_exception("`fiber` of an actuated tetrahedron must be a unit vector.")

        self._solver.set_muscle(self._el_start, group.astype(gs.np_int), fiber.astype(gs.np_float))

    def set_actuation(self, actus):
        """
        Set the actuation signal of each muscle group.

        Parameters
        ----------
        actus : array_like, shape (n_groups,) or (n_groups, B)
            Actuation of each muscle group, in [0, 1]. A 1D array is tiled across environments.
        """
        if not isinstance(self.material, gs.materials.VBD.Muscle):
            gs.raise_exception("`set_actuation` is only supported by entities with `VBD.Muscle` material.")

        actus = np.asarray(actus)
        n_groups = self.material.n_groups
        if actus.ndim == 1:
            if actus.shape != (n_groups,):
                gs.raise_exception(f"`actus` should have shape ({n_groups},), got {actus.shape}.")
            actus = np.tile(actus[:, None], (1, self._sim._B))
        elif actus.shape != (n_groups, self._sim._B):
            gs.raise_exception(
                f"`actus` should have shape ({n_groups},) or ({n_groups}, {self._sim._B}), got {actus.shape}."
            )

        if not ((actus >= 0.0) & (actus <= 1.0)).all():
            gs.raise_exception("`actus` must be in [0, 1].")

        self._solver.set_actuation(actus.astype(gs.np_float))

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        morph = self._morph

        if isinstance(morph, gs.morphs.Box):
            return "vbd_box"
        if isinstance(morph, gs.morphs.Sphere):
            return "vbd_sphere"
        if isinstance(morph, gs.morphs.Cylinder):
            return "vbd_cylinder"
        if isinstance(morph, gs.morphs.Mesh):
            return f"vbd_{Path(morph.file).stem}"
        return "vbd_entity"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_vertices(self):
        """Number of vertices in the VBD entity."""
        return self._n_vertices

    @property
    def n_elements(self):
        """Number of tetrahedra in the VBD entity."""
        return self._n_elements

    @property
    def vgeoms(self):
        """The list of visual geoms (`VBDVisGeom`) in the entity, one per morph sub-mesh."""
        return self._vgeoms

    @property
    def v_start(self):
        """Global vertex index offset for this entity."""
        return self._v_start

    @property
    def el_start(self):
        """Global element index offset for this entity."""
        return self._el_start

    @property
    def n_vverts(self):
        """Number of render vertices in the VBD entity, summed over its visual geoms."""
        return sum(vgeom.n_vverts for vgeom in self._vgeoms)

    @property
    def n_vfaces(self):
        """Number of render faces in the VBD entity, summed over its visual geoms."""
        return sum(vgeom.n_vfaces for vgeom in self._vgeoms)

    @property
    def vvert_start(self):
        """Global render vertex index offset for this entity."""
        return self._vvert_start

    @property
    def vface_start(self):
        """Global render face index offset for this entity."""
        return self._vface_start

    @property
    def vvert_end(self):
        """Global render vertex index past this entity's last one."""
        return self._vvert_start + self.n_vverts

    @property
    def vface_end(self):
        """Global render face index past this entity's last one."""
        return self._vface_start + self.n_vfaces

    @property
    def tet_cfg(self):
        """Configuration of tetrahedralization."""
        tet_cfg = mu.generate_tetgen_config_from_morph(self.morph)
        return tet_cfg
