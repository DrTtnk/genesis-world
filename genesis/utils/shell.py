"""Rest-state geometry for VBD shells: the membrane's rest frame and the bending stencils built from the
mesh's shared edges.

The formulas here are exactly those proved in `spikes/verify_avbd_shell_math.py` (the AVBD shell math
spike), transcribed from torch to numpy with no change: this module derives nothing on its own.
"""

import numpy as np

import genesis as gs


def triangle_rest_frames(verts, tris):
    """The rest area and the 2x2 rest edge matrix inverse (`B_rest = Dm^-1`) of every triangle, in an
    orthonormal frame of its own rest plane. `F = Ds @ B_rest` with `Ds` the deformed edges in world space
    is then the in-plane deformation gradient, matching `membrane_energy` of the shell math spike."""
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    e0, e1 = p1 - p0, p2 - p0
    u = e0 / np.linalg.norm(e0, axis=-1, keepdims=True)
    n = np.cross(e0, e1)
    v = np.cross(n / np.linalg.norm(n, axis=-1, keepdims=True), u)
    Dm = np.stack(
        [
            np.stack([np.einsum("ij,ij->i", e0, u), np.einsum("ij,ij->i", e1, u)], axis=-1),
            np.stack([np.einsum("ij,ij->i", e0, v), np.einsum("ij,ij->i", e1, v)], axis=-1),
        ],
        axis=1,
    )
    area_rest = 0.5 * np.abs(np.linalg.det(Dm))
    B_rest = np.linalg.inv(Dm)
    return area_rest.astype(gs.np_float), B_rest.astype(gs.np_float)


def _flatten_stencil(rest):
    """`flatten_stencil` of the spike: the four vertex stencil laid isometrically into the plane."""
    x0, x1, x2, x3 = rest
    e = x1 - x0
    length = np.linalg.norm(e)
    u = e / length
    p = np.zeros((4, 2))
    p[1, 0] = length
    for k, x in ((2, x2), (3, x3)):
        d = x - x0
        along = d @ u
        across = np.sqrt(max((d @ d) - along**2, 0.0))
        p[k, 0] = along
        p[k, 1] = across if k == 2 else -across
    return p


def _bending_coefficients(rest):
    """`bending_coefficients` of the spike: the null space of the affine conditions on the flattened
    stencil, normalised so its first entry has unit size."""
    p = _flatten_stencil(rest)
    a = np.vstack([np.ones(4), p.T])
    _, _, vh = np.linalg.svd(a)
    c = vh[-1]
    return c / c[0]


def _bending_weight(rest):
    """`bending_weight` of the spike: 3 / (combined rest area of the two triangles)."""
    x0, x1, x2, x3 = rest
    area = 0.5 * (np.linalg.norm(np.cross(x1 - x0, x2 - x0)) + np.linalg.norm(np.cross(x1 - x0, x3 - x0)))
    return 3.0 / area


def bending_stencils(verts, tris):
    """One stencil `(v0, v1, v2, v3)` for every interior edge of the mesh: `v0, v1` the shared edge and
    `v2, v3` the opposite vertex of each of its two triangles. Returns the vertex quads, the coefficients
    `c`, the weight `w` and `Kx_rest = sum_i c_i x_rest_i` of every stencil, following `bending_energy` of
    the shell math spike. A boundary edge, shared by one triangle only, gets no stencil."""
    edge_tri = {}
    for tri in tris:
        for a, b, opp in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            key = (int(tri[a]), int(tri[b])) if tri[a] < tri[b] else (int(tri[b]), int(tri[a]))
            edge_tri.setdefault(key, []).append(int(tri[opp]))

    quads = [(*edge, *opp) for edge, opp in edge_tri.items() if len(opp) == 2]
    if not quads:
        return (
            np.zeros((0, 4), dtype=gs.np_int),
            np.zeros((0, 4), dtype=gs.np_float),
            np.zeros((0,), dtype=gs.np_float),
            np.zeros((0, 3), dtype=gs.np_float),
        )

    v = np.asarray(quads, dtype=gs.np_int)
    c = np.zeros((len(v), 4))
    w = np.zeros(len(v))
    kx_rest = np.zeros((len(v), 3))
    for i, quad in enumerate(v):
        rest = verts[quad]
        ci = _bending_coefficients(rest)
        c[i] = ci
        w[i] = _bending_weight(rest)
        kx_rest[i] = (ci[:, None] * rest).sum(0)
    return v, c.astype(gs.np_float), w.astype(gs.np_float), kx_rest.astype(gs.np_float)
