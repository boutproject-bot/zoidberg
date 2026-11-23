"""Unstructured poloidal grids for BOUT++ FCI.

This module generates BOUT++ grid files for flux-coordinate independent
(FCI) simulations in which every poloidal plane is an unstructured
triangle mesh instead of a logically rectangular ``(x, z)`` grid.

Grid-file contract
==================

The BOUT++ side (``FCIMap``, ``PrecomputedInterpolation``,
``FV::Div_a_Grad_perp_FCI_unstructured``, ``PetscOperators``) relies on
the following layout and variables.

Layout ("nodes along x")
    ``nx = n_nodes + 2*MXG``, ``ny`` poloidal planes, ``nz = 1``. Node ``k`` of
    plane ``j`` is stored at index ``(MXG + k, j, 0)``. ``MXG`` and ``MYG = 1``
    are written to the file (``MXG = 1``, or ``max(n_inner, n_outer)`` with
    Dirichlet walls, see below). The ``MXG`` guard cells on either side are
    padding.
    Their ``cell_number`` is ``-1``, so no unstructured operator ever
    references them. Their geometric fields are copies of the nearest
    node, so all values are finite. BOUT++ splits the evolving nodes evenly
    between the ``NXPE`` processors in x, so ``NXPE`` must divide their
    number. With the default ``multiple_of`` = 4, NXPE in {1, 2, 4} always
    works; other counts only give a warning with the valid ``NXPE`` values.
    Nodes are sorted in Morton (Z-order) to improve locality.

Dirichlet walls (optional, ``dirichlet_walls=True``)
    The wall nodes are stored in the x guard cells instead: inner-wall nodes
    at ``x = 0 .. n_inner-1``, outer-wall nodes at the end, and
    ``MXG = max(n_inner, n_outer)``. The walls are the planes' marked
    ``walls`` (any shape; e.g. the flux surfaces or the boundary loops of a
    gmsh mesh, see :meth:`UnstructuredPoloidalGrid.from_gmsh`), or, for
    unmarked meshes not periodic in R, the lines of minimum and maximum R.
    Wall nodes have ``cell_number >= 0`` (the x-boundary cells of the cell
    space), no divergence row and no parallel legs. BOUT++ does not evolve
    them; set them with the ``dirichlet_node`` boundary condition, which
    evaluates the boundary value at each guard cell. ``n_nodes`` then counts
    all nodes, and ``nx = n_evolving + 2*MXG``. Because BOUT++ exchanges
    ``MXG`` x guard cells between processors, ``NXPE > 1`` needs at least
    ``MXG`` evolving nodes per processor (BOUT++ checks this at start-up).

Meshing
    :meth:`UnstructuredPoloidalGrid.from_gmsh` meshes a disc or annulus with
    gmsh; :func:`gmsh_planes` meshes every poloidal cross-section
    independently and equalises the node counts that the layout requires
    (same number of nodes and of wall nodes in every plane). gmsh is only
    imported there.

Flag
    ``unstructured = 1`` (int).

Cell and leg numbering (as :func:`zoidberg.weights.assign_cell_space_numbers`)
    ``n_nodes``: nodes per plane (with Dirichlet walls including the wall
    nodes).
    ``total_cells``: size of the cell space (rows of the divergence matrices).
    ``cell_number``: stored cell number of each node (``-1`` for padding).
    ``forward_cell_number`` / ``backward_cell_number``: parallel-boundary
    virtual cell of a node whose forward/backward field line leaves the
    domain (else ``-1``).
    ``n_forward_legs`` / ``n_backward_legs``: number of legs (one per
    evolving node and plane) in each direction.
    ``forward_leg_interior_number`` / ``backward_leg_interior_number``: leg
    number of a node whose field line stays inside the domain (else ``-1``).
    ``forward_leg_boundary_number`` / ``backward_leg_boundary_number``: leg
    number of a node whose field line leaves the domain (else ``-1``).

Geometry and maps
    ``R``, ``Z``, ``Rxy``; ``forward_R``, ``forward_Z``, ``backward_R``,
    ``backward_Z`` (traced field-line end points). ``forward_xt_prime`` /
    ``backward_xt_prime`` are classification sentinels only:
    ``MXG + k`` if the field line stays inside the domain, ``-1`` for an inner wall
    hit and ``nx`` for an outer wall hit. ``forward_zt_prime`` /
    ``backward_zt_prime`` are 0.

Parallel operators (support operator method, read by ``PetscOperators``)
    These use the same names and numbering scheme as
    :func:`zoidberg.weights.calculate_parallel_map_operators`, with
    single-sample legs. Interior leg rows hold a value-interpolation stencil:
    a least-squares fit of order ``p = 4`` (default ``map_order``) over the
    ``num_coefs(p) + 16`` nearest target-plane unknowns (``map_extra``) in
    anisotropically scaled coordinates ``(R/h_R, Z/h_Z)``, weighted by
    ``(r0/max(r, r0))^3`` (``map_weight_power``; r the scaled distance, r0
    the larger of the nearest distance and one node spacing), using
    minimum-image offsets in periodic directions. The weighting keeps the
    interpolation amplification sum|w| small (on the Dommaschk example a
    median of 1.6 and a maximum of 2.6, instead of 2.1 and 6.6 for an
    unweighted fit over ``num_coefs(p) + 4`` nodes), which reduces the
    number of solver steps in stiff parallel-conduction problems.
    If an end point coincides with a node (within ``snap_tol*h``), the row is
    an injection.

Perpendicular diffusion (poloidal-plane form)
    ``dagp_us_div``, ``dagp_us_gradp`` and ``dagp_us_value``, each as
    ``_rows``, ``_columns``, ``_weights`` and ``_ncols`` (CSR). Each matrix
    is global over all planes and block-diagonal in y. Cell indices are
    stored cell numbers; face-point indices are global. Also written:
    ``dagp_us_n_facepoints``, ``dagp_us_facepoint_owner`` (stored cell
    number of the lower-numbered node of each face) and ``dagp_us_volume``
    (dual-cell area, a Field3D). Optional (written by this module, used for
    upwinding and flows): ``dagp_us_left``/``dagp_us_right`` (upwind
    reconstruction from the unknown on either side of each face point: its
    order-p least-squares fit evaluated at the face point) and ``dagp_us_flowR``/
    ``dagp_us_flowZ`` (cell average of the flux vector -a grad f,
    ``(1/V) sum_f (x_f - x_node) * outward flux``, exact for linear f).

    ``div @ ((value @ a) * (gradp @ f))`` approximates
    ``div_pol(a grad_pol f)`` with node-centred, centroid-dual finite
    volumes. Each face has quadrature points with least-squares gradients
    (order 4) plus one stabilisation point (compact two-point difference
    minus its high-order prediction) that removes growing grid-scale modes.
    The face values of a come from a separate order-2 fit (``value_order``),
    which is enough for second order. Interior dual faces connect the
    centroids of the two adjacent triangles; at a boundary edge the face
    runs from the centroid to the edge midpoint, and the wall itself is
    zero-flux.

Metric
    ``dx = dz = 1`` (Coordinates requires uniform dx, dz for FCI; the
    dual-cell areas are in ``dagp_us_volume``), ``dy = Ly/ny``. ``g11 = g33 = 1``,
    ``g13 = g12 = g23 = 0`` and the matching covariant components.
    ``g_22 = (|B|/B_y)^2`` for Cartesian (Slab-type) fields and
    ``(R |B|/B_y)^2`` for cylindrical ones (y the toroidal angle), as in
    zoidberg's structured metric; ``g22 = 1/g_22``; ``Bxy = |B|``.
    ``d2x = d2y = d2z = 0``. Parallel slices ``forward_*``/``backward_*`` of
    ``g_11, g_22, g_33, g_13, g11, g22, g33, g13, dy, Bxy`` are evaluated at
    the traced end points. ``By``, ``By_cell_ylow/yhigh`` and
    ``g_22_cell_ylow/yhigh`` (half-step positions) as for structured grids.
    For cylindrical fields the perpendicular operator weights volumes and
    fluxes by R.

Parallel divergence ("K2", written by default)
    K2: the flux-reconstructed conservative parallel divergence used by
    BOUT++'s conduction method ``petscoperatork2``. ``k2_forward_rows``,
    ``k2_forward_columns``, ``k2_forward_weights`` and the same for
    ``k2_backward``: the forward and backward half-divergences as CSR
    matrices, rows in stored cell numbers (``total_cells`` rows), columns in
    stored forward/backward leg numbers. The weights contain ``1/dl`` with
    the lengths ``dl = dy*sqrt(g_22)`` of the grid file (metres); BOUT++
    rescales them to the lengths it uses. ``k2_order`` (reconstruction
    order: 2 quadratic, 1 linear, 0 constant) and ``k2_version`` (format
    version, checked by BOUT++). See :func:`k2_divergence` for the
    convention; ``build_grid_data(..., k2=False)`` omits them and
    :func:`add_k2_to_file` adds them to an existing file.
"""

import copy
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
from scipy.spatial import KDTree

from . import fieldtracer
from .field import MagneticField, Slab

# Two-point Gauss-Legendre rule on [0, 1]: (positions, weights)
_GAUSS2 = (
    np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)]),
    np.array([0.5, 0.5]),
)


class StencilFallbackWarning(UserWarning):
    """Raised when a least-squares stencil has to drop to a lower order."""


def num_coefs(order):
    """Number of 2D monomials of total degree <= order."""
    return (order + 1) * (order + 2) // 2


def _monomials(u, v, order):
    """Vandermonde columns [1, u, v, u^2, uv, v^2, ...] for arrays u, v."""
    cols = []
    for degree in range(order + 1):
        for j in range(degree + 1):
            cols.append(u ** (degree - j) * v**j)
    return np.stack(cols, axis=-1)


def _pinv_from_svd(u, s, vt):
    """Pseudo-inverses of stacked matrices from their (reduced) SVD, exactly
    as :func:`numpy.linalg.pinv` computes them with its default tolerance
    (singular values below 1e-15 times the largest are dropped)."""
    cutoff = np.asarray(1e-15)[..., None] * np.amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    s = np.divide(1, s, where=large, out=s.copy())
    s[~large] = 0
    return np.matmul(
        np.swapaxes(vt, -1, -2), np.multiply(s[..., None], np.swapaxes(u, -1, -2))
    )


def _morton_order(R, Z, bits=16):
    """Return the permutation sorting points along a Morton (Z-order) curve."""

    def quantise(a):
        span = np.ptp(a)
        if span == 0:
            return np.zeros(a.shape, dtype=np.uint64)
        return ((a - a.min()) / span * (2**bits - 1)).astype(np.uint64)

    qr, qz = quantise(R), quantise(Z)
    key = np.zeros(R.shape, dtype=np.uint64)
    for b in range(bits):
        key |= ((qr >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b)
        key |= ((qz >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b + 1)
    return np.argsort(key, kind="stable")


def _curve_length(c):
    """Length of the closed polygon through the points ``c[:, 2]``."""
    d = np.diff(np.vstack([c, c[:1]]), axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _gmsh_mesh(
    loops, counts, mesh_size, embed=None, wall_size=None, wall_distance=None
):
    """Triangulate the region bounded by closed curves with gmsh.

    ``loops[0]`` is the outer boundary, further loops are holes; each is a
    closed spline with ``counts[i]`` equally spaced nodes (transfinite). With
    ``embed``, these interior points are fixed mesh nodes and no other
    interior nodes are inserted. With ``wall_size``, the triangle size grows
    linearly from ``wall_size`` at ``loops[0]`` to ``mesh_size`` at the
    distance ``wall_distance`` from it. Returns ``(points, triangles,
    wall_nodes)``, where ``wall_nodes[i]`` are the nodes on ``loops[i]``.
    """
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plane")
        geo = gmsh.model.geo
        curves, curve_loops = [], []
        for pts, n in zip(loops, counts):
            tags = [geo.addPoint(r, z, 0.0, mesh_size) for r, z in pts]
            spline = geo.addSpline(tags + [tags[0]])
            geo.mesh.setTransfiniteCurve(spline, n + 1)
            curves.append(spline)
            curve_loops.append(geo.addCurveLoop([spline]))
        surface = geo.addPlaneSurface(curve_loops)
        embedded = []
        if embed is not None:
            embedded = [geo.addPoint(r, z, 0.0, mesh_size) for r, z in embed]
        geo.synchronize()
        if wall_size is not None and not embedded:
            field = gmsh.model.mesh.field
            dist = field.add("Distance")
            field.setNumbers(dist, "CurvesList", [curves[0]])
            field.setNumber(dist, "Sampling", 4 * counts[0])
            grade = field.add("Threshold")
            field.setNumber(grade, "InField", dist)
            field.setNumber(grade, "SizeMin", wall_size)
            field.setNumber(grade, "SizeMax", mesh_size)
            field.setNumber(grade, "DistMin", 0.0)
            field.setNumber(grade, "DistMax", wall_distance)
            field.setAsBackgroundMesh(grade)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        if embedded:
            gmsh.model.mesh.embed(0, embedded, 2, surface)
            # Only the embedded points: no size-driven node insertion
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            big = 10.0 * max(np.ptp(np.vstack(loops), axis=0))
            gmsh.option.setNumber("Mesh.MeshSizeMin", big)
            gmsh.option.setNumber("Mesh.MeshSizeMax", big)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
        gmsh.model.mesh.generate(2)
        _, conn = gmsh.model.mesh.getElementsByType(2, surface)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        wall_tags = [
            gmsh.model.mesh.getNodes(1, c, includeBoundary=True)[0] for c in curves
        ]
    finally:
        gmsh.finalize()
    conn = conn.reshape(-1, 3)
    # Keep the nodes used by triangles (spline control points are not)
    used = np.unique(conn)
    pos = dict(zip(tags, coords.reshape(-1, 3)[:, :2]))
    points = np.array([pos[t] for t in used])
    index = np.searchsorted(used, conn)
    walls = [
        np.unique(np.searchsorted(used, np.intersect1d(w, used))) for w in wall_tags
    ]
    for w, n in zip(walls, counts):
        if len(w) != n:
            raise RuntimeError(f"Wall has {len(w)} nodes, expected {n}")
    return points, index, walls


class UnstructuredPoloidalGrid:
    """A triangulated poloidal plane.

    Parameters
    ----------
    R, Z : array_like
        Node coordinates, shape ``(n_nodes,)``
    triangles : array_like
        Node indices of each triangle, shape ``(n_tri, 3)``
    period_R, period_Z : float, optional
        Period of the domain in R / Z. ``None`` means non-periodic
    h_R, h_Z : float, optional
        Characteristic node spacing in R / Z, used to scale coordinates
        for neighbour selection and least-squares fits. Defaults to
        ``sqrt(area / n_nodes)`` in both directions
    origin : (float, float), optional
        Lower-left corner of the periodic box (only used for wrapping)
    walls : (array_like, array_like), optional
        Node indices on the inner and outer wall of a non-periodic domain
        (either may be empty). Used for Dirichlet wall nodes
        (``UnstructuredGrid(dirichlet_walls=True)``), to classify field lines
        that leave the domain, and to keep wall nodes fixed during
        :meth:`centroidal_relax`. ``None`` (default): for meshes from
        :meth:`slab`, the walls are the lines of minimum and maximum R.
        :meth:`from_gmsh` marks the walls of the meshes it makes.
    """

    #: Number of targets per block in :meth:`lsq_weights` (memory bound)
    lsq_block_size = 8192

    def __init__(
        self,
        R,
        Z,
        triangles,
        period_R=None,
        period_Z=None,
        h_R=None,
        h_Z=None,
        origin=(0.0, 0.0),
        walls=None,
    ):
        self.R = np.asarray(R, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.period = np.array(
            [
                np.inf if period_R is None else float(period_R),
                np.inf if period_Z is None else float(period_Z),
            ]
        )
        self.origin = np.asarray(origin, dtype=float)
        self.n_nodes = len(self.R)
        if h_R is None or h_Z is None:
            area = self._total_area()
            h = np.sqrt(area / self.n_nodes)
            h_R = h if h_R is None else h_R
            h_Z = h if h_Z is None else h_Z
        self.h = np.array([float(h_R), float(h_Z)])
        self.walls = None
        self.centroid_dofs = False
        self._dof = None
        if walls is not None:
            self.walls = tuple(np.asarray(w, dtype=int).ravel() for w in walls)
        self._faces = None
        self._topology = None
        self._kdtree = None

    # ------------------------------------------------------------------
    # Constructors

    @classmethod
    def from_meshio(cls, path, **kwargs):
        """Read a (non-periodic) triangle mesh with meshio."""
        import meshio

        mesh = meshio.read(path)
        tri = mesh.cells_dict["triangle"]
        return cls(mesh.points[:, 0], mesh.points[:, 1], tri, **kwargs)

    @classmethod
    def from_gmsh(
        cls,
        outer,
        inner=None,
        mesh_size=None,
        n_outer=None,
        n_inner=None,
        n_nodes=None,
        relax=False,
        morton=True,
        centroid_dofs=True,
        wall_size=None,
        wall_distance=None,
    ):
        """Mesh a disc or annulus with gmsh (Delaunay triangulation).

        Parameters
        ----------
        outer, inner : array_like
            Closed wall curves, shape ``(n, 2)`` (R, Z) without repeating the
            first point. Each is meshed as a closed spline through these
            points. ``inner=None`` gives a disc.
        mesh_size : float
            Target triangle edge length
        n_outer, n_inner : int, optional
            Number of nodes on each wall (default: curve length/mesh_size).
            Fix these to give several planes the same wall-node counts.
        n_nodes : int, optional
            Required total number of nodes. If gmsh gives fewer, the missing
            nodes are added at the centroids of the largest triangles and the
            domain is re-triangulated by gmsh with all nodes fixed. More
            nodes than ``n_nodes`` is an error (use :func:`gmsh_planes`,
            which takes the largest count of all planes).
        relax : bool
            Apply :meth:`centroidal_relax` (wall nodes stay fixed). Not
            needed with ``centroid_dofs``
        morton : bool
            Sort the interior nodes in Morton order
        centroid_dofs : bool
            Place the unknowns at the dual-cell centroids
            (:meth:`use_centroid_dofs`), which gives second-order
            truncation error on the unrelaxed gmsh mesh
        wall_size, wall_distance : float, optional
            Graded mesh: node spacing ``wall_size`` on the outer wall, growing
            linearly to ``mesh_size`` at the distance ``wall_distance`` from it
            (e.g. to resolve the layer next to a material wall as finely as a
            flux-aligned structured grid)

        The nodes are ordered inner wall, interior, outer wall, and
        ``walls`` is set accordingly. gmsh is imported only here.
        """
        outer = np.asarray(outer, dtype=float)
        loops = [outer] if inner is None else [outer, np.asarray(inner, dtype=float)]
        if mesh_size is None:
            raise ValueError("mesh_size is required")
        counts = [n_outer, n_inner][: len(loops)]
        sizes = [mesh_size if wall_size is None else wall_size, mesh_size]
        counts = [
            int(np.ceil(_curve_length(c) / h)) if n is None else int(n)
            for c, n, h in zip(loops, counts, sizes)
        ]
        pts, tri, wall_nodes = _gmsh_mesh(
            loops, counts, mesh_size, wall_size=wall_size, wall_distance=wall_distance
        )
        if n_nodes is not None and len(pts) != n_nodes:
            if len(pts) > n_nodes:
                raise ValueError(
                    f"gmsh produced {len(pts)} nodes, more than n_nodes = {n_nodes}"
                )
            on_wall = np.zeros(len(pts), dtype=bool)
            on_wall[np.concatenate(wall_nodes)] = True
            P = pts[tri]
            area = 0.5 * np.abs(
                (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1])
            )
            largest = np.argsort(area, kind="stable")[::-1][: n_nodes - len(pts)]
            embed = np.vstack([pts[~on_wall], P[largest].mean(axis=1)])
            pts, tri, wall_nodes = _gmsh_mesh(loops, counts, mesh_size, embed=embed)
            if len(pts) != n_nodes:
                raise RuntimeError(
                    f"gmsh re-triangulation gave {len(pts)} nodes, expected {n_nodes}"
                )
        # Order: inner wall, interior, outer wall
        outer_nodes = wall_nodes[0]
        inner_nodes = wall_nodes[1] if len(wall_nodes) > 1 else np.zeros(0, int)
        on_wall = np.zeros(len(pts), dtype=bool)
        on_wall[outer_nodes] = True
        on_wall[inner_nodes] = True
        interior = np.flatnonzero(~on_wall)
        if morton and len(interior):
            interior = interior[_morton_order(pts[interior, 0], pts[interior, 1])]
        perm = np.concatenate([inner_nodes, interior, outer_nodes])
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(len(perm))
        n_in, n_out = len(inner_nodes), len(outer_nodes)
        grid = cls(
            pts[perm, 0],
            pts[perm, 1],
            inverse[tri],
            walls=(np.arange(n_in), np.arange(len(perm) - n_out, len(perm))),
        )
        if relax:
            grid.centroidal_relax(omega=1.5)
        if centroid_dofs:
            grid.use_centroid_dofs()
        return grid

    @classmethod
    def slab(
        cls,
        nR,
        nZ,
        Lx,
        Lz,
        R0=0.0,
        Z0=0.0,
        jitter=0.2,
        jitter_dirs="z",
        periodic_R=False,
        periodic_Z=True,
        seed=0,
        morton=True,
        relax=True,
        r_on_walls=False,
    ):
        """Jittered lattice on ``[R0, R0+Lx] x [Z0, Z0+Lz)``.

        R positions are cell centred, ``R0 + (i + 1/2) Lx/nR``, as for BOUT++
        with ``symmetricglobalx``, or with ``r_on_walls`` evenly spaced from
        ``R0`` to ``R0 + Lx`` so that nodes lie on both walls. Z positions are ``Z0 + k Lz/nZ``, matching
        BOUT++ ``z = 2 pi k / nz``. Each lattice quadrilateral is split
        along a randomly chosen diagonal, which gives irregular connectivity.
        With ``jitter <= 0.25`` every quadrilateral stays convex, so the
        triangulation stays valid. Nodes on non-periodic R walls are never
        moved in R, so the walls stay straight.

        With ``relax=True`` (default), nodes are then moved to the centroids
        of their dual cells (:meth:`centroidal_relax`). This is needed for a
        second-order pointwise truncation error of the finite-volume
        perpendicular operator on irregular meshes. The irregular connectivity
        from the random diagonals remains.
        """
        if jitter > 0.25:
            raise ValueError("jitter must be <= 0.25 to keep a valid triangulation")
        rng = np.random.default_rng(seed)
        hR, hZ = Lx / nR, Lz / nZ
        i, k = np.meshgrid(np.arange(nR), np.arange(nZ), indexing="ij")
        if r_on_walls:
            if periodic_R:
                raise ValueError("r_on_walls needs a domain that is not periodic in R")
            # Nodes on both walls (R0 and R0 + Lx), e.g. for Dirichlet walls
            hR = Lx / (nR - 1)
            R = R0 + i * hR
        else:
            R = R0 + (i + 0.5) * hR
        Z = Z0 + k * hZ
        if "r" in jitter_dirs.lower():
            dR = rng.uniform(-jitter, jitter, R.shape) * hR
            if not periodic_R:
                dR[0, :] = 0.0
                dR[-1, :] = 0.0
            R = R + dR
        if "z" in jitter_dirs.lower():
            Z = Z + rng.uniform(-jitter, jitter, Z.shape) * hZ

        def node(ii, kk):
            return (ii % nR) * nZ + (kk % nZ)

        tris = []
        flips = rng.integers(0, 2, size=(nR, nZ))
        imax = nR if periodic_R else nR - 1
        kmax = nZ if periodic_Z else nZ - 1
        for ii in range(imax):
            for kk in range(kmax):
                a, b = node(ii, kk), node(ii + 1, kk)
                c, d = node(ii + 1, kk + 1), node(ii, kk + 1)
                if flips[ii, kk]:
                    tris += [(a, b, c), (a, c, d)]
                else:
                    tris += [(a, b, d), (b, c, d)]
        tris = np.array(tris)
        R, Z = R.ravel(), Z.ravel()

        if morton:
            perm = _morton_order(R, Z)
            inverse = np.empty_like(perm)
            inverse[perm] = np.arange(len(perm))
            R, Z, tris = R[perm], Z[perm], inverse[tris]

        grid = cls(
            R,
            Z,
            tris,
            period_R=Lx if periodic_R else None,
            period_Z=Lz if periodic_Z else None,
            h_R=hR,
            h_Z=hZ,
            origin=(R0, Z0),
        )
        if relax:
            grid.centroidal_relax(omega=1.5)
        return grid

    # ------------------------------------------------------------------
    # Geometry helpers

    @property
    def points(self):
        return np.column_stack((self.R, self.Z))

    @property
    def dof(self):
        """Positions of the unknowns, shape ``(n_nodes, 2)``: the mesh nodes,
        or with :meth:`use_centroid_dofs` the centroids of their dual cells
        (wall nodes stay on the wall)."""
        if not self.centroid_dofs:
            return self.points
        if self._dof is None:
            offset = self.dual_centroid_offsets()
            if self.walls is not None:
                offset[np.concatenate(self.walls)] = 0.0
            self._dof = self.wrap(self.points + offset)
        return self._dof

    def use_centroid_dofs(self, enable=True):
        """Place the unknowns at the centroids of their dual cells.

        Node-centred finite volumes give the average of the divergence over
        each dual cell, which is a second-order approximation at the cell
        centroid but only first order at the node when the node is offset
        from the centroid by O(h), as on any unrelaxed mesh (e.g. gmsh).
        With centroid unknowns the dual cells of the triangulation are kept
        unchanged (cell-centred finite volumes on the dual mesh, so the
        scheme stays conservative); the least-squares stencils, the
        two-point stabilisation, the field-line start points, the metric and
        the written ``R``, ``Z`` all use the centroids. This makes
        :meth:`centroidal_relax` unnecessary. Marked wall nodes (Dirichlet
        nodes) stay on the wall.
        """
        self.centroid_dofs = bool(enable)
        self._dof = None
        self._kdtree = None
        for attr in ("_centroid_tree", "_centroids"):
            if hasattr(self, attr):
                delattr(self, attr)
        return self

    def minimum_image(self, d):
        """Wrap displacement vectors ``d[..., 2]`` into the periodic box."""
        d = np.array(d, dtype=float, copy=True)
        for dim in range(2):
            p = self.period[dim]
            if np.isfinite(p):
                d[..., dim] -= p * np.round(d[..., dim] / p)
        return d

    def wrap(self, pos):
        """Wrap absolute positions ``pos[..., 2]`` into the periodic box."""
        pos = np.array(pos, dtype=float, copy=True)
        for dim in range(2):
            p = self.period[dim]
            if np.isfinite(p):
                pos[..., dim] = self.origin[dim] + np.mod(
                    pos[..., dim] - self.origin[dim], p
                )
        return pos

    def _unwrapped_triangles(self):
        """Triangle vertex positions, unwrapped relative to vertex 0."""
        pts = self.points[self.triangles]  # (n_tri, 3, 2)
        rel = self.minimum_image(pts - pts[:, :1, :])
        return pts[:, :1, :] + rel

    def _total_area(self):
        tri = self._unwrapped_triangles()
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        return 0.5 * np.sum(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))

    def total_area(self):
        return self._total_area()

    def euler_characteristic(self):
        """V - E + F (2 on a sphere/disc+outside, 1 for a disc, 0 on a torus)."""
        edges = set()
        for t in self.triangles:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                edges.add((min(a, b), max(a, b)))
        return self.n_nodes - len(edges) + len(self.triangles)

    # ------------------------------------------------------------------
    # Dual mesh

    def _edge_topology(self):
        """Primal edges ``i < j`` with their first and second triangle. They
        depend only on the connectivity, so they are kept when the nodes move
        (e.g. in :meth:`centroidal_relax`)."""
        if self._topology is not None:
            return self._topology
        t = self.triangles
        pairs = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        tri_of = np.tile(np.arange(len(t)), 3)
        pairs = np.sort(pairs, axis=1)
        edges, inverse, counts = np.unique(
            pairs, axis=0, return_inverse=True, return_counts=True
        )
        inverse = inverse.ravel()
        if np.any(counts > 2):
            raise ValueError("An edge is shared by more than two triangles")
        order = np.argsort(inverse, kind="stable")
        first = np.zeros(len(edges), dtype=int)
        second = np.full(len(edges), -1)
        starts = np.searchsorted(inverse[order], np.arange(len(edges)))
        first = tri_of[order[starts]]
        has2 = counts == 2
        second[has2] = tri_of[order[starts[has2] + 1]]
        self._topology = (edges, first, second, has2)
        return self._topology

    def faces(self):
        """Dual faces of the centroid-dual mesh.

        Returns a dict with arrays (one entry per primal edge ``i < j``):

        - ``i``, ``j``: node indices
        - ``start``, ``end``: face end points, relative to node ``i``
          (unwrapped), oriented so that the normal ``(dZ, -dR)`` of
          ``end - start`` points from ``i`` to ``j``
        - ``boundary``: True if the edge lies on a domain wall
        """
        if self._faces is not None:
            return self._faces
        pts = self.points
        tri_pts = self._unwrapped_triangles()
        centroids = tri_pts.mean(axis=1)  # unwrapped near vertex 0

        edges, first, second, has2 = self._edge_topology()
        I, J = edges[:, 0], edges[:, 1]
        xi = pts[I]
        xj_rel = self.minimum_image(pts[J] - xi)
        s = self.minimum_image(centroids[first] - xi)
        e = np.where(
            has2[:, None],
            self.minimum_image(centroids[np.maximum(second, 0)] - xi),
            0.5 * xj_rel,
        )
        d = e - s
        flip = (d[:, 1] * xj_rel[:, 0] - d[:, 0] * xj_rel[:, 1]) < 0
        s2 = np.where(flip[:, None], e, s)
        e2 = np.where(flip[:, None], s, e)
        self._faces = {"i": I, "j": J, "start": s2, "end": e2, "boundary": ~has2}
        return self._faces

    def _invalidate(self):
        self._faces = None
        self._kdtree = None
        self._dof = None
        for attr in ("_centroid_tree", "_centroids"):
            if hasattr(self, attr):
                delattr(self, attr)

    def dual_centroid_offsets(self):
        """Offset of each node from the centroid of its dual cell."""
        f = self.faces()
        s, e = f["start"], f["end"]
        n = self.n_nodes
        mom = np.zeros((n, 2))
        vol = np.zeros(n)
        a_i = 0.5 * (s[:, 0] * e[:, 1] - s[:, 1] * e[:, 0])
        np.add.at(vol, f["i"], a_i)
        np.add.at(mom, f["i"], a_i[:, None] * (s + e) / 3.0)
        xj_rel = self.minimum_image(self.points[f["j"]] - self.points[f["i"]])
        sj, ej = s - xj_rel, e - xj_rel
        a_j = -0.5 * (sj[:, 0] * ej[:, 1] - sj[:, 1] * ej[:, 0])
        np.add.at(vol, f["j"], a_j)
        np.add.at(mom, f["j"], a_j[:, None] * (sj + ej) / 3.0)
        return mom / vol[:, None]

    def centroidal_relax(self, tol=1e-4, max_iter=5000, omega=1.0):
        """Move nodes to the centroids of their dual cells (Lloyd-type
        iteration with fixed connectivity).

        Node-centred finite volumes have an O(h) pointwise truncation error
        when nodes are offset from their control-volume centroids, which
        happens on any irregular mesh. After relaxation the offset is below
        ``tol*h`` in each direction, which restores second order. Nodes on
        straight non-periodic walls only move along the wall; nodes in
        ``walls`` do not move. Returns the number of iterations.
        """
        wall = np.zeros((self.n_nodes, 2), dtype=bool)
        for dim in range(2):
            if not np.isfinite(self.period[dim]):
                c = self.points[:, dim]
                span = np.ptp(c)
                on_wall = (np.abs(c - c.min()) < 1e-12 * span) | (
                    np.abs(c - c.max()) < 1e-12 * span
                )
                wall[on_wall, dim] = True
        if self.walls is not None:
            # Marked (possibly curved) walls: their nodes do not move
            wall[np.concatenate(self.walls)] = True
        for it in range(1, max_iter + 1):
            delta = self.dual_centroid_offsets()
            delta[wall] = 0.0
            if np.max(np.abs(delta) / self.h) < tol:
                break
            delta *= omega
            pos = self.wrap(self.points + delta)
            self.R, self.Z = pos[:, 0], pos[:, 1]
            self._invalidate()
        else:
            # Nodes on a wall can keep sliding slowly along it; an offset this
            # small is negligible next to the O(h^2) truncation error.
            final_delta = self.dual_centroid_offsets()
            final_delta[wall] = 0.0  # wall-normal offsets are inherent
            final = np.max(np.abs(final_delta) / self.h)
            if final > 10 * tol:
                warnings.warn(
                    f"centroidal_relax: offset {final:.2g}*h after {max_iter} iterations"
                )
        self._check_orientation()
        return it

    def _check_orientation(self):
        tri = self._unwrapped_triangles()
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        if not (np.all(cross > 0) or np.all(cross < 0)):
            raise ValueError("Triangulation became tangled")

    def dual_volumes(self, cylindrical=False):
        """Area of each node's centroid-dual cell (``cylindrical``: the
        integral of R over it, i.e. the volume per radian of toroidal angle).

        Each face contributes the signed area of the triangle (node, start,
        end). Wall segments pass through the node itself and contribute
        nothing.
        """
        f = self.faces()
        vol = np.zeros(self.n_nodes)
        s, e = f["start"], f["end"]
        # relative to node i
        area_i = 0.5 * (s[:, 0] * e[:, 1] - s[:, 1] * e[:, 0])
        if cylindrical:
            # Exact integral of R over the triangle (node, start, end)
            area_i = area_i * (self.R[f["i"]] + (s[:, 0] + e[:, 0]) / 3.0)
        np.add.at(vol, f["i"], area_i)
        # relative to node j: positions shifted by -xj_rel, traversal reversed
        xj_rel = self.minimum_image(self.points[f["j"]] - self.points[f["i"]])
        sj, ej = s - xj_rel, e - xj_rel
        area_j = -0.5 * (sj[:, 0] * ej[:, 1] - sj[:, 1] * ej[:, 0])
        if cylindrical:
            area_j = area_j * (self.R[f["j"]] + (sj[:, 0] + ej[:, 0]) / 3.0)
        np.add.at(vol, f["j"], area_j)
        if np.any(vol <= 0):
            raise ValueError("Non-positive dual volume; triangulation is not valid")
        return vol

    # ------------------------------------------------------------------
    # Least-squares stencils

    def _tree(self):
        """KD-tree on scaled coordinates (periodic where needed).

        Non-periodic coordinates are shifted so that they are positive and
        given a box much larger than their extent, which makes the
        periodic KD-tree behave as non-periodic in that direction.
        """
        if self._kdtree is None:
            scaled = (self.dof - self.origin) / self.h
            self._tree_shift = np.zeros(2)
            box = np.zeros(2)
            for dim in range(2):
                p = self.period[dim]
                if np.isfinite(p):
                    box[dim] = p / self.h[dim]
                else:
                    span = np.ptp(scaled[:, dim])
                    self._tree_shift[dim] = scaled[:, dim].min() - span - 1.0
                    box[dim] = 3.0 * span + 3.0
            self._tree_box = box
            self._kdtree = KDTree(self._scaled(self.dof), boxsize=box)
        return self._kdtree

    def _scaled(self, pos):
        q = (np.asarray(pos, dtype=float) - self.origin) / self.h
        for dim in range(2):
            if np.isfinite(self.period[dim]):
                q[..., dim] = np.mod(q[..., dim], self.period[dim] / self.h[dim])
            else:
                q[..., dim] -= self._tree_shift[dim]
        # Points far outside a non-periodic domain are wall hits and never
        # interpolated; clip them into the box to keep the KD-tree happy
        return np.clip(q, 0.0, np.nextafter(self._tree_box, 0.0))

    def lsq_weights(
        self,
        targets,
        order=3,
        extra=4,
        gradient=False,
        snap_tol=1e-10,
        max_cond=1e8,
        strict=False,
        weight_power=0.0,
        eval_offsets=None,
        nodes=None,
    ):
        """Least-squares value (and gradient) weights at target points.

        Returns ``(inds, w_value[, w_dR, w_dZ][, w_eval], info)``, each of shape
        ``(m, k)``, with ``k = num_coefs(order) + extra``. With
        ``eval_offsets`` (shape ``(m, q, 2)``, physical offsets from each
        target), ``w_eval`` (shape ``(m, q, k)``) evaluates the same fitted
        polynomial at ``target + offset``. ``info`` contains
        ``fallback`` (number of points that dropped to a lower order) and
        ``snapped`` (number of injections). Value weights are exact for
        polynomials of total degree <= order. ``nodes`` (index array) restricts
        the stencils to these nodes (default: all nodes).
        """
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        m = len(targets)
        k0 = num_coefs(order) + extra
        n_avail = self.n_nodes if nodes is None else len(nodes)
        if k0 > n_avail:
            raise ValueError(
                f"Need at least {k0} nodes for an order-{order} stencil, "
                f"mesh has {n_avail}"
            )
        # The fits are independent per target: process the targets in blocks
        # so that the work arrays (Vandermonde matrices, SVDs, pseudo-
        # inverses: O(block * k * num_coefs)) stay small
        block = self.lsq_block_size
        if m > block:
            parts = []
            for start in range(0, m, block):
                sl = slice(start, start + block)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", StencilFallbackWarning)
                    parts.append(
                        self.lsq_weights(
                            targets[sl],
                            order=order,
                            extra=extra,
                            gradient=gradient,
                            snap_tol=snap_tol,
                            max_cond=max_cond,
                            strict=strict,
                            weight_power=weight_power,
                            eval_offsets=None
                            if eval_offsets is None
                            else np.asarray(eval_offsets, dtype=float)[sl],
                            nodes=nodes,
                        )
                    )
            info = {
                key: sum(part[-1][key] for part in parts)
                for key in ("fallback", "snapped")
            }
            if info["fallback"]:
                warnings.warn(
                    f"{info['fallback']} of {m} stencils dropped below order {order}",
                    StencilFallbackWarning,
                )
            n_out = len(parts[0]) - 1
            out = tuple(
                np.concatenate([part[i] for part in parts]) for i in range(n_out)
            )
            return out + (info,)
        # Targets that occur several times (e.g. the reconstructions around
        # a node, evaluated at each of its face points) are fitted once
        inverse = None
        if eval_offsets is not None and m > 1:
            uniq, inverse = np.unique(targets, axis=0, return_inverse=True)
            inverse = np.ravel(inverse)
            if len(uniq) > 0.75 * m:
                inverse = None
            else:
                targets = uniq
        mu = len(targets)

        # Stencils grow before they drop in order: next to a wall, the k0
        # nearest nodes can span too few layers for a full-order fit.
        k_list = [kk for kk in (k0, k0 + 6, k0 + 14) if kk <= n_avail]
        kmax = k_list[-1]
        tree = self._tree()
        if nodes is not None:
            tree = KDTree(self._scaled(self.dof[nodes]), boxsize=self._tree_box)
        _, inds_all = tree.query(self._scaled(targets), k=kmax)
        inds_all = np.atleast_2d(inds_all)
        if nodes is not None:
            inds_all = np.asarray(nodes)[inds_all]
        offsets = self.minimum_image(self.dof[inds_all] - targets[:, None, :])
        u_all = offsets[..., 0] / self.h[0]
        v_all = offsets[..., 1] / self.h[1]

        inds = np.zeros((mu, kmax), dtype=int)
        w_val = np.zeros((mu, kmax))
        w_dR = np.zeros((mu, kmax))
        w_dZ = np.zeros((mu, kmax))
        groups = []  # (targets, order, stencil size, pseudo-inverses)
        done = np.zeros(mu, dtype=bool)
        fallback_targets = np.zeros(mu, dtype=bool)
        attempts = [(p, kk) for p in range(order, 0, -1) for kk in k_list]
        for attempt, (p, kk) in enumerate(attempts):
            todo = np.flatnonzero(~done)
            if len(todo) == 0:
                break
            u, v = u_all[todo, :kk], v_all[todo, :kk]
            V = _monomials(u, v, p)  # (mt, kk, nc)
            if weight_power:
                # Weighted least squares: w = (r0 / max(r, r0))^weight_power,
                # with r the scaled distance and r0 the larger of the closest
                # distance and one node spacing (so that a target next to a
                # node does not make the fit ill-conditioned)
                r = np.sqrt(u**2 + v**2)
                r0 = np.maximum(r.min(axis=1, keepdims=True), 1.0)
                sw = np.sqrt((r0 / np.maximum(r, r0)) ** weight_power)
                V = V * sw[..., None]
            # One SVD gives both the condition number and the pseudo-inverse
            # (computed as numpy.linalg.pinv does)
            su, sv, svt = np.linalg.svd(V, full_matrices=False)
            cond = sv[:, 0] / np.maximum(sv[:, -1], 1e-300)
            last = attempt == len(attempts) - 1
            good = (cond <= max_cond) | last
            if p < order and np.any(good) and strict:
                raise ValueError(
                    f"{np.sum(good)} stencils ill-conditioned at order {order} "
                    f"even with {k_list[-1]} neighbours; strict mode forbids "
                    "fallback"
                )
            idx = todo[good]
            P = _pinv_from_svd(su[good], sv[good], svt[good])
            if weight_power:
                P = P * sw[good][:, None, :]
            inds[idx] = inds_all[idx]
            w_val[idx, :kk] = P[:, 0, :]
            w_dR[idx, :kk] = P[:, 1, :] / self.h[0]
            w_dZ[idx, :kk] = P[:, 2, :] / self.h[1]
            if eval_offsets is not None:
                groups.append((idx, p, kk, P))
            done[idx] = True
            if p < order:
                fallback_targets[idx] = True
        u_all0 = u_all[:, 0]
        v_all0 = v_all[:, 0]

        snapped = np.zeros(mu, dtype=bool)
        if snap_tol is not None:
            nearest = np.sqrt(u_all0**2 + v_all0**2)
            snapped = nearest < snap_tol
            w_val[snapped] = 0.0
            w_val[snapped, 0] = 1.0
        if inverse is not None:
            inds, w_val, w_dR, w_dZ = (a[inverse] for a in (inds, w_val, w_dR, w_dZ))
            snapped, fallback_targets = snapped[inverse], fallback_targets[inverse]
        fallback = int(np.sum(fallback_targets))
        if fallback:
            warnings.warn(
                f"{fallback} of {m} stencils dropped below order {order}",
                StencilFallbackWarning,
            )
        out = (inds, w_val)
        if gradient:
            out += (w_dR, w_dZ)
        if eval_offsets is not None:
            eval_offsets = np.asarray(eval_offsets, dtype=float)
            w_eval = np.zeros((m, eval_offsets.shape[1], kmax))
            for idx, p, kk, P in groups:
                if inverse is None:
                    sel, Psel = idx, P
                else:
                    row = np.full(mu, -1)
                    row[idx] = np.arange(len(idx))
                    sel = np.flatnonzero(row[inverse] >= 0)
                    Psel = P[row[inverse[sel]]]
                off = eval_offsets[sel]
                mono = _monomials(off[..., 0] / self.h[0], off[..., 1] / self.h[1], p)
                w_eval[sel, :, :kk] = mono @ Psel
            out += (w_eval,)
        info = {"fallback": fallback, "snapped": int(np.sum(snapped))}
        return out + (info,)

    def in_domain(self, pos, tol=1e-9):
        """True where positions lie inside the meshed domain, or within
        ``tol*h`` of it. Uses a barycentric test against the triangles whose
        centroids are nearest to each point (periodic-aware)."""
        pos = np.atleast_2d(np.asarray(pos, dtype=float))
        if np.all(np.isfinite(self.period)):
            return np.ones(len(pos), dtype=bool)
        tri = self._unwrapped_triangles()
        if not hasattr(self, "_centroid_tree"):
            self._tree()  # sets the scaling used by _scaled
            cen = self.wrap(tri.mean(axis=1))
            self._centroids = cen
            self._centroid_tree = KDTree(self._scaled(cen), boxsize=self._tree_box)
        k = min(12, len(self.triangles))
        q = self.wrap(pos)
        _, cand = self._centroid_tree.query(self._scaled(q), k=k)
        inside = np.zeros(len(pos), dtype=bool)
        for c in range(k):
            t = tri[cand[:, c]]  # (m, 3, 2) unwrapped near vertex 0
            p = t[:, 0] + self.minimum_image(q - t[:, 0])
            a, b, cc = t[:, 0], t[:, 1], t[:, 2]
            v0, v1, v2 = b - a, cc - a, p - a
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
            l0 = 1.0 - l1 - l2
            eps = tol * 10
            inside |= (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        return inside


@dataclass
class UnstructuredGrid:
    """A stack of poloidal planes along y.

    ``poloidal`` is either one :class:`UnstructuredPoloidalGrid` (used for
    every plane) or a list of ``ny`` grids with the same ``n_nodes`` (and,
    with ``dirichlet_walls``, the same wall nodes); their triangulations may
    differ (e.g. from :func:`gmsh_planes`).
    """

    poloidal: object
    ycoords: np.ndarray
    Ly: float
    yperiodic: bool = True
    walls_y: bool = False
    MXG: int = 1
    MYG: int = 1
    multiple_of: int = 4
    dirichlet_walls: bool = False
    _planes: list = dc_field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self.ycoords = np.asarray(self.ycoords, dtype=float)
        ny = len(self.ycoords)
        if isinstance(self.poloidal, UnstructuredPoloidalGrid):
            self._planes = [self.poloidal] * ny
        else:
            self._planes = list(self.poloidal)
            if len(self._planes) != ny:
                raise ValueError("Need one poloidal grid per y coordinate")
        n = {p.n_nodes for p in self._planes}
        if len(n) != 1:
            raise ValueError("All planes must have the same number of nodes")
        self.n_nodes = n.pop()
        self._layout()
        if self.n_interior % self.multiple_of != 0:
            # Only a restriction of the processor split: BOUT++ needs NXPE to
            # divide the number of evolving nodes (and checks it at start-up)
            valid = [d for d in range(1, 129) if self.n_interior % d == 0]
            warnings.warn(
                f"{self.n_interior} evolving nodes is not a multiple of "
                f"{self.multiple_of}: BOUT++ can only use NXPE in {valid} "
                "(NXPE must divide the number of evolving nodes). To allow "
                f"NXPE = {self.multiple_of}, mesh with "
                f"gmsh_planes(..., multiple_of={self.multiple_of}, "
                f"dirichlet_walls={self.dirichlet_walls})",
                stacklevel=2,
            )
        if self.MYG != 1:
            raise ValueError("Only MYG = 1 is supported for unstructured grids")

    @property
    def ny(self):
        return len(self.ycoords)

    def _layout(self):
        """x index of every node (``node_x``) and which nodes evolve.

        Without Dirichlet walls node k is at ``MXG + k``. With
        ``dirichlet_walls`` the wall nodes go into the x guard cells:
        inner-wall nodes at the start, outer-wall nodes at the end. ``MXG``
        is the larger of the two counts. The walls are the planes' marked
        ``walls`` (any shape, e.g. from :meth:`UnstructuredPoloidalGrid.from_gmsh`)
        or, if unmarked, the lines of minimum and maximum R.
        BOUT++ does not evolve guard cells and sets them with the
        ``dirichlet_node`` boundary condition.
        """
        n = self.n_nodes
        if not self.dirichlet_walls:
            self.node_x = self.MXG + np.arange(n)
            self.interior = np.ones(n, dtype=bool)
        else:
            pol = self._planes[0]
            if pol.walls is not None:
                # Marked walls (e.g. from_gmsh): curved, any shape
                inner, outer = (np.sort(w) for w in pol.walls)
                for other in self._planes[1:]:
                    if other.walls is None or not (
                        np.array_equal(np.sort(other.walls[0]), inner)
                        and np.array_equal(np.sort(other.walls[1]), outer)
                    ):
                        raise ValueError("All planes must have the same wall nodes")
            else:
                if np.isfinite(pol.period[0]):
                    raise ValueError(
                        "dirichlet_walls needs a domain that is not periodic in R"
                    )
                tol = 1e-9 * np.ptp(pol.R)
                inner = np.flatnonzero(pol.R <= pol.R.min() + tol)
                outer = np.flatnonzero(pol.R >= pol.R.max() - tol)
                for other in self._planes[1:]:
                    if not (
                        np.array_equal(
                            np.flatnonzero(other.R <= pol.R.min() + tol), inner
                        )
                        and np.array_equal(
                            np.flatnonzero(other.R >= pol.R.max() - tol), outer
                        )
                    ):
                        raise ValueError("All planes must have the same wall nodes")
            self.MXG = max(len(inner), len(outer))
            self.interior = np.ones(n, dtype=bool)
            self.interior[inner] = False
            self.interior[outer] = False
            n_int = int(self.interior.sum())
            self.node_x = np.empty(n, dtype=int)
            self.node_x[inner] = np.arange(len(inner))
            self.node_x[self.interior] = self.MXG + np.arange(n_int)
            self.node_x[outer] = self.MXG + n_int + np.arange(len(outer))
        self.n_interior = int(self.interior.sum())
        # For guard-cell padding: copy from the nearest occupied x index
        occupied = np.zeros(self.nx, dtype=bool)
        occupied[self.node_x] = True
        xs = np.flatnonzero(occupied)
        nearest = xs[
            np.argmin(np.abs(np.arange(self.nx)[:, None] - xs[None, :]), axis=1)
        ]
        node_at = np.full(self.nx, -1)
        node_at[self.node_x] = np.arange(n)
        self._fill_node = node_at[nearest]

    @property
    def nx(self):
        return self.n_interior + 2 * self.MXG

    @property
    def shape(self):
        return (self.nx, self.ny, 1)

    def plane(self, j):
        return self._planes[j]

    def dy(self):
        if self.ny == 1:
            return np.array([self.Ly])
        y = np.concatenate(
            (self.ycoords - self.Ly, self.ycoords, self.ycoords + self.Ly)
        )
        return np.gradient(y)[self.ny : 2 * self.ny]

    def to3d(self, per_plane):
        """Place per-plane node arrays ``(ny, n_nodes)`` into ``(nx, ny, 1)``
        with guard cells copied from the nearest node."""
        per_plane = np.asarray(per_plane)
        out = np.empty(self.shape, dtype=per_plane.dtype)
        out[:, :, 0] = per_plane[:, self._fill_node].T
        return out


# ----------------------------------------------------------------------
# Field-line maps


def trace_maps(
    grid, field, rtol=1e-10, field_tracer=None, domain_tol=1e-9, near_wall_hits=False
):
    """Trace field lines from every node to the neighbouring planes.

    Returns a dict with per-plane arrays of shape ``(ny, n_nodes)``:
    ``forward_R``, ``forward_Z``, ``forward_hit`` (0 inside, -1 inner wall,
    +1 outer wall), ``forward_half_R``/``forward_half_Z`` (positions half
    way to the next plane), and the same for ``backward``.

    End points outside the target plane's mesh are wall hits. On planes with
    marked ``walls`` the hit is on the wall with the nearest wall node,
    otherwise on the outer wall if R is above the mesh centre. End points
    within about ``domain_tol`` node spacings outside the mesh count as
    inside (their stencils extrapolate slightly), so that round-off at the
    mesh edge does not create wall hits. With ``near_wall_hits`` (planes with
    marked walls only), end points in a triangle with a wall node (between
    the last plasma nodes and the wall) are wall hits too, as end points
    beyond the last cell centre on structured grids, so that the sheath at
    the parallel boundary takes them instead of an extrapolation next to the
    wall.
    """
    tracer = field_tracer or fieldtracer.FieldTracer(field)
    ny, n = grid.ny, grid.n_nodes
    dy = grid.dy()
    out = {}
    plasma_meshes = {}  # planes without the triangles that touch a wall
    for name, sign in (("forward", +1), ("backward", -1)):
        Rp = np.zeros((ny, n))
        Zp = np.zeros((ny, n))
        Rh = np.zeros((ny, n))
        Zh = np.zeros((ny, n))
        hit = np.zeros((ny, n), dtype=int)
        for j in range(ny):
            pol = grid.plane(j)
            y0 = grid.ycoords[j]
            y1 = y0 + sign * dy[j]
            traced = np.array(
                tracer.follow_field_lines(
                    pol.dof[:, 0],
                    pol.dof[:, 1],
                    [y0, y0 + 0.5 * sign * dy[j], y1],
                    rtol=rtol,
                )
            )
            Rh[j], Zh[j] = traced[1, :, 0], traced[1, :, 1]
            end = traced[-1]
            jn = j + sign
            if not grid.yperiodic and (jn < 0 or jn >= ny):
                # Past the end of a non-periodic domain
                Rp[j], Zp[j] = end[:, 0], end[:, 1]
                hit[j] = +1
                continue
            target = grid.plane(jn % ny)
            pos = target.wrap(end)
            Rp[j], Zp[j] = pos[:, 0], pos[:, 1]
            inside = target.in_domain(pos, tol=domain_tol)
            if target.walls is not None and len(target.walls[1]):
                w_in, w_out = target.walls
                wall_nodes = np.concatenate([w_in, w_out])
                _, near = KDTree(target.points[wall_nodes]).query(pos)
                outer = near >= len(w_in)
                if near_wall_hits:
                    if id(target) not in plasma_meshes:
                        on_wall = np.zeros(target.n_nodes, dtype=bool)
                        on_wall[wall_nodes] = True
                        sub = copy.copy(target)
                        for key in ("_centroid_tree", "_centroids"):
                            sub.__dict__.pop(key, None)
                        sub.triangles = target.triangles[
                            ~on_wall[target.triangles].any(axis=1)
                        ]
                        plasma_meshes[id(target)] = sub
                    inside &= plasma_meshes[id(target)].in_domain(pos, tol=domain_tol)
            else:
                centre = target.points.mean(axis=0)
                outer = pos[:, 0] >= centre[0]
            hit[j] = np.where(inside, 0, np.where(outer, 1, -1))
        if grid.walls_y:
            last = ny - 1 if sign > 0 else 0
            hit[last] = +1
        out[f"{name}_R"] = Rp
        out[f"{name}_Z"] = Zp
        out[f"{name}_half_R"] = Rh
        out[f"{name}_half_Z"] = Zh
        out[f"{name}_hit"] = hit
    return out


# ----------------------------------------------------------------------
# Parallel (support operator method) weights


def som_weights(
    grid,
    maps,
    order=4,
    strict=True,
    snap_tol=1e-10,
    extra=16,
    weight_power=3.0,
    map_walls=True,
):
    """Cell numbering, leg metadata and CSR operators for ``PetscOperators``.

    Uses the numbering of :func:`zoidberg.weights.assign_cell_space_numbers`:
    interior cells first (loop order x, y), then (none) X-boundary cells,
    then for each interior cell the backward and forward parallel boundary
    virtual cells. Legs are single-sample. With ``map_walls=False`` the
    interpolation stencils use only evolving nodes, so the values at
    Dirichlet wall nodes do not enter the parallel slices.
    """
    nx, ny, _ = grid.shape
    node_x = grid.node_x
    by_x = np.argsort(node_x)
    evolving = by_x[grid.interior[by_x]]
    walls = by_x[~grid.interior[by_x]]

    cell_number = np.full((nx, ny, 1), -1, dtype=int)
    next_cell = 0
    for k in evolving:
        for y in range(ny):
            cell_number[node_x[k], y, 0] = next_cell
            next_cell += 1
    n_evolving = next_cell
    # Dirichlet wall nodes (x boundary cells), inner then outer
    for k in walls:
        for y in range(ny):
            cell_number[node_x[k], y, 0] = next_cell
            next_cell += 1

    fwd_bnd = np.full((nx, ny, 1), -1, dtype=int)
    bwd_bnd = np.full((nx, ny, 1), -1, dtype=int)
    for k in evolving:
        for y in range(ny):
            if maps["backward_hit"][y, k] != 0:
                bwd_bnd[node_x[k], y, 0] = next_cell
                next_cell += 1
            if maps["forward_hit"][y, k] != 0:
                fwd_bnd[node_x[k], y, 0] = next_cell
                next_cell += 1

    result = {
        "total_cells": next_cell,
        "cell_number": cell_number,
        "forward_cell_number": fwd_bnd,
        "backward_cell_number": bwd_bnd,
    }
    stats = {"n_evolving": n_evolving, "fallback": 0, "snapped": 0}

    for name, sign, bnd in (("forward", +1, fwd_bnd), ("backward", -1, bwd_bnd)):
        hit = maps[f"{name}_hit"]
        interior_leg = np.full((nx, ny, 1), -1, dtype=int)
        boundary_leg = np.full((nx, ny, 1), -1, dtype=int)
        next_leg = 0
        for k in evolving:
            for y in range(ny):
                if hit[y, k] == 0:
                    interior_leg[node_x[k], y, 0] = next_leg
                else:
                    boundary_leg[node_x[k], y, 0] = next_leg
                next_leg += 1
        n_legs = next_leg

        rows = [None] * n_legs
        # Small meshes (e.g. the 1D test grids) cannot provide the full
        # stencil; use as many nodes as there are
        n_min = min(grid.plane(j).n_nodes for j in range(ny))
        if num_coefs(order) + extra > n_min:
            warnings.warn(
                f"Parallel-map stencils reduced to {n_min} nodes "
                f"(mesh too small for num_coefs({order}) + {extra})"
            )
            extra = max(n_min - num_coefs(order), 0)
        # Stencils, computed per target plane in one batch
        for y in range(ny):
            yt = (y + sign) % ny
            target = grid.plane(yt)
            src = np.flatnonzero((hit[y] == 0) & grid.interior)
            if len(src):
                pos = np.column_stack(
                    (maps[f"{name}_R"][y, src], maps[f"{name}_Z"][y, src])
                )
                inds, w, info = target.lsq_weights(
                    pos,
                    order=order,
                    extra=extra,
                    weight_power=weight_power,
                    strict=strict,
                    snap_tol=snap_tol,
                    nodes=None if map_walls else np.flatnonzero(grid.interior),
                )
                stats["fallback"] += info["fallback"]
                stats["snapped"] += info["snapped"]
                for s, ii, ww in zip(src, inds, w):
                    keep = np.abs(ww) > 1e-14
                    cols = cell_number[node_x[ii[keep]], yt, 0]
                    rows[interior_leg[node_x[s], y, 0]] = (cols, ww[keep])
            for s in np.flatnonzero((hit[y] != 0) & grid.interior):
                rows[boundary_leg[node_x[s], y, 0]] = (
                    np.array([bnd[node_x[s], y, 0]]),
                    np.array([1.0]),
                )
        ptr = np.zeros(n_legs + 1, dtype=int)
        ptr[1:] = np.cumsum([len(r[0]) for r in rows])
        cols = np.concatenate([r[0] for r in rows]).astype(int)
        wts = np.concatenate([r[1] for r in rows])
        if np.any(cols < 0):
            raise ValueError(f"{name}: negative column index in stencil")
        result.update(
            {
                f"n_{name}_legs": n_legs,
                f"{name}_leg_interior_number": interior_leg,
                f"{name}_leg_boundary_number": boundary_leg,
                f"{name}_leg_weights": np.ones(n_legs),
                f"{name}_rows": ptr,
                f"{name}_columns": cols,
                f"{name}_weights": wts,
            }
        )
    return result, stats


# ----------------------------------------------------------------------
# Perpendicular diffusion operator (poloidal-plane form)


def _max_real_eigenvalue(L, dense_limit=4096):
    """Largest real part of the eigenvalues of the sparse matrix ``L``,
    relative to its infinity norm. Dense for small matrices, ARPACK
    otherwise."""
    from scipy.sparse.linalg import ArpackNoConvergence, eigs

    scale = abs(L).sum(axis=1).max()
    n = L.shape[0]
    if n <= dense_limit:
        ev = np.linalg.eigvals(L.toarray())
    else:
        try:
            ev = eigs(L, k=6, which="LR", return_eigenvectors=False, maxiter=20 * n)
        except ArpackNoConvergence as err:
            ev = err.eigenvalues
    return float(np.max(ev.real)) / scale


def _drop_trees(pol):
    """Drop the search trees of ``pol``, which depend on its scaling ``pol.h``."""
    pol._kdtree = None
    for attr in ("_centroid_tree", "_centroids"):
        if hasattr(pol, attr):
            delattr(pol, attr)


def _plane_dagp(
    pol,
    order=3,
    strict=True,
    stabilisation=1.0,
    extra=16,
    max_stabilisation=8.0,
    stability_tol=1e-10,
    cylindrical=False,
    value_order=None,
    value_extra=4,
):
    """Per-plane CSR blocks (local node / face-point numbering).

    Each face contributes two Gauss points, where the flux is
    ``a (S . grad f)`` with least-squares gradients, plus one stabilisation
    point at the edge midpoint ``m`` with flux

        beta * c * [(f_j - f_i) - d . grad f(m)],   c = |S|^2 / (d . S)

    (S: face normal vector, d = x_j - x_i). The bracket is O(h^3) for smooth
    f, so accuracy is unchanged, but the compact difference damps
    grid-scale modes. Without it the operator has eigenvalues with positive
    real part on irregular meshes, so explicit or implicit time stepping
    diverges. Starting from ``beta = stabilisation``, beta is doubled (up to
    ``max_stabilisation``) until the operator with a = 1 has no eigenvalue
    with real part above ``stability_tol`` times its norm.

    The fits use ``num_coefs(order) + extra`` nodes (default ``extra = 16``;
    order 4 is the default from :func:`build_grid_data`).
    Compared with the minimal stencils of the parallel maps, the larger
    stencils cut the coarse-grid error on anisotropic meshes by more than 10x
    and remove the remaining growing modes. Larger ``beta`` alone does not
    achieve that, and it degrades accuracy on coarse grids, hence the cap.
    The fits are unweighted (distance weighting makes stability worse).

    With ``cylindrical=True`` (fields with a major radius) volumes and face
    fluxes are weighted by R, giving (1/R) d/dR (R a df/dR) + d/dZ (a df/dZ).

    The least-squares fits use physical distances. Scaling R and Z
    separately by the node spacings (as for the parallel maps) amplifies
    errors on strongly anisotropic meshes, where the gradient across thin
    layers is divided by the small spacing.
    """
    from scipy.sparse import csr_matrix

    # Small meshes (e.g. the 1D test grids) cannot provide the full stencil;
    # use as many nodes as there are
    max_extra = pol.n_nodes - num_coefs(order)
    if extra > max_extra:
        warnings.warn(
            f"Perpendicular-operator stencils reduced to {pol.n_nodes} nodes "
            f"(mesh too small for num_coefs({order}) + {extra})"
        )
        extra = max(max_extra, 0)

    saved_h = pol.h.copy()
    pol.h = np.full(2, np.sqrt(saved_h[0] * saved_h[1]))
    _drop_trees(pol)
    try:
        f = pol.faces()
        vol = pol.dual_volumes(cylindrical=cylindrical)
        tq, wq = _GAUSS2
        nq = len(tq)
        n_faces = len(f["i"])
        s, e = f["start"], f["end"]
        d = e - s  # (n_faces, 2)
        # normal (dZ, -dR) has length |d|; quadrature weight includes length
        normal = np.column_stack((d[:, 1], -d[:, 0]))
        xi = pol.points[f["i"]]
        fp = xi[:, None, :] + s[:, None, :] + tq[None, :, None] * d[:, None, :]
        fp = fp.reshape(-1, 2)
        lsq = {
            "order": order,
            "extra": extra,
            "gradient": True,
            "snap_tol": None,
            "strict": strict,
        }
        inds, w_val, w_dR, w_dZ, info = pol.lsq_weights(pol.wrap(fp), **lsq)
        nvec = np.repeat(normal, nq, axis=0) * np.tile(wq, n_faces)[:, None]
        if cylindrical:
            nvec = nvec * pol.wrap(fp)[:, :1]  # R at each face point
        w_grad = nvec[:, :1] * w_dR + nvec[:, 1:] * w_dZ
        fi = np.repeat(f["i"], nq)
        fj = np.repeat(f["j"], nq)
        # Face-point positions relative to node i, and x_j - x_i (for flows)
        edge = pol.minimum_image(pol.points[f["j"]] - xi)
        pos_i = (s[:, None, :] + tq[None, :, None] * d[:, None, :]).reshape(-1, 2)
        edge_fp = np.repeat(edge, nq, axis=0)
        if pol.centroid_dofs:
            # Unknowns at the dual-cell centroids: the two-point
            # stabilisation connects the unknowns, and reconstructions start
            # from them. shift = node - unknown (relative positions)
            dof = pol.dof
            shift = pol.minimum_image(pol.points - dof)
            edge_s = pol.minimum_image(dof[f["j"]] - dof[f["i"]])
            mid_s = pol.wrap(dof[f["i"]] + 0.5 * edge_s)
        else:
            shift = None
            edge_s = edge
            mid_s = None
        fallback = info["fallback"]
        beta = 0.0
        max_re = None

        if stabilisation:
            mid = pol.wrap(xi + 0.5 * edge) if mid_s is None else mid_s
            c = np.sum(normal**2, axis=1) / np.sum(edge_s * normal, axis=1)
            if cylindrical:
                c = c * mid[:, 0]
            # Prediction P(x_j) - P(x_i) of the order-p fit around the
            # midpoint: the bracket is O(h^(p+1))
            half = 0.5 * edge_s[:, None, :]
            m_inds, m_val, _, _, m_eval, m_info = pol.lsq_weights(
                mid, eval_offsets=np.concatenate([-half, half], axis=1), **lsq
            )
            m_grad = -c[:, None] * (m_eval[:, 1, :] - m_eval[:, 0, :])
            fallback += m_info["fallback"]
            # Append the two-point difference as two extra columns
            k = max(inds.shape[1], m_inds.shape[1]) + 2

            def pad(a, width, fill):
                out = np.full((a.shape[0], width), fill, dtype=a.dtype)
                out[:, : a.shape[1]] = a
                return out

            s_inds = pad(m_inds, k, 0)
            s_inds[:, -2] = f["i"]
            s_inds[:, -1] = f["j"]
            s_grad = pad(m_grad, k, 0.0)  # for beta = 1
            s_grad[:, -2] -= c
            s_grad[:, -1] += c
            s_val = pad(m_val, k, 0.0)
            inds = np.vstack([pad(inds, k, 0), s_inds])
            w_grad_q = pad(w_grad, k, 0.0)
            w_val = np.vstack([pad(w_val, k, 0.0), s_val])
            fi = np.concatenate([fi, f["i"]])
            fj = np.concatenate([fj, f["j"]])
            if shift is None:
                pos_i = np.vstack([pos_i, 0.5 * edge])
            else:
                pos_i = np.vstack([pos_i, 0.5 * edge_s - shift[f["i"]]])
            edge_fp = np.vstack([edge_fp, edge])

            n, nfp = pol.n_nodes, len(fi)
            rows = np.repeat(np.arange(nfp), inds.shape[1])
            div = csr_matrix(
                (
                    np.concatenate([1.0 / vol[fi], -1.0 / vol[fj]]),
                    (np.concatenate([fi, fj]), np.tile(np.arange(nfp), 2)),
                ),
                shape=(n, nfp),
            )
            beta = stabilisation
            while True:
                w_grad = np.vstack([w_grad_q, beta * s_grad])
                grad = csr_matrix(
                    (w_grad.ravel(), (rows, inds.ravel())), shape=(nfp, n)
                )
                max_re = _max_real_eigenvalue(div @ grad)
                if max_re <= stability_tol or 2 * beta > max_stabilisation:
                    break
                beta *= 2
            if max_re > stability_tol:
                warnings.warn(
                    f"Perpendicular operator still has growing modes "
                    f"(max Re = {max_re:.2g} * norm) with beta = {beta}"
                )

        # Upwind reconstruction of a at each face point from either side:
        # order-p fit around the side's unknown, evaluated at the face point,
        # O(h^(p+1)). (A linear reconstruction is only O(h^2) per face point,
        # which on irregular meshes gives an O(h) truncation error once
        # upwinding is on.)
        lsq_v = dict(lsq, gradient=False)

        def recon(node, offset):
            r_inds, _, w_e, r_info = pol.lsq_weights(
                pol.dof[node], eval_offsets=offset[:, None, :], **lsq_v
            )
            return r_inds, w_e[:, 0, :], r_info["fallback"]

        # Offsets from the unknowns (nodes or dual-cell centroids)
        off_l, off_r = pos_i, pos_i - edge_fp
        if shift is not None:
            off_l, off_r = off_l + shift[fi], off_r + shift[fj]
        left_inds, left_w, fb_l = recon(fi, off_l)
        right_inds, right_w, fb_r = recon(fj, off_r)
        fallback += fb_l + fb_r

        # Face values of the coefficient a. By default they come from the
        # gradient fit; a separate lower-order fit is enough for second order
        # (the flux error (delta a) S.grad f is O(h^(q+2)) per face for an
        # order-q value fit) and needs far fewer nodes
        v_inds = v_val = None
        if value_order is not None:
            pts = pol.wrap(fp)
            if stabilisation:
                pts = np.vstack([pts, mid])
            v_inds, v_val, v_info = pol.lsq_weights(
                pts,
                order=value_order,
                extra=min(value_extra, max(pol.n_nodes - num_coefs(value_order), 0)),
                snap_tol=None,
                strict=strict,
            )
            fallback += v_info["fallback"]
    finally:
        pol.h = saved_h
        _drop_trees(pol)
    owner_node = np.minimum(fi, fj)

    return {
        "n_fp": len(fi),
        "inds": inds,
        "w_val": w_val,
        "w_grad": w_grad,
        "v_inds": v_inds,
        "v_val": v_val,
        "fi": fi,
        "fj": fj,
        "owner_node": owner_node,
        "pos_i": pos_i,
        "edge": edge_fp,
        "left_inds": left_inds,
        "left_w": left_w,
        "right_inds": right_inds,
        "right_w": right_w,
        "vol": vol,
        "fallback": fallback,
        "beta": beta,
        "max_re": max_re,
    }


def _csr(n_rows, row_idx, col_idx, vals):
    """CSR arrays with sorted, unique columns in each row.

    Duplicate (row, column) entries are summed: BOUT++ inserts the entries
    into PETSc with INSERT_VALUES, which would keep only one of them.
    """
    order = np.lexsort((col_idx, row_idx))
    row_idx, col_idx, vals = row_idx[order], col_idx[order], vals[order]
    if len(row_idx):
        new = np.ones(len(row_idx), dtype=bool)
        new[1:] = (row_idx[1:] != row_idx[:-1]) | (col_idx[1:] != col_idx[:-1])
        group = np.cumsum(new) - 1
        vals = np.bincount(group, weights=vals)
        row_idx, col_idx = row_idx[new], col_idx[new]
    ptr = np.zeros(n_rows + 1, dtype=int)
    np.add.at(ptr, row_idx + 1, 1)
    np.cumsum(ptr, out=ptr)
    return ptr, col_idx.astype(int), vals


def dagp_unstructured(
    grid,
    cell_number,
    total_cells,
    order=3,
    strict=True,
    processes=1,
    perp_walls=True,
    **plane_options,
):
    """Global CSR matrices for ``FV::Div_a_Grad_perp_FCI_unstructured``.

    Only evolving nodes get a divergence row. Face points between two
    Dirichlet wall nodes are dropped; faces between a wall node and an
    evolving node use the wall value, which BOUT++ holds in the guard cell.
    With ``perp_walls=False`` those faces are dropped too, so that no flux
    crosses them (insulating walls).

    ``processes > 1`` computes the planes in that many worker processes (at
    most ``processes`` planes ahead of the assembly, so that memory stays
    bounded); the result is the same as with one process.
    """
    interior = grid.interior
    # Planes are processed one at a time and their blocks released after use
    # (planes that share one UnstructuredPoloidalGrid are computed once and
    # kept until their last use). The face-point rows of the gradient, value
    # and reconstruction matrices are contiguous per plane, so their CSR
    # arrays are built per plane and concatenated; this gives the same arrays
    # as one global assembly, without holding the padded triplets of all
    # planes at once.
    remaining = Counter(id(grid.plane(y)) for y in range(grid.ny))
    cache = {}
    plane_kwargs = dict(order=order, strict=strict, **plane_options)
    # First plane index of each distinct poloidal grid, in order of use
    first_use = sorted(
        {id(grid.plane(y)): y for y in reversed(range(grid.ny))}.values()
    )
    pending = {}
    executor = None
    if processes > 1 and len(first_use) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=processes, mp_context=multiprocessing.get_context("fork")
        )
    next_submit = 0

    def submit_ahead():
        nonlocal next_submit
        while next_submit < len(first_use) and len(pending) < processes:
            y_s = first_use[next_submit]
            pending[id(grid.plane(y_s))] = executor.submit(
                _plane_dagp, grid.plane(y_s), **plane_kwargs
            )
            next_submit += 1

    stats = {"fallback": 0, "beta": -np.inf, "max_re": -np.inf}
    row_mats = {name: ([], [], []) for name in ("gradp", "value", "left", "right")}
    dr, dc, dv = [], [], []
    fr, fc, fR, fZ = [], [], [], []  # cell-averaged flux vector
    owners = []
    volume = np.zeros((grid.ny, grid.n_nodes))
    offset = 0

    def add_rows(name, inds, w, cells):
        """Per-plane CSR block of a matrix with one row per face point."""
        ptrs, cols, vals = row_mats[name]
        nfp, k = inds.shape
        r = np.repeat(np.arange(nfp), k)
        c = cells[inds].ravel()
        v = w.ravel()
        keep = np.abs(v) > 1e-300
        ptr, col, val = _csr(nfp, r[keep], c[keep], v[keep])
        ptrs.append(ptr[1:] + sum(len(a) for a in cols))
        cols.append(col)
        vals.append(val)

    try:
        for y in range(grid.ny):
            pol = grid.plane(y)
            key = id(pol)
            if key in cache:
                b = cache[key]
            else:
                if executor is not None:
                    submit_ahead()
                    b = pending.pop(key).result()
                    submit_ahead()
                else:
                    b = _plane_dagp(pol, **plane_kwargs)
                stats["fallback"] += b["fallback"]
                stats["beta"] = max(stats["beta"], b["beta"])
                stats["max_re"] = max(stats["max_re"], b["max_re"] or 0.0)
            remaining[key] -= 1
            if remaining[key] > 0:
                cache[key] = b
            else:
                cache.pop(key, None)

            keep = interior[b["fi"]] | interior[b["fj"]]
            if not perp_walls:
                keep = interior[b["fi"]] & interior[b["fj"]]
            cells = cell_number[grid.node_x, y, 0]
            nfp = int(np.sum(keep))
            rows = offset + np.arange(nfp)
            fi, fj = b["fi"][keep], b["fj"][keep]
            add_rows("gradp", b["inds"][keep], b["w_grad"][keep], cells)
            if b["v_inds"] is None:
                add_rows("value", b["inds"][keep], b["w_val"][keep], cells)
            else:
                add_rows("value", b["v_inds"][keep], b["v_val"][keep], cells)
            add_rows("left", b["left_inds"][keep], b["left_w"][keep], cells)
            add_rows("right", b["right_inds"][keep], b["right_w"][keep], cells)
            mi, mj = interior[fi], interior[fj]
            dr.append(cells[fi][mi])
            dc.append(rows[mi])
            dv.append(1.0 / b["vol"][fi][mi])
            dr.append(cells[fj][mj])
            dc.append(rows[mj])
            dv.append(-1.0 / b["vol"][fj][mj])
            # The physical flux is -a grad f, so the outward flux of node i through
            # this face point is -F (F = a S.grad f) and that of node j is +F.
            # Cell average of the flux vector: sum (x_f - x_node) * outward / V
            pos_i, pos_j = b["pos_i"][keep], b["pos_i"][keep] - b["edge"][keep]
            vi, vj = b["vol"][fi], b["vol"][fj]
            fr += [cells[fi][mi], cells[fj][mj]]
            fc += [rows[mi], rows[mj]]
            fR += [-pos_i[mi, 0] / vi[mi], pos_j[mj, 0] / vj[mj]]
            fZ += [-pos_i[mi, 1] / vi[mi], pos_j[mj, 1] / vj[mj]]
            # Owned with an evolving node
            owner_node = np.where(mi & mj, np.minimum(fi, fj), np.where(mi, fi, fj))
            owners.append(cells[owner_node])
            volume[y] = b["vol"]
            offset += nfp
            del b
    finally:
        if executor is not None:
            executor.shutdown()
    n_fp_total = offset

    def build(r, c, v, n_rows):
        r, c, v = np.concatenate(r), np.concatenate(c), np.concatenate(v)
        keep = np.abs(v) > 1e-300
        return _csr(n_rows, r[keep], c[keep], v[keep])

    out = {
        "dagp_us_n_facepoints": n_fp_total,
        "dagp_us_facepoint_owner": np.concatenate(owners).astype(int),
    }
    for name in ("gradp", "value", "left", "right"):
        ptrs, cols, vals = row_mats.pop(name)
        out[f"dagp_us_{name}_rows"] = np.concatenate([np.zeros(1, dtype=int)] + ptrs)
        del ptrs
        out[f"dagp_us_{name}_columns"] = np.concatenate(cols)
        del cols
        out[f"dagp_us_{name}_weights"] = np.concatenate(vals)
        del vals
        out[f"dagp_us_{name}_ncols"] = total_cells
    for name, (r, c, v, nrow, ncol) in {
        "div": (dr, dc, dv, total_cells, n_fp_total),
        "flowR": (fr, fc, fR, total_cells, n_fp_total),
        "flowZ": (fr, fc, fZ, total_cells, n_fp_total),
    }.items():
        ptr, cols, vals = build(r, c, v, nrow)
        out[f"dagp_us_{name}_rows"] = ptr
        out[f"dagp_us_{name}_columns"] = cols
        out[f"dagp_us_{name}_weights"] = vals
        out[f"dagp_us_{name}_ncols"] = ncol
    out["dagp_us_volume"] = grid.to3d(volume)
    return out, stats


# ----------------------------------------------------------------------
# Metric


def is_cylindrical(field):
    """True for fields in cylindrical coordinates (y is the toroidal angle)."""
    return field.Rfunc(np.ones(1), np.zeros(1), 0.0) is not None


def slab_metric(grid, field, maps):
    """Metric; see the module docstring.

    Cartesian fields (Slab type): ``g_22 = (|B|/B_y)^2``. Cylindrical fields
    (``Rfunc`` gives the major radius, y is the toroidal angle):
    ``g_22 = (R |B|/B_y)^2``, as in :func:`zoidberg.zoidberg.get_metric`.
    """
    ny, n = grid.ny, grid.n_nodes
    shape = grid.shape
    dy = grid.dy()

    cylindrical = is_cylindrical(field)

    def bfield(R, Z, y):
        """|B| and B_y, with B_y divided by the major radius if cylindrical,
        so that (|B|/B_y)^2 is g_22."""
        By = field.Byfunc(R, Z, y)
        if cylindrical:
            By = By / field.Rfunc(R, Z, y)
        return field.Bmag(R, Z, y), By

    Bmag = np.zeros((ny, n))
    By = np.zeros((ny, n))
    for j in range(ny):
        pol = grid.plane(j)
        Bmag[j], By[j] = bfield(pol.dof[:, 0], pol.dof[:, 1], grid.ycoords[j])

    ones = np.ones(shape)
    zeros = np.zeros(shape)
    g_22 = grid.to3d((Bmag / By) ** 2)
    metric = {
        "dy": grid.to3d(np.repeat(dy[:, None], n, axis=1)),
        "dz": ones.copy(),
        "g11": ones.copy(),
        "g22": 1.0 / g_22,
        "g33": ones.copy(),
        "g12": zeros.copy(),
        "g13": zeros.copy(),
        "g23": zeros.copy(),
        "g_11": ones.copy(),
        "g_22": g_22,
        "g_33": ones.copy(),
        "g_12": zeros.copy(),
        "g_13": zeros.copy(),
        "g_23": zeros.copy(),
        "Bxy": grid.to3d(Bmag),
        "B": grid.to3d(Bmag),
        "d2x": zeros.copy(),
        "d2y": zeros.copy(),
        "d2z": zeros.copy(),
    }
    # dx and dz must be uniform (Coordinates asserts this for FCI). The
    # parallel finite-volume operators only use ratios of cell areas and
    # volumes, in which dx*dz cancels; the dual-cell areas are in
    # dagp_us_volume.
    metric["dx"] = ones.copy()
    metric["By"] = grid.to3d(
        By
        * (
            np.array([grid.plane(j).dof[:, 0] for j in range(ny)])
            if cylindrical
            else 1.0
        )
    )

    # Staggered (half-step) quantities, as written by zoidberg.make_maps
    for name, post, sign in (
        ("forward", "_cell_yhigh", +1),
        ("backward", "_cell_ylow", -1),
    ):
        Bh = np.zeros((ny, n))
        Byh = np.zeros((ny, n))
        for j in range(ny):
            Bh[j], Byh[j] = bfield(
                maps[f"{name}_half_R"][j],
                maps[f"{name}_half_Z"][j],
                grid.ycoords[j] + 0.5 * sign * dy[j],
            )
        # g_22 from B_y / R (as g_22 at the nodes); By_cell_* is B_y itself
        metric[f"g_22{post}"] = grid.to3d((Bh / Byh) ** 2)
        if cylindrical:
            Byh = Byh * maps[f"{name}_half_R"]
        metric[f"By{post}"] = grid.to3d(Byh)

    for name, sign in (("forward", +1), ("backward", -1)):
        Bm = np.zeros((ny, n))
        Byp = np.zeros((ny, n))
        for j in range(ny):
            Bm[j], Byp[j] = bfield(
                maps[f"{name}_R"][j],
                maps[f"{name}_Z"][j],
                grid.ycoords[j] + sign * dy[j],
            )
        pg_22 = grid.to3d((Bm / Byp) ** 2)
        metric.update(
            {
                f"{name}_g_11": ones.copy(),
                f"{name}_g_22": pg_22,
                f"{name}_g_33": ones.copy(),
                f"{name}_g_13": zeros.copy(),
                f"{name}_g11": ones.copy(),
                f"{name}_g22": 1.0 / pg_22,
                f"{name}_g33": ones.copy(),
                f"{name}_g13": zeros.copy(),
                f"{name}_dy": metric["dy"].copy(),
                f"{name}_Bxy": grid.to3d(Bm),
            }
        )
    return metric


# ----------------------------------------------------------------------
# Flux-reconstructed conservative parallel half-divergences ("K2")

#: Version of the K2 matrices in the grid file (``k2_version``)
K2_VERSION = 1


def _tri_area_signed(t):
    """Signed areas of triangles ``t[..., 3, 2]``."""
    a = t[..., 1, :] - t[..., 0, :]
    b = t[..., 2, :] - t[..., 0, :]
    return 0.5 * (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])


def _dual_triangles(pol):
    """Two triangles per dual face: ``(node i, start, end)`` owned by ``i``
    and ``(node j, end, start)`` owned by ``j``, counter-clockwise, unwrapped
    near the owning node. Their union per node is its dual cell. Returns
    ``(tri (N, 3, 2), owner (N,))``."""
    fc = pol.faces()
    pts = pol.points
    s, e = fc["start"], fc["end"]
    i, j = fc["i"], fc["j"]
    xj = pol.minimum_image(pts[j] - pts[i])
    pi = pts[i]
    t1 = np.stack([pi, pi + s, pi + e], axis=1)
    pj = pi + xj
    t2 = np.stack([pj, pi + e, pi + s], axis=1)
    tri = np.concatenate([t1, t2])
    own = np.concatenate([i, j])
    a = _tri_area_signed(tri)
    flip = a < 0
    tri[flip] = tri[flip][:, ::-1]
    keep = np.abs(a) > 1e-300
    return tri[keep], own[keep]


def _refine_triangles(tri, own, levels):
    """Split every triangle into ``4**levels`` similar triangles."""
    for _ in range(levels):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        tri = np.concatenate(
            [
                np.stack([a, ab, ca], 1),
                np.stack([ab, b, bc], 1),
                np.stack([ca, bc, c], 1),
                np.stack([ab, bc, ca], 1),
            ]
        )
        own = np.tile(own, 4)
    return tri, own


def _clip_tri_tri(S, D):
    """Intersection polygons of the counter-clockwise triangles ``S[m]`` and
    ``D[m]`` (vectorised Sutherland-Hodgman). Returns ``(P (M, 9, 2), cnt)``:
    the vertices, padded with the first one (so that the padding edges have
    zero length)."""
    M = len(S)
    maxv = 9
    P = np.zeros((M, maxv, 2))
    P[:, :3] = S
    cnt = np.full(M, 3)
    idx = np.arange(maxv)
    for e in range(3):
        a = D[:, e]
        ed = D[:, (e + 1) % 3] - a
        valid = idx[None, :] < cnt[:, None]
        # Signed distance, >= 0 inside the (counter-clockwise) clip triangle
        dist = ed[:, None, 0] * (P[..., 1] - a[:, None, 1]) - ed[:, None, 1] * (
            P[..., 0] - a[:, None, 0]
        )
        nxt = (idx[None, :] + 1) % np.maximum(cnt[:, None], 1)
        Pn = np.take_along_axis(P, nxt[..., None].repeat(2, axis=2), axis=1)
        dn = np.take_along_axis(dist, nxt, axis=1)
        inside = dist >= 0
        cross = valid & (inside != (dn >= 0))
        t = np.where(cross, dist / np.where(cross, dist - dn, 1.0), 0.0)
        X = P + t[..., None] * (Pn - P)
        # Emit the current vertex if inside, then the crossing point
        out = np.zeros((M, 2 * maxv, 2))
        keep = np.zeros((M, 2 * maxv), dtype=bool)
        out[:, 0::2] = P
        out[:, 1::2] = X
        keep[:, 0::2] = valid & inside
        keep[:, 1::2] = cross
        order = np.argsort(~keep, axis=1, kind="stable")
        P = np.take_along_axis(out, order[..., None].repeat(2, axis=2), axis=1)[
            :, :maxv
        ]
        cnt = np.minimum(keep.sum(axis=1), maxv)
    pad = idx[None, :] >= cnt[:, None]
    P = np.where(pad[..., None], P[:, :1], P)
    return P, cnt


def _poly_moments(P):
    """Exact moments ``[A, Mx, My, Mxx, Mxy, Myy]`` of the polygons
    ``P (M, V, 2)`` (counter-clockwise, padded with the first vertex)."""
    x, y = P[..., 0], P[..., 1]
    xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    c = x * yn - xn * y
    return np.column_stack(
        [
            0.5 * c.sum(1),
            (c * (x + xn)).sum(1) / 6.0,
            (c * (y + yn)).sum(1) / 6.0,
            (c * (x * x + x * xn + xn * xn)).sum(1) / 12.0,
            (c * (x * yn + 2 * x * y + 2 * xn * yn + xn * y)).sum(1) / 24.0,
            (c * (y * y + y * yn + yn * yn)).sum(1) / 12.0,
        ]
    )


def _k2_cell_moments(pol):
    """Moments of every dual cell relative to its unknown (dof)."""
    t, o = _dual_triangles(pol)
    rel0 = pol.minimum_image(t[:, 0] - pol.dof[o])
    m = _poly_moments(t - t[:, :1] + rel0[:, None])
    out = np.zeros((pol.n_nodes, 6))
    np.add.at(out, o, m)
    return out


def _periodic_shifts(pol):
    """Translations by -1, 0, +1 periods in each periodic direction."""
    shifts = [np.zeros(2)]
    for dim in range(2):
        p = pol.period[dim]
        if np.isfinite(p):
            unit = np.zeros(2)
            unit[dim] = p
            shifts = [s + k * unit for s in shifts for k in (-1.0, 0.0, 1.0)]
    return np.array(shifts)


def _k2_pieces(src, td, od, chunk=50000):
    """Overlaps of the dual cells of ``src`` with the destination dual
    triangles ``td`` (owners ``od``), which are already pulled back into the
    source plane.

    Returns ``(o, k, mom, contains, pulled_area)``: source node, destination
    node, moments of each piece relative to the source dof, whether the
    piece contains the source dof, and the pulled-back area of every
    destination cell.
    """
    import shapely

    ts, os_ = _dual_triangles(src)
    td = np.array(td, dtype=float, copy=True)
    a = _tri_area_signed(td)
    pulled_area = np.bincount(od, weights=np.abs(a), minlength=int(od.max()) + 1)
    flip = a < 0
    td[flip] = td[flip][:, ::-1]
    # Wrap each pulled-back triangle into the periodic box by its first
    # vertex, then add the periodic copies
    wrapped0 = src.wrap(td[:, 0])
    td += (wrapped0 - td[:, 0])[:, None, :]
    shifts = _periodic_shifts(src)
    td = np.concatenate([td + s for s in shifts])
    od = np.tile(od, len(shifts))
    bs = shapely.box(
        ts[..., 0].min(1), ts[..., 1].min(1), ts[..., 0].max(1), ts[..., 1].max(1)
    )
    bd = shapely.box(
        td[..., 0].min(1), td[..., 1].min(1), td[..., 0].max(1), td[..., 1].max(1)
    )
    ii, kk = shapely.STRtree(bd).query(bs)
    del bs, bd
    X = src.dof
    outo, outk, outm, outc = [], [], [], []
    for c0 in range(0, len(ii), chunk):
        i_, k_ = ii[c0 : c0 + chunk], kk[c0 : c0 + chunk]
        o = os_[i_]
        # Coordinates relative to the source dof
        rel0 = src.minimum_image(ts[i_, 0] - X[o])
        S = ts[i_] - ts[i_, :1] + rel0[:, None]
        D = td[k_] - ts[i_, :1] + rel0[:, None]
        P, cnt = _clip_tri_tri(S, D)
        m = _poly_moments(P)
        good = (cnt >= 3) & (m[:, 0] > 1e-14 * np.abs(_tri_area_signed(S)))
        # The pieces are convex: the dof (origin) is inside if it is on the
        # left of every edge
        Pn = np.roll(P, -1, axis=1)
        crs = P[..., 0] * Pn[..., 1] - P[..., 1] * Pn[..., 0]
        contains = np.all(crs >= 0, axis=1)
        outo.append(o[good])
        outk.append(od[k_][good])
        outm.append(m[good])
        outc.append(contains[good])
    if not outo:
        return (
            np.zeros(0, int),
            np.zeros(0, int),
            np.zeros((0, 6)),
            np.zeros(0, bool),
            pulled_area,
        )
    return (
        np.concatenate(outo),
        np.concatenate(outk),
        np.concatenate(outm),
        np.concatenate(outc),
        pulled_area,
    )


def _k2_reconstruction(pol, candidates, order, max_cond=1e6):
    """Weighted least-squares reconstruction of a density around each dof
    from the densities of the ``candidates`` (boolean mask: nodes whose leg
    in this direction is interior).

    For order 2 (quadratic flux reconstruction) the fit is quadratic, with
    the value fixed at the dof, over the 14 nearest candidates; order 1
    (linear reconstruction) is linear over 8 (on small planes with fewer
    candidates: all of them, if that is at least
    two more than the number of coefficients). The neighbours are distinct
    nodes, each at its nearest periodic image.
    Distances are Euclidean (minimum image), weights ``1/(d^2/<d^2> + 0.1)``.
    A stencil whose scaled least-squares matrix has a condition number
    above ``max_cond`` grows (k, k+6, k+14 neighbours), then drops to the
    next lower order, and finally to a constant (no correction).

    Returns ``(nb (n, kmax), P (n, 5, kmax), used_order (n,))``: neighbour
    node indices and the weights of the coefficients ``[d/dR, d/dZ,
    d2/dR2, d2/dRdZ, d2/dZ2]`` applied to ``rho_nb - rho_dof``. Only rows
    of candidate nodes are filled (others: order -1).
    """
    from scipy.spatial import cKDTree

    n = pol.n_nodes
    cand = np.flatnonzero(candidates)
    nc = len(cand)
    sizes = {2: 14, 1: 8}
    kmax = 0
    attempts = []
    for p in range(order, 0, -1):
        sizes_p = [kk for kk in (sizes[p], sizes[p] + 6, sizes[p] + 14) if kk < nc]
        if not sizes_p and nc - 1 >= num_coefs(p) + 1:
            # Small plane: as many neighbours as there are
            sizes_p = [nc - 1]
        for kk in sizes_p:
            attempts.append((p, kk))
            kmax = max(kmax, kk)
    nb = np.repeat(np.arange(n)[:, None], max(kmax, 1), axis=1)
    P = np.zeros((n, 5, max(kmax, 1)))
    used = np.full(n, -1)
    if nc == 0:
        return nb, P, used
    used[cand] = 0
    if kmax == 0:
        return nb, P, used
    X = pol.dof
    shifts = _periodic_shifts(pol)
    big = np.concatenate([X[cand] + s for s in shifts])
    tree = cKDTree(big)
    # Nearest distinct candidates other than the dof itself, each at its
    # nearest periodic image: on small periodic planes the nearest points
    # include further images of one node (or of the dof), which would make
    # the stencil inconsistent with a local polynomial
    kq = min(len(big), kmax + 1)
    while True:
        dist, idx = tree.query(X[cand], kq)
        dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
        node = idx % nc
        srt = np.sort(node, axis=1, kind="stable")
        dup = np.zeros(node.shape, dtype=bool)
        # a point is a duplicate if its node appeared earlier in the row
        # (rows are sorted by distance, so the first is the nearest image)
        order = np.argsort(node, axis=1, kind="stable")
        same = np.zeros(node.shape, dtype=bool)
        same[:, 1:] = srt[:, 1:] == srt[:, :-1]
        np.put_along_axis(dup, order, same, axis=1)
        dup |= node == np.arange(nc)[:, None]
        n_ok = np.sum(~dup, axis=1)
        if np.all(n_ok >= kmax) or kq == len(big):
            break
        kq = min(len(big), 2 * kq)
    if np.any(dist[:, 0] > 0) or np.any(node[:, 0] != np.arange(nc)):
        raise AssertionError("K2 reconstruction: the dof is not its own neighbour")
    if np.any(n_ok < kmax):
        raise AssertionError("K2 reconstruction: too few distinct neighbours")
    keep = np.argsort(dup, axis=1, kind="stable")[:, :kmax]
    idx = np.take_along_axis(idx, keep, axis=1)
    r = big[idx] - X[cand][:, None, :]
    nbr = cand[idx % nc]
    done = np.zeros(nc, dtype=bool)
    for p, kk in attempts:
        todo = np.flatnonzero(~done)
        if len(todo) == 0:
            break
        rr = r[todo, :kk]
        d2 = np.sum(rr**2, axis=-1)
        h2 = np.mean(d2, axis=1, keepdims=True)
        w = 1.0 / (d2 / h2 + 0.1)
        s = np.sqrt(h2)[..., None]
        u = rr / s
        cols = [u[..., 0], u[..., 1]]
        if p == 2:
            cols += [0.5 * u[..., 0] ** 2, u[..., 0] * u[..., 1], 0.5 * u[..., 1] ** 2]
        A = np.stack(cols, axis=-1) * w[..., None]
        su, sv, svt = np.linalg.svd(A, full_matrices=False)
        cond = sv[:, 0] / np.maximum(sv[:, -1], 1e-300)
        good = cond <= max_cond
        if not np.any(good):
            continue
        Pk = _pinv_from_svd(su[good], sv[good], svt[good]) * w[good][:, None, :]
        sg = np.sqrt(h2[good])  # (m, 1)
        Pk[:, :2] /= sg[..., None]
        if p == 2:
            Pk[:, 2:] /= (sg**2)[..., None]
        rows = cand[todo[good]]
        nb[rows, :kk] = nbr[todo[good], :kk]
        nb[rows, kk:] = rows[:, None]
        P[rows, :, :] = 0.0
        P[rows, : Pk.shape[1], :kk] = Pk
        used[rows] = p
        done[todo[good]] = True
    return nb, P, used


def _k2_basis_moments(m):
    """Integrals of the basis ``[dR, dZ, dR^2/2, dR dZ, dZ^2/2]`` over
    regions with moments ``m`` (relative to the dof)."""
    return np.column_stack([m[:, 1], m[:, 2], 0.5 * m[:, 3], m[:, 4], 0.5 * m[:, 5]])


def _k2_check_reconstruction(pol, nb, P, used, seed=0):
    """Self-check: the reconstruction is exact for a random quadratic
    (order-2 rows) or linear (order-1 rows) density. Returns the largest
    scaled error."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for p in (1, 2):
        rows = np.flatnonzero(used == p)
        if len(rows) == 0:
            continue
        r = pol.minimum_image(pol.dof[nb[rows]] - pol.dof[rows][:, None, :])
        h = np.sqrt(np.mean(np.sum(r**2, axis=-1), axis=1))
        g = rng.standard_normal((len(rows), 2)) / h[:, None]
        H = np.zeros((len(rows), 3))
        if p == 2:
            H = rng.standard_normal((len(rows), 3)) / (h**2)[:, None]
        drho = g[:, None, 0] * r[..., 0] + g[:, None, 1] * r[..., 1]
        drho += (
            0.5 * H[:, None, 0] * r[..., 0] ** 2
            + H[:, None, 1] * r[..., 0] * r[..., 1]
            + 0.5 * H[:, None, 2] * r[..., 1] ** 2
        )
        coef = np.einsum("pck,pk->pc", P[rows], drho)
        err = np.abs(coef[:, :2] - g) * h[:, None]
        worst = max(worst, float(err.max()))
        if p == 2:
            errq = np.abs(coef[:, 2:] - H) * (h**2)[:, None]
            worst = max(worst, float(errq.max()))
    return worst


def _k2_pair_task(task):
    """Pieces, reconstruction and local deposit/own-face matrices of one
    plane pair and direction (run in a worker process or inline).

    Returns local triplets: ``T`` (destination node, source node, value),
    the flux deposited into the destination cells from the source
    densities, and ``Own`` (source node, source node, value), the integral
    of each tube's reconstruction over its own source cell; and statistics.
    """
    src, dst = task["src"], task["dst"]
    interior = task["interior"]  # source nodes with an interior leg
    order = task["order"]
    t0 = time.time()
    o, k, mom, contains, pulled_area = _k2_pieces(src, task["td"], task["od"])
    n_src, n_dst = src.n_nodes, dst.n_nodes
    cellmom = _k2_cell_moments(src)
    # --- Self-check: partition of the areas (no overlaps; full coverage
    # wherever the tubes do not leave the mesh)
    a_src = np.bincount(o, weights=mom[:, 0], minlength=n_src)
    a_dst = np.bincount(k, weights=mom[:, 0], minlength=n_dst)
    pulled_area = np.concatenate(
        [pulled_area, np.zeros(max(0, n_dst - len(pulled_area)))]
    )
    def_src = (cellmom[:, 0] - a_src) / cellmom[:, 0]
    def_dst = (pulled_area - a_dst) / np.maximum(pulled_area, 1e-300)
    tol = 1e-12
    if def_src.min() < -tol or def_dst.min() < -tol:
        raise AssertionError(
            "K2 overlaps: pieces overlap (area sums exceed the cell areas by "
            f"{-def_src.min():.2e} (source) / {-def_dst.min():.2e} (destination))"
        )
    # --- Reconstruction (interior-leg cells only)
    if order >= 1:
        nb, P, used = _k2_reconstruction(
            src, interior, order, max_cond=task["max_cond"]
        )
        check = _k2_check_reconstruction(src, nb, P, used, seed=task["seed"])
        if check > 1e-8:
            raise AssertionError(f"K2 reconstruction not exact: error {check:.2e}")
    else:
        nb = np.repeat(np.arange(n_src)[:, None], 1, axis=1)
        P = np.zeros((n_src, 5, 1))
        used = np.where(interior, 0, -1)
        check = 0.0
    # --- Pieces of interior-leg tubes; the part of a tube outside the
    # destination mesh (the cell moments minus the pieces) goes to the
    # destination cell that contains the landing point of the field line
    sel = interior[o]
    walls = task["walls"]
    if walls is not None:
        # Wall nodes take no part in the parallel coupling: their parts of
        # the tubes go to the landing cell, like the parts outside the mesh
        sel &= ~walls[k]
    o, k, mom, contains = o[sel], k[sel], mom[sel], contains[sel]
    res = (
        cellmom
        - np.array(
            [np.bincount(o, weights=mom[:, c], minlength=n_src) for c in range(6)]
        ).T
    )
    land = np.full(n_src, -1)
    if len(o):
        # Piece containing the dof (largest if several), else the largest
        score = mom[:, 0] + np.where(contains, 1e300, 0.0)
        best = np.lexsort((-score, o))
        first = np.ones(len(best), dtype=bool)
        first[1:] = o[best][1:] != o[best][:-1]
        land[o[best][first]] = k[best][first]
    missing = np.flatnonzero(interior & (land < 0))
    if len(missing):
        # The tube misses the destination mesh entirely: nearest node to
        # the landing point
        lp = task["landing"][missing]
        cand = np.arange(n_dst) if walls is None else np.flatnonzero(~walls)
        d = dst.minimum_image(dst.dof[None, cand, :] - lp[:, None, :])
        land[missing] = cand[np.argmin(np.sum(d**2, axis=-1), axis=1)]
    rsrc = np.flatnonzero(interior)
    o = np.concatenate([o, rsrc])
    k = np.concatenate([k, land[rsrc]])
    mom = np.concatenate([mom, res[rsrc]])
    # --- Deposit and own face (in chunks of pieces, merging duplicate
    # entries per chunk, to bound the memory of the padded stencils)
    kk = nb.shape[1]

    def merged(r, c, v, n_rows):
        keep = v != 0.0
        ptr, cols, vals = _csr(n_rows, r[keep], c[keep], v[keep])
        return np.repeat(np.arange(n_rows), np.diff(ptr)), cols, vals

    def deposit(rows, src_nodes, m, n_rows, chunk=20000):
        parts = []
        for c0 in range(0, len(rows), chunk):
            r_, o_, m_ = (a[c0 : c0 + chunk] for a in (rows, src_nodes, m))
            coef = np.einsum("pc,pck->pk", _k2_basis_moments(m_), P[o_])
            parts.append(
                merged(
                    np.concatenate([r_, np.repeat(r_, kk)]),
                    np.concatenate([o_, nb[o_].ravel()]),
                    np.concatenate([m_[:, 0] - coef.sum(1), coef.ravel()]),
                    n_rows,
                )
            )
        if not parts:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        return merged(*(np.concatenate(x) for x in zip(*parts)), n_rows)

    T = deposit(k, o, mom, n_dst)
    Own = deposit(rsrc, rsrc, cellmom[rsrc], n_src)

    stats = {
        "pieces": int(len(o) - len(rsrc)),
        "max_deficit_src": float(def_src[interior].max()) if interior.any() else 0.0,
        "n_deficit_src": int(np.sum(def_src[interior] > tol)),
        "max_deficit_dst": float(def_dst.max()) if n_dst else 0.0,
        "n_deficit_dst": int(np.sum(def_dst > tol)),
        "order_counts": {
            int(p): int(np.sum(used == p)) for p in np.unique(used[interior])
        },
        "reconstruction_error": check,
        "time": time.time() - t0,
    }
    return T, Own, stats


def _k2_vertices(pol, tri):
    """Unique (wrapped) vertices of the triangles ``tri`` and, for every
    vertex of ``tri``, its index in that list. Copies of one vertex (shared
    by several triangles or periodic images) get the same index, so they
    are traced once and map identically."""
    pts = tri.reshape(-1, 2)
    w = pol.wrap(pts)
    key = np.round((w - pol.origin) / (1e-9 * pol.h)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return w[first], inverse.ravel()


_K2_TRACE = None  # (tracer, rtol) inherited by the trace workers of k2_divergence


def _k2_trace_task(points, y0, y1, trace=None):
    """Positions in the plane at y1 of the points of the plane at y0
    (``trace`` = (tracer, rtol), by default the inherited ``_K2_TRACE``)"""
    tracer, rtol = _K2_TRACE if trace is None else trace
    traced = np.array(
        tracer.follow_field_lines(points[:, 0], points[:, 1], [y0, y1], rtol=rtol)
    )
    return traced[-1]


def k2_divergence(
    grid,
    field,
    maps,
    data,
    order=2,
    refine="auto",
    tube_volume="auto",
    rtol=1e-10,
    max_cond=1e6,
    refine_tol=1e-3,
    processes=1,
    pullback=None,
    map_walls=True,
):
    """Flux-reconstructed conservative parallel half-divergences ("K2").

    Computes the grid variables ``k2_forward_rows/columns/weights`` and
    ``k2_backward_rows/columns/weights`` (CSR, ``total_cells`` rows in
    stored cell numbers, columns in stored forward/backward **leg**
    numbers) and the scalars ``k2_order`` and ``k2_version``.

    ``data`` must hold the variables of :func:`som_weights` (cell and leg
    numbering, ``forward_rows/columns/weights``), the metric (``dy``,
    ``g_22``) and ``dagp_us_volume``; ``maps`` is the output of
    :func:`trace_maps`.

    Convention (drop-in replacement of BOUT++'s ``Div_minus``/``Div_plus``
    of ``PetscOperators::getParallel``). With ``dV = dy*dagp_us_volume``,
    ``dl = dy*sqrt(g_22)``, the leg spacing ``dl+ = Interp_plus(dl)`` (as in
    BOUT++: ``(dl_i + sum_c F_lc dl_c)/2`` for an interior leg, ``dl_i`` for
    a boundary leg) and a leg flux ``q`` (``K_plus*Grad_plus f``),
    ``(D_f q)_c`` approximates the forward half of ``div(q b)`` in cell
    ``c``, exactly where ``Div_minus`` does::

        D_f = dV^-1 (Own_f - T_f) diag(V_tube / (dl+ A))
        D_b = dV^-1 (T_b - Own_b) diag(V_tube / (dl- A))

    ``q V_tube/(dl A)`` is the flux density on the source plane (``A`` the
    planar dual-cell area); ``T`` integrates its reconstruction around each
    source dof over the overlaps of the source cell with the destination
    cells (pulled back to the source plane), ``Own`` over the source cell.
    So the column of an interior leg sums to zero with weights ``dV``
    (exact conservation), and a leg flux that is constant across the tubes
    gives ``+q/dl`` in the source cell like ``Div_minus``.

    **Boundary legs** (field line leaves the mesh) have the same column as
    in ``Div_minus``/``Div_plus``: ``+1/dl+`` (``D_f``) or ``-1/dl-``
    (``D_b``) in the row of the source cell and the opposite value in the
    row of its parallel-boundary virtual cell. Their tubes are not deposited
    and their densities do not enter any reconstruction, so the factor-2
    boundary weight and the ``Interior_plus`` mask of
    ``Div_par_K_Grad_par`` apply unchanged. A tube with an interior leg
    that partly leaves the destination mesh deposits the outside part into
    the destination cell containing its landing point (conservative).
    Dirichlet wall nodes (no legs) carry no tube; deposits into them stay
    in their rows, as with the support operators. With ``map_walls=False``
    (as for :func:`som_weights`) the parts of the tubes over wall-node
    cells go to the landing cell instead, so that the flux stays in the
    evolving cells.

    ``order``: 2 = quadratic flux reconstruction (default), 1 = linear
    reconstruction, 0 = constant. ``refine``: the destination dual
    triangles are split into ``4**refine`` triangles before their vertices
    are traced back (piecewise-linear image of the curved cells);
    ``"auto"`` uses 0 if the
    maps are nearly affine (the traced midpoints of the triangle edges lie
    within ``refine_tol`` node spacings of the midpoints of the traced
    vertices, so that the straight-edged image misplaces at most about that
    fraction of a cell; exactly affine maps give round-off) and 1 otherwise. ``tube_volume``:
    ``"dV"`` (``V_tube = dV`` of the source cell, exact for area-preserving
    maps), ``"B"`` (``A B_n integral(ds/|B|)`` along the leg, Simpson rule
    with the traced half point, for fields whose ``B_y`` changes along the
    field lines) or ``"auto"`` (``"dV"`` if ``B_y`` is constant along every
    leg to 1e-10, else ``"B"``). ``pullback(R, Z, j, sign)``: optional
    exact map of points of plane ``j + sign`` to plane ``j`` (tests),
    instead of tracing with ``rtol``. ``processes``: worker processes, one
    plane pair at a time. Identical plane pairs (same meshes, legs and
    pulled-back vertices to 1e-12 node spacings) are computed once.

    Always-on self-checks (``AssertionError``): the pieces do not overlap
    (area sums at most the source and destination cell areas, 1e-12
    relative; equal where the tubes stay inside the mesh); every
    reconstruction is exact for a random quadratic (linear) density; every
    interior-leg column of ``dV D`` sums to zero (1e-12 relative); boundary
    columns as above.

    Returns ``(variables, stats)``.
    """

    t_start = time.time()
    if order not in (0, 1, 2):
        raise ValueError("k2 order must be 0, 1 or 2")
    ny = grid.ny
    node_x = grid.node_x
    cell_number = data["cell_number"]
    total_cells = int(data["total_cells"])
    dyj = grid.dy()
    tracer = None
    if pullback is None:
        tracer = fieldtracer.FieldTracer(field)
    cyl = is_cylindrical(field)

    # Cell-space metric (boundary virtual cells: as their parent cell, as
    # BOUT++'s parallel_neumann_o1)
    dl_cell = np.zeros(total_cells)
    dV_cell = np.zeros(total_cells)
    dlf = (data["dy"] * np.sqrt(data["g_22"]))[:, :, 0]
    dVf = (data["dy"] * data["dagp_us_volume"])[:, :, 0]
    cn = cell_number[:, :, 0]
    m = cn >= 0
    dl_cell[cn[m]] = dlf[m]
    dV_cell[cn[m]] = dVf[m]
    for name in ("forward", "backward"):
        b = data[f"{name}_cell_number"][:, :, 0]
        mb = b >= 0
        dl_cell[b[mb]] = dlf[mb]
        dV_cell[b[mb]] = dVf[mb]

    def bfield(R, Z, y):
        By = field.Byfunc(R, Z, y)
        return By, (By / field.Rfunc(R, Z, y) if cyl else By)

    # Tube volumes per source node and direction
    if tube_volume == "auto":
        const = True
        for name, sign in (("forward", 1), ("backward", -1)):
            for j in range(ny):
                pol = grid.plane(j)
                b0, _ = bfield(pol.dof[:, 0], pol.dof[:, 1], grid.ycoords[j])
                for R, Z, y in (
                    (
                        maps[f"{name}_half_R"][j],
                        maps[f"{name}_half_Z"][j],
                        grid.ycoords[j] + 0.5 * sign * dyj[j],
                    ),
                    (
                        maps[f"{name}_R"][j],
                        maps[f"{name}_Z"][j],
                        grid.ycoords[j] + sign * dyj[j],
                    ),
                ):
                    b1, _ = bfield(R, Z, y)
                    if np.max(np.abs(b1 - b0) / np.abs(b0)) > 1e-10:
                        const = False
        tube_volume = "dV" if const else "B"
    if tube_volume not in ("dV", "B"):
        raise ValueError("tube_volume must be 'auto', 'dV' or 'B'")

    # --- Pull back the destination dual triangles of every plane pair
    items = []  # (j, sign)
    for name, sign in (("forward", 1), ("backward", -1)):
        for j in range(ny):
            jd = j + sign
            if not grid.yperiodic and (jd < 0 or jd >= ny):
                continue
            hit = maps[f"{name}_hit"][j]
            interior = (hit == 0) & grid.interior
            if not interior.any():
                continue
            items.append((name, sign, j, jd % ny, interior))
    t_trace = time.time()
    tri_cache = {}

    def dest_triangles(pol, levels):
        key = (id(pol), levels)
        if key not in tri_cache:
            td, od = _refine_triangles(*_dual_triangles(pol), levels)
            verts, inv = _k2_vertices(pol, td)
            tri_cache[key] = (td, od, verts, inv)
        return tri_cache[key]

    def pull(points, j, sign):
        """Positions in plane j of points of plane j + sign."""
        if pullback is not None:
            R, Z = pullback(points[:, 0], points[:, 1], j, sign)
            return np.column_stack([R, Z])
        y0 = grid.ycoords[j] + sign * dyj[j]
        return _k2_trace_task(points, y0, grid.ycoords[j], (tracer, rtol))

    levels_traced = 1 if refine == "auto" else int(refine)
    pulled = []
    max_dev = 0.0
    # The traces of the items are independent: in parallel if processes > 1
    # (fork, so that the tracer is inherited rather than pickled)
    traced_items = [None] * len(items)
    if processes > 1 and pullback is None and len(items) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        global _K2_TRACE
        _K2_TRACE = (tracer, rtol)
        try:
            with ProcessPoolExecutor(
                max_workers=processes, mp_context=multiprocessing.get_context("fork")
            ) as ex:
                futures = [
                    ex.submit(
                        _k2_trace_task,
                        dest_triangles(grid.plane(jd), levels_traced)[2],
                        grid.ycoords[j] + sign * dyj[j],
                        grid.ycoords[j],
                    )
                    for _, sign, j, jd, _ in items
                ]
                traced_items = [f.result() for f in futures]
        finally:
            _K2_TRACE = None
    for it, (name, sign, j, jd, interior) in enumerate(items):
        dst = grid.plane(jd)
        td, od, verts, inv = dest_triangles(dst, levels_traced)
        end = traced_items[it] if traced_items[it] is not None else pull(verts, j, sign)
        disp = end - verts
        tdp = td + disp[inv].reshape(td.shape)
        if refine == "auto":
            # Affine test: traced midpoints of the first refinement against
            # the midpoints of the traced vertices. In the refined list the
            # parent triangle t has children t (a, ab, ca), N + t (ab, b,
            # bc), 2N + t (ca, bc, c), 3N + t (ab, bc, ca)
            N = len(td) // 4
            a, b, c = tdp[:N, 0], tdp[N : 2 * N, 1], tdp[2 * N : 3 * N, 2]
            mid = tdp[3 * N :]
            lin = np.stack([0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)], 1)
            dev = np.max(np.abs(mid - lin) / dst.h)
            max_dev = max(max_dev, float(dev))
        pulled.append((tdp, od))
    t_trace = time.time() - t_trace
    if refine == "auto":
        levels = 0 if max_dev < refine_tol else 1
        if levels == 0:
            # Parent triangles: vertices a, b, c of the children
            new = []
            for tdp, od in pulled:
                N = len(tdp) // 4
                parent = np.stack(
                    [tdp[:N, 0], tdp[N : 2 * N, 1], tdp[2 * N : 3 * N, 2]], 1
                )
                new.append((parent, od[:N]))
            pulled = new
    else:
        levels = int(refine)

    # --- Group identical plane pairs
    unique = []  # indices into items of the computed pairs
    which = []
    for it, (name, sign, j, jd, interior) in enumerate(items):
        found = -1
        for u in unique:
            _, s2, j2, jd2, int2 = items[u]
            if (
                s2 == sign
                and grid.plane(j2) is grid.plane(j)
                and grid.plane(jd2) is grid.plane(jd)
                and np.array_equal(int2, interior)
                and pulled[u][0].shape == pulled[it][0].shape
                and np.max(np.abs(pulled[u][0] - pulled[it][0]))
                <= 1e-12 * np.min(grid.plane(j).h)
            ):
                found = u
                break
        if found < 0:
            unique.append(it)
            found = it
        which.append(found)

    def task_of(it):
        name, _, j, jd, interior = items[it]
        return {
            "src": grid.plane(j),
            "dst": grid.plane(jd),
            "td": pulled[it][0],
            "od": pulled[it][1],
            "interior": interior,
            "order": order,
            "max_cond": max_cond,
            "seed": it,
            "landing": np.column_stack((maps[f"{name}_R"][j], maps[f"{name}_Z"][j])),
            "walls": None if map_walls else ~grid.interior,
        }

    results = {}
    t_pairs = time.time()
    if processes > 1 and len(unique) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=processes, mp_context=multiprocessing.get_context("fork")
        ) as ex:
            futures = {}
            queue = list(unique)
            while queue or futures:
                while queue and len(futures) < processes:
                    u = queue.pop(0)
                    futures[u] = ex.submit(_k2_pair_task, task_of(u))
                u0 = next(iter(futures))
                results[u0] = futures.pop(u0).result()
    else:
        for u in unique:
            results[u] = _k2_pair_task(task_of(u))
    t_pairs = time.time() - t_pairs

    # --- Assemble in stored indices
    out = {}
    stats = {
        "order": order,
        "refine": levels,
        "max_affine_deviation": max_dev,
        "tube_volume": tube_volume,
        "pairs": len(items),
        "unique_pairs": len(unique),
        "pieces": 0,
        "order_counts": {},
        "max_deficit_src": 0.0,
        "max_deficit_dst": 0.0,
        "n_deficit_src": 0,
        "n_deficit_dst": 0,
        "reconstruction_error": 0.0,
    }
    for u in unique:
        st = results[u][2]
        for key in ("pieces", "n_deficit_src", "n_deficit_dst"):
            stats[key] += st[key]
        for key in ("max_deficit_src", "max_deficit_dst", "reconstruction_error"):
            stats[key] = max(stats[key], st[key])
        for p, c in st["order_counts"].items():
            stats["order_counts"][p] = stats["order_counts"].get(p, 0) + c
    for name, sign in (("forward", 1), ("backward", -1)):
        leg_int = data[f"{name}_leg_interior_number"][:, :, 0]
        leg_bnd = data[f"{name}_leg_boundary_number"][:, :, 0]
        bnd_cell = data[f"{name}_cell_number"][:, :, 0]
        n_legs = int(data[f"n_{name}_legs"])
        # Leg spacing as BOUT++: Interp_plus(dl)
        F = (
            data[f"{name}_weights"],
            data[f"{name}_columns"],
            data[f"{name}_rows"],
        )
        Fdl = (
            np.add.reduceat(np.concatenate([F[0] * dl_cell[F[1]], [0.0]]), F[2][:-1])
            if len(F[0])
            else np.zeros(n_legs)
        )
        Fdl = np.where(np.diff(F[2]) > 0, Fdl, 0.0)
        rows_all, cols_all, vals_all = [], [], []
        for it, (nm, sg, j, jd, interior) in enumerate(items):
            if nm != name:
                continue
            (Tr, Tc, Tv), (Orr, Oc, Ov), _ = results[which[it]]
            src, dst = grid.plane(j), grid.plane(jd)
            cells_j = cn[node_x, j]
            cells_d = cn[node_x, jd]
            legs_j = leg_int[node_x, j]
            if np.any(legs_j[interior] < 0):
                raise AssertionError("K2: leg numbering disagrees with the hits")
            # Column scale V_tube / (dl A) per source node (interior legs)
            A = src.dual_volumes()
            dlp = 0.5 * (
                dl_cell[cells_j] + np.where(interior, Fdl[np.maximum(legs_j, 0)], 0.0)
            )
            if tube_volume == "dV":
                V = dV_cell[cells_j]
            else:
                y0 = grid.ycoords[j]
                Bn, b0 = bfield(src.dof[:, 0], src.dof[:, 1], y0)
                _, bh = bfield(
                    maps[f"{name}_half_R"][j],
                    maps[f"{name}_half_Z"][j],
                    y0 + 0.5 * sign * dyj[j],
                )
                _, b1 = bfield(
                    maps[f"{name}_R"][j], maps[f"{name}_Z"][j], y0 + sign * dyj[j]
                )
                V = A * Bn * dyj[j] * (1 / b0 + 4 / bh + 1 / b1) / 6.0
            scale = V / (dlp * A)
            # D = sign * dV^-1 (Own - T) scale
            rows_all += [cells_j[Orr], cells_d[Tr]]
            cols_all += [legs_j[Oc], legs_j[Tc]]
            vals_all += [
                sign * Ov * scale[Oc] / dV_cell[cells_j[Orr]],
                -sign * Tv * scale[Tc] / dV_cell[cells_d[Tr]],
            ]
        # Boundary legs: the column of Div_minus / Div_plus
        ev = grid.interior
        for j in range(ny):
            lb = leg_bnd[node_x, j]
            sel = ev & (lb >= 0)
            if not sel.any():
                continue
            c = cn[node_x[sel], j]
            bc = bnd_cell[node_x[sel], j]
            dlb = dl_cell[c]
            rows_all += [c, bc]
            cols_all += [lb[sel], lb[sel]]
            vals_all += [sign / dlb, -sign / dlb]
        r = np.concatenate(rows_all) if rows_all else np.zeros(0, int)
        c = np.concatenate(cols_all) if cols_all else np.zeros(0, int)
        v = np.concatenate(vals_all) if vals_all else np.zeros(0)
        if np.any(r < 0) or np.any(c < 0) or np.any(c >= n_legs):
            raise AssertionError(f"K2 {name}: invalid row or column index")
        ptr, cols, vals = _csr(total_cells, r, c, v)
        # Self-check: column conservation sum_rows dV D = 0 for every leg
        rows = np.repeat(np.arange(total_cells), np.diff(ptr))
        colsum = np.bincount(cols, weights=dV_cell[rows] * vals, minlength=n_legs)
        # (scale from the entries before duplicates were merged: own-face
        # and deposit entries in the same cell cancel when a plane maps
        # onto itself)
        colabs = np.bincount(c, weights=np.abs(dV_cell[r] * v), minlength=n_legs)
        bad = np.abs(colsum) > 1e-12 * np.maximum(colabs, 1e-300)
        if np.any(bad):
            raise AssertionError(
                f"K2 {name}: {int(bad.sum())} columns not conservative "
                f"(max relative {np.max(np.abs(colsum) / np.maximum(colabs, 1e-300)):.2e})"
            )
        stats[f"max_column_sum_{name}"] = float(
            np.max(np.abs(colsum) / np.maximum(colabs, 1e-300), initial=0.0)
        )
        out[f"k2_{name}_rows"] = ptr
        out[f"k2_{name}_columns"] = cols
        out[f"k2_{name}_weights"] = vals
        stats[f"nnz_{name}"] = len(vals)
    out["k2_order"] = int(order)
    out["k2_version"] = K2_VERSION
    stats["time_trace"] = t_trace
    stats["time_pairs"] = t_pairs
    stats["time"] = time.time() - t_start
    return out, stats


# ----------------------------------------------------------------------
# Top level


def build_grid_data(
    grid,
    field,
    map_order=4,
    dagp_order=4,
    strict=True,
    rtol=1e-10,
    domain_tol=1e-9,
    map_extra=16,
    map_weight_power=3.0,
    map_walls=True,
    near_wall_hits=False,
    perp_walls=True,
    dagp_extra=16,
    value_order=2,
    processes=1,
    k2=True,
    k2_order=2,
    k2_refine="auto",
    k2_tube_volume="auto",
):
    """Compute every variable of the grid-file contract. Returns
    ``(data, stats)``.

    ``map_order`` is the polynomial order of the parallel-map interpolation.
    Cubic exactness (order 3) is the minimum for second-order parallel second
    derivatives; on irregular meshes its O(h^4) error is still
    pre-asymptotic at typical resolutions, so order 4 is the default.
    ``dagp_order`` is the order of the face gradient/value fits of the
    perpendicular operator (order 4 with the larger stencils of
    :func:`_plane_dagp` gives robust second order and no growing modes on
    the test grids). ``domain_tol``: see :func:`trace_maps`. ``map_extra``
    and ``map_weight_power`` set the stencil size (``num_coefs(map_order) +
    map_extra`` nodes) and the distance weighting of the parallel-map fits
    (see :meth:`UnstructuredPoloidalGrid.lsq_weights`); ``map_walls=False``
    keeps the Dirichlet wall nodes out of the parallel-map stencils (see
    :func:`som_weights`) and out of the K2 deposits (see
    :func:`k2_divergence`); ``near_wall_hits``: see :func:`trace_maps`.
    Without ``near_wall_hits`` the stencils of end points next to a wall
    extrapolate one-sidedly, so use ``map_walls=False`` only together with
    ``near_wall_hits=True``. ``perp_walls=False`` makes the walls insulating
    for the perpendicular operator (see :func:`dagp_unstructured`).
    ``dagp_extra`` is the number of extra nodes of
    the face-gradient fits (16 by default, needed
    on coarse anisotropic meshes; 4 is enough on near-isotropic gmsh meshes)
    and ``value_order`` the order of the separate fit for the face values of
    the coefficient a (2 by default: second order with about a third of the
    nonzeros of the gradient fit; ``None`` reuses the gradient fit).
    ``processes`` is the number of worker processes for the perpendicular
    operator of the planes (see :func:`dagp_unstructured`; each worker holds
    one plane, about 0.3 GB for a 7000-node gmsh plane).
    ``k2`` (default True) also computes the flux-reconstructed conservative
    parallel half-divergences ``k2_*`` (see :func:`k2_divergence`) with
    reconstruction order ``k2_order`` (2 = quadratic, 1 = linear),
    ``k2_refine`` and ``k2_tube_volume``; ``k2=False`` skips them (faster, for tests that
    do not need them).
    """
    maps = trace_maps(
        grid, field, rtol=rtol, domain_tol=domain_tol, near_wall_hits=near_wall_hits
    )
    som, som_stats = som_weights(
        grid,
        maps,
        order=map_order,
        strict=strict,
        extra=map_extra,
        weight_power=map_weight_power,
        map_walls=map_walls,
    )
    dagp, dagp_stats = dagp_unstructured(
        grid,
        som["cell_number"],
        som["total_cells"],
        order=dagp_order,
        strict=strict,
        cylindrical=is_cylindrical(field),
        extra=dagp_extra,
        value_order=value_order,
        processes=processes,
        perp_walls=perp_walls,
    )
    data = {}
    data.update(slab_metric(grid, field, maps))
    data.update(som)
    data.update(dagp)

    # Positions of the unknowns (the nodes, or the dual-cell centroids)
    R = np.array([grid.plane(j).dof[:, 0] for j in range(grid.ny)])
    Z = np.array([grid.plane(j).dof[:, 1] for j in range(grid.ny)])
    data["R"] = grid.to3d(R)
    data["Z"] = grid.to3d(Z)
    data["Rxy"] = data["R"]
    # Attributes of the magnetic field (e.g. psi), as for structured grids
    for name, func in getattr(field, "attributes", {}).items():
        data[name] = grid.to3d(
            np.array([func(R[j], Z[j], grid.ycoords[j]) for j in range(grid.ny)])
        )
    for name in ("forward", "backward"):
        data[f"{name}_R"] = grid.to3d(maps[f"{name}_R"])
        data[f"{name}_Z"] = grid.to3d(maps[f"{name}_Z"])
        hit = maps[f"{name}_hit"]
        node_index = grid.node_x[None, :] + 0.0 * hit
        xt = np.where(hit == 0, node_index, np.where(hit > 0, grid.nx, -1.0))
        data[f"{name}_xt_prime"] = grid.to3d(xt)
        data[f"{name}_zt_prime"] = np.zeros(grid.shape)
        # The leg classification must agree with the sentinels
        evolving = grid.interior
        legs = data[f"{name}_leg_interior_number"][grid.node_x[evolving], :, 0].T
        if not np.array_equal(legs >= 0, hit[:, evolving] == 0):
            raise AssertionError(f"{name}: leg classification disagrees with xt_prime")
    k2_stats = None
    if k2:
        k2_data, k2_stats = k2_divergence(
            grid,
            field,
            maps,
            data,
            order=k2_order,
            refine=k2_refine,
            tube_volume=k2_tube_volume,
            rtol=rtol,
            processes=processes,
            map_walls=map_walls,
        )
        data.update(k2_data)
    data["unstructured"] = 1
    data["MXG"] = grid.MXG
    data["MYG"] = grid.MYG
    data["n_nodes"] = grid.n_nodes
    stats = {
        "n_nodes": grid.n_nodes,
        "ny": grid.ny,
        "som": som_stats,
        "dagp": dagp_stats,
        "k2": k2_stats,
        "forward_hits": int(np.sum(maps["forward_hit"] != 0)),
        "backward_hits": int(np.sum(maps["backward_hit"] != 0)),
    }
    return data, stats


def write_unstructured(filename, grid, data):
    """Write ``data`` (from :func:`build_grid_data`) to a BOUT++ grid file."""
    from .zoidberg import MapWriter

    with MapWriter(filename, quiet=True) as mw:
        mw.nxyz = grid.shape
        mw.write_dict(data)
        mw._final()


def make_unstructured_grid(
    filename,
    poloidal,
    field,
    ny,
    Ly,
    yperiodic=True,
    walls_y=False,
    map_order=4,
    dagp_order=4,
    strict=True,
    multiple_of=4,
    dirichlet_walls=False,
    domain_tol=1e-9,
    **grid_options,
):
    """Convenience wrapper: build the 3D grid, compute and write all data.

    ``poloidal`` is one :class:`UnstructuredPoloidalGrid` for every plane or
    a list of ``ny`` of them (e.g. from :func:`gmsh_planes`). Further keyword
    arguments (``map_extra``, ``dagp_extra``, ``value_order``, ...) are passed
    to :func:`build_grid_data`.
    """
    ycoords = np.linspace(0.0, Ly, ny, endpoint=False)
    grid = UnstructuredGrid(
        poloidal,
        ycoords,
        Ly,
        yperiodic=yperiodic,
        walls_y=walls_y,
        multiple_of=multiple_of,
        dirichlet_walls=dirichlet_walls,
    )
    data, stats = build_grid_data(
        grid,
        field,
        map_order=map_order,
        dagp_order=dagp_order,
        strict=strict,
        domain_tol=domain_tol,
        **grid_options,
    )
    if filename is not None:
        write_unstructured(filename, grid, data)
    return grid, data, stats


def gmsh_planes(
    outers,
    inners=None,
    mesh_size=None,
    relax=False,
    multiple_of=4,
    centroid_dofs=True,
    dirichlet_walls=True,
    wall_size=None,
    wall_distance=None,
):
    """Mesh every poloidal cross-section independently with gmsh.

    ``outers[j]`` (and ``inners[j]``, or ``None`` for discs) are the closed
    wall curves of plane ``j`` (see :meth:`UnstructuredPoloidalGrid.from_gmsh`).
    The grid format stores nodes along x, so all planes need the same number
    of nodes and of wall nodes. The wall-node counts are fixed from the
    longest curve of all planes; the planes are meshed, and those with fewer
    nodes than the largest are re-meshed with the missing nodes added in
    their largest triangles. The total is rounded up so that the number of
    evolving nodes is a multiple of ``multiple_of``: the non-wall nodes with
    ``dirichlet_walls=True`` (the default; the wall nodes then go into the x
    guard cells), all nodes with ``dirichlet_walls=False``. Pass the same
    ``dirichlet_walls`` as to :class:`UnstructuredGrid`. ``relax``,
    ``centroid_dofs``, ``wall_size`` and ``wall_distance``: see
    :meth:`UnstructuredPoloidalGrid.from_gmsh`.
    """
    ny = len(outers)
    if inners is None:
        inners = [None] * ny
    outer_size = mesh_size if wall_size is None else wall_size
    n_outer = int(np.ceil(max(_curve_length(c) for c in outers) / outer_size))
    n_inner = None
    if inners[0] is not None:
        n_inner = int(np.ceil(max(_curve_length(c) for c in inners) / mesh_size))
    kw = {"mesh_size": mesh_size, "n_outer": n_outer, "n_inner": n_inner}
    kw.update(wall_size=wall_size, wall_distance=wall_distance)
    kw["centroid_dofs"] = False  # set after relaxing
    planes = [
        UnstructuredPoloidalGrid.from_gmsh(o, i, relax=False, **kw)
        for o, i in zip(outers, inners)
    ]
    n_walls = n_outer + (n_inner or 0) if dirichlet_walls else 0
    target = max(p.n_nodes for p in planes)
    target += -(target - n_walls) % multiple_of
    planes = [
        p
        if p.n_nodes == target
        else UnstructuredPoloidalGrid.from_gmsh(o, i, n_nodes=target, relax=False, **kw)
        for p, o, i in zip(planes, outers, inners)
    ]
    for p in planes:
        if relax:
            p.centroidal_relax(omega=1.5)
        if centroid_dofs:
            p.use_centroid_dofs()
    return planes


# Parameters of the BOUT-dev MMS-spatial-fci-unstructured test grids. These
# match tests/MMS/spatial/fci (Slab(Bz=0.05, Bzprime=0.1), Lx = 0.1,
# Lz = 1, ylength = 10), with 8 fixed R layers and jitter in Z only.
BOUT_MMS = {
    "nR": 8,
    "Lx": 0.1,
    "Lz": 1.0,
    "R0": -0.05,
    "Z0": 0.0,
    "Ly": 10.0,
    "jitter": 0.2,
    "seed": 1234,
}


# Isotropic variant of BOUT_MMS (same geometry and field): nR grows with n
# (nR = n / nR_divisor), jitter in R and Z, so that the plane mesh is
# isotropic at every resolution (BOUT_MMS keeps nR = 8 fixed, so its plane
# mesh goes from anisotropic to isotropic within n = 8-64)
BOUT_MMS_ISO = dict(BOUT_MMS, nR_divisor=8, nR_min=4, seed=2345)


# Cylindrical test grids: pure toroidal field (Screwpinch without shear) on
# an annulus R in [1, 2] with Dirichlet walls, periodic in Z (Lz = 1).
BOUT_MMS_CYL = {"R0": 1.0, "Lx": 1.0, "Lz": 1.0, "ny": 4, "jitter": 0.2, "seed": 4321}

# Cylindrical grids with a sheared field for the parallel operators: the
# annulus of BOUT_MMS_CYL with B_phi = 1/R and a vertical field
# B_Z = Bz0 + Bzprime (R - Rc), so that field lines move in Z (periodic) and
# the parallel maps interpolate; n planes over the full torus. The walls are
# evolving nodes by default (field lines do not reach them), so that the
# support operators conserve exactly; with dirichlet_walls=True part of the
# parallel flux goes into the wall nodes through the interpolation stencils.
BOUT_MMS_CYL_SHEARED = dict(
    BOUT_MMS_CYL,
    ny=None,
    Bz0=0.1,
    Bzprime=0.1,
    Rc=1.5,
    seed=5678,
    dirichlet_walls=False,
)


class ShearedCylinderField(MagneticField):
    """Field of :data:`BOUT_MMS_CYL_SHEARED` (cylindrical, y = toroidal angle):
    ``B_R = 0``, ``B_phi = 1/R``, ``B_Z = Bz0 + Bzprime (R - Rc)``."""

    def __init__(self, Bz0=0.1, Bzprime=0.1, Rc=1.5):
        self.Bz0, self.Bzprime, self.Rc = float(Bz0), float(Bzprime), float(Rc)
        self.attributes = {}

    def Bxfunc(self, x, z, phi):
        return np.zeros(np.shape(x))

    def Byfunc(self, x, z, phi):
        return 1.0 / np.asarray(x, dtype=float)

    def Bzfunc(self, x, z, phi):
        return self.Bz0 + self.Bzprime * (np.asarray(x, dtype=float) - self.Rc)

    def Rfunc(self, x, z, phi):
        return np.asarray(x, dtype=float)


def family_spec(family, n, **kwargs):
    """Poloidal mesh, field and layout of a test-grid family (see
    :func:`make_test_grid`). Returns ``(poloidal, field, layout)`` with
    ``layout`` the keyword arguments ``ny``, ``Ly``, ``yperiodic`` and
    ``dirichlet_walls`` of :func:`make_unstructured_grid`. ``kwargs``
    override the family parameters. The planes have centroidally relaxed
    nodes as unknowns (:meth:`UnstructuredPoloidalGrid.slab`, ``relax=True``;
    on the walls of ``bout_mms_iso`` and the cylinder families the wall
    nodes stay on the wall, half a cell from their dual-cell centroid);
    ``centroid_dofs=True`` puts the unknowns at the dual-cell centroids
    instead (:meth:`UnstructuredPoloidalGrid.use_centroid_dofs`), which the
    K2 divergence needs for second order next to walls."""
    centroid_dofs = kwargs.pop("centroid_dofs", False)
    if family == "bout_mms_cyl_sheared":
        p = dict(BOUT_MMS_CYL_SHEARED, **kwargs)
        pol = UnstructuredPoloidalGrid.slab(
            n + 1,
            n,
            p["Lx"],
            p["Lz"],
            R0=p["R0"],
            Z0=0.0,
            jitter=p["jitter"],
            jitter_dirs="rz",
            periodic_R=False,
            periodic_Z=True,
            seed=p["seed"] + n,
            r_on_walls=True,
        )
        field = ShearedCylinderField(p["Bz0"], p["Bzprime"], p["Rc"])
        layout = {
            "ny": p["ny"] or n,
            "Ly": 2.0 * np.pi,
            "yperiodic": True,
            "dirichlet_walls": p["dirichlet_walls"],
        }
    elif family == "bout_mms_cyl":
        from .field import Screwpinch

        p = dict(BOUT_MMS_CYL, **kwargs)
        pol = UnstructuredPoloidalGrid.slab(
            n + 1,
            n,
            p["Lx"],
            p["Lz"],
            R0=p["R0"],
            Z0=0.0,
            jitter=p["jitter"],
            jitter_dirs="rz",
            periodic_R=False,
            periodic_Z=True,
            seed=p["seed"] + n,
            r_on_walls=True,
        )
        field = Screwpinch(xcentre=p["R0"] + 0.5 * p["Lx"], shear=0.0, Btor=1.0)
        layout = {
            "ny": p["ny"],
            "Ly": 2.0 * np.pi,
            "yperiodic": True,
            "dirichlet_walls": True,
        }
    elif family == "bout_mms_iso":
        p = dict(BOUT_MMS_ISO, **kwargs)
        pol = UnstructuredPoloidalGrid.slab(
            max(p["nR_min"], n // p["nR_divisor"]),
            n,
            p["Lx"],
            p["Lz"],
            R0=p["R0"],
            Z0=p["Z0"],
            jitter=p["jitter"],
            jitter_dirs="rz",
            periodic_R=False,
            periodic_Z=True,
            seed=p["seed"] + n,
            r_on_walls=True,
        )
        field = Slab(Bz=0.05, Bzprime=0.1)
        layout = {"ny": n, "Ly": p["Ly"], "yperiodic": True, "dirichlet_walls": False}
    elif family == "bout_mms":
        p = dict(BOUT_MMS, **kwargs)
        pol = UnstructuredPoloidalGrid.slab(
            p["nR"],
            n,
            p["Lx"],
            p["Lz"],
            R0=p["R0"],
            Z0=p["Z0"],
            jitter=p["jitter"],
            jitter_dirs="z",
            periodic_R=False,
            periodic_Z=True,
            seed=p["seed"] + n,
        )
        field = Slab(Bz=0.05, Bzprime=0.1)
        layout = {"ny": n, "Ly": p["Ly"], "yperiodic": True, "dirichlet_walls": False}
    else:
        raise ValueError(f"Unknown test-grid family '{family}'")
    if centroid_dofs:
        pol.use_centroid_dofs()
    return pol, field, layout


def make_test_grid(family, n, filename=None, **kwargs):
    """Shared entry point for test grids (used by BOUT-dev and Hermes-3).

    Families: ``"bout_mms"`` (MMS-spatial-fci-unstructured, Slab field),
    ``"bout_mms_iso"`` (same geometry and field, nR = n/8 and jitter in R and
    Z, so the plane mesh is isotropic at every n),
    ``"bout_mms_cyl"`` (cylindrical case of that test, toroidal field) and
    ``"bout_mms_cyl_sheared"`` (cylindrical, sheared field, ``n`` planes; for
    the parallel operators). Other families are built by the callers with
    :func:`make_unstructured_grid`.

    Keyword arguments that are options of :func:`build_grid_data` (e.g.
    ``k2=False``, ``processes``) are passed to it; the others override the
    family parameters (``seed``, ``jitter``, ...).
    """
    import inspect

    build_keys = set(inspect.signature(build_grid_data).parameters) - {
        "grid",
        "field",
    }
    build = {k: kwargs.pop(k) for k in list(kwargs) if k in build_keys}
    pol, field, layout = family_spec(family, n, **kwargs)
    return make_unstructured_grid(filename, pol, field, **layout, **build)


def family_grid(family, n, **kwargs):
    """The :class:`UnstructuredGrid` and field of a test-grid family, without
    computing the grid data (e.g. for :func:`add_k2_to_file`)."""
    pol, field, layout = family_spec(family, n, **kwargs)
    ny = layout.pop("ny")
    grid = UnstructuredGrid(
        pol,
        np.linspace(0.0, layout["Ly"], ny, endpoint=False),
        layout["Ly"],
        yperiodic=layout["yperiodic"],
        dirichlet_walls=layout["dirichlet_walls"],
    )
    return grid, field


def add_k2_to_file(
    filename,
    grid,
    field,
    rtol=1e-10,
    domain_tol=1e-9,
    near_wall_hits=False,
    **k2_options,
):
    """Add (or overwrite) the ``k2_*`` variables of an existing unstructured
    grid file, without regenerating its other variables.

    The file stores neither the triangulations nor the field, so ``grid``
    (:class:`UnstructuredGrid`) and ``field`` must be the ones the file was
    made with (e.g. from :func:`family_grid`). They are checked against
    the file: the unknowns' positions, the traced end points and the leg
    classification must agree to 1e-10 (relative to the node spacing).
    The numbering, the leg stencils, ``dy``, ``g_22`` and ``dagp_us_volume``
    are read from the file, so the matrices match its other variables.
    ``rtol``, ``domain_tol`` and ``near_wall_hits`` as for
    :func:`build_grid_data`; further options go to :func:`k2_divergence`.
    Returns the statistics.
    """
    import os

    import netCDF4

    names = [
        "cell_number",
        "total_cells",
        "forward_cell_number",
        "backward_cell_number",
        "dy",
        "g_22",
        "dagp_us_volume",
        "R",
        "Z",
    ]
    for d in ("forward", "backward"):
        names += [
            f"{d}_leg_interior_number",
            f"{d}_leg_boundary_number",
            f"n_{d}_legs",
            f"{d}_rows",
            f"{d}_columns",
            f"{d}_weights",
            f"{d}_R",
            f"{d}_Z",
            f"{d}_xt_prime",
        ]
    with netCDF4.Dataset(filename) as f:
        if int(np.asarray(f["unstructured"][...])) != 1:
            raise ValueError(f"{filename} is not an unstructured grid file")
        data = {}
        for k in names:
            v = np.asarray(f[k][...])
            data[k] = v if v.ndim else v.item()
        existing = [k for k in f.variables if k.startswith("k2_")]
    if tuple(data["cell_number"].shape) != grid.shape:
        raise ValueError(
            f"Grid shape {grid.shape} does not match the file {data['cell_number'].shape}"
        )
    h = min(np.min(grid.plane(j).h) for j in range(grid.ny))
    dof = np.array([grid.plane(j).dof for j in range(grid.ny)])
    for c, key in enumerate(("R", "Z")):
        if np.max(np.abs(grid.to3d(dof[..., c]) - data[key])) > 1e-10 * h:
            raise ValueError(f"{key} of the grid does not match the file")
    maps = trace_maps(
        grid, field, rtol=rtol, domain_tol=domain_tol, near_wall_hits=near_wall_hits
    )
    ev = grid.interior
    for d in ("forward", "backward"):
        for c in ("R", "Z"):
            diff = grid.to3d(maps[f"{d}_{c}"]) - data[f"{d}_{c}"]
            if np.max(np.abs(diff[grid.node_x[ev]])) > 1e-10 * max(h, 1.0):
                raise ValueError(f"{d}_{c}: traced end points differ from the file")
        legs = data[f"{d}_leg_interior_number"][grid.node_x[ev], :, 0].T
        if not np.array_equal(legs >= 0, maps[f"{d}_hit"][:, ev] == 0):
            raise ValueError(f"{d}: leg classification differs from the file")
    k2_data, stats = k2_divergence(grid, field, maps, data, rtol=rtol, **k2_options)
    if existing:
        # netCDF cannot resize dimensions: copy the file without the old
        # k2 variables
        tmp = str(filename) + ".k2tmp"
        with (
            netCDF4.Dataset(filename) as src,
            netCDF4.Dataset(tmp, "w", format=src.data_model) as dst,
        ):
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
            used = {
                d
                for k, v in src.variables.items()
                if k not in existing
                for d in v.dimensions
            }
            for name, dim in src.dimensions.items():
                if name in used:
                    dst.createDimension(name, None if dim.isunlimited() else len(dim))
            for name, var in src.variables.items():
                if name in existing:
                    continue
                out = dst.createVariable(name, var.datatype, var.dimensions)
                out.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
                out[...] = var[...]
        os.replace(tmp, filename)
    from .zoidberg import MapWriter

    with MapWriter(str(filename), quiet=True, create=False) as mw:
        mw.write_dict(k2_data)
    return stats


__all__ = [
    "BOUT_MMS",
    "BOUT_MMS_CYL",
    "BOUT_MMS_CYL_SHEARED",
    "BOUT_MMS_ISO",
    "ShearedCylinderField",
    "StencilFallbackWarning",
    "UnstructuredGrid",
    "UnstructuredPoloidalGrid",
    "add_k2_to_file",
    "build_grid_data",
    "dagp_unstructured",
    "family_grid",
    "family_spec",
    "gmsh_planes",
    "is_cylindrical",
    "k2_divergence",
    "make_test_grid",
    "make_unstructured_grid",
    "num_coefs",
    "slab_metric",
    "som_weights",
    "trace_maps",
    "write_unstructured",
]
