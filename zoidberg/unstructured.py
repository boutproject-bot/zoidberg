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
    plane ``j`` is stored at index ``(MXG + k, j, 0)``. ``MXG = MYG = 1``
    are written to the file. The ``MXG`` guard cells on either side are padding.
    Their ``cell_number`` is ``-1``, so no unstructured operator ever
    references them. Their geometric fields are copies of the nearest
    node, so all values are finite. ``n_nodes`` must be a multiple of
    ``multiple_of`` (default 4) so that ``NXPE`` in {1, 2, 4} divides the
    interior. Nodes are sorted in Morton (Z-order) to improve locality.

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
    plus one stabilisation point (compact two-point difference minus its
    high-order prediction) that removes growing grid-scale modes. Interior dual faces connect the centroids of the two adjacent
    triangles; at a boundary edge the face runs from the centroid to the edge
    midpoint, and the wall itself is zero-flux.

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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
from scipy.spatial import KDTree

from . import fieldtracer
from .field import Slab

# Gauss-Legendre and Boole rules on [0, 1]: (positions, weights)
_FACE_RULES = {
    "gauss2": (
        np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)]),
        np.array([0.5, 0.5]),
    ),
    "boole5": (
        np.linspace(0.0, 1.0, 5),
        np.array([7.0, 32.0, 12.0, 32.0, 7.0]) / 90.0,
    ),
}


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


def _gmsh_mesh(loops, counts, mesh_size, embed=None):
    """Triangulate the region bounded by closed curves with gmsh.

    ``loops[0]`` is the outer boundary, further loops are holes; each is a
    closed spline with ``counts[i]`` equally spaced nodes (transfinite). With
    ``embed``, these interior points are fixed mesh nodes and no other
    interior nodes are inserted. Returns ``(points, triangles,
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
        See :meth:`from_structured` and :meth:`mark_boundary_walls`.
    """

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
        seed=0,
        relax=False,
        morton=True,
        centroid_dofs=True,
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
        seed : int
            Unused by gmsh's deterministic Delaunay mesher; kept for API
            symmetry with :meth:`slab`
        relax : bool
            Apply :meth:`centroidal_relax` (wall nodes stay fixed). Not
            needed with ``centroid_dofs``
        morton : bool
            Sort the interior nodes in Morton order
        centroid_dofs : bool
            Place the unknowns at the dual-cell centroids
            (:meth:`use_centroid_dofs`), which gives second-order
            truncation error on the unrelaxed gmsh mesh

        The nodes are ordered inner wall, interior, outer wall, and
        ``walls`` is set accordingly. gmsh is imported only here.
        """
        del seed
        outer = np.asarray(outer, dtype=float)
        loops = [outer] if inner is None else [outer, np.asarray(inner, dtype=float)]
        if mesh_size is None:
            raise ValueError("mesh_size is required")
        counts = [n_outer, n_inner][: len(loops)]
        counts = [
            int(np.ceil(_curve_length(c) / mesh_size)) if n is None else int(n)
            for c, n in zip(loops, counts)
        ]
        pts, tri, wall_nodes = _gmsh_mesh(loops, counts, mesh_size)
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
        f = self.faces()
        wall = np.zeros((self.n_nodes, 2), dtype=bool)
        bnodes = np.unique(
            np.concatenate([f["i"][f["boundary"]], f["j"][f["boundary"]]])
        )
        for dim in range(2):
            if not np.isfinite(self.period[dim]):
                c = self.points[:, dim]
                span = np.ptp(c)
                on_wall = (np.abs(c - c.min()) < 1e-12 * span) | (
                    np.abs(c - c.max()) < 1e-12 * span
                )
                wall[on_wall, dim] = True
        del bnodes
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
    ):
        """Least-squares value (and gradient) weights at target points.

        Returns ``(inds, w_value[, w_dR, w_dZ][, w_eval], info)``, each of shape
        ``(m, k)``, with ``k = num_coefs(order) + extra``. With
        ``eval_offsets`` (shape ``(m, q, 2)``, physical offsets from each
        target), ``w_eval`` (shape ``(m, q, k)``) evaluates the same fitted
        polynomial at ``target + offset``. ``info`` contains
        ``fallback`` (number of points that dropped to a lower order) and
        ``snapped`` (number of injections). Value weights are exact for
        polynomials of total degree <= order.
        """
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        m = len(targets)
        k0 = num_coefs(order) + extra
        if k0 > self.n_nodes:
            raise ValueError(
                f"Need at least {k0} nodes for an order-{order} stencil, "
                f"mesh has {self.n_nodes}"
            )
        # Stencils grow before they drop in order: next to a wall, the k0
        # nearest nodes can span too few layers for a full-order fit.
        k_list = [kk for kk in (k0, k0 + 6, k0 + 14) if kk <= self.n_nodes]
        kmax = k_list[-1]
        tree = self._tree()
        _, inds_all = tree.query(self._scaled(targets), k=kmax)
        inds_all = np.atleast_2d(inds_all)
        offsets = self.minimum_image(self.dof[inds_all] - targets[:, None, :])
        u_all = offsets[..., 0] / self.h[0]
        v_all = offsets[..., 1] / self.h[1]

        inds = np.zeros((m, kmax), dtype=int)
        w_val = np.zeros((m, kmax))
        w_dR = np.zeros((m, kmax))
        w_dZ = np.zeros((m, kmax))
        if eval_offsets is not None:
            eval_offsets = np.asarray(eval_offsets, dtype=float)
            w_eval = np.zeros((m, eval_offsets.shape[1], kmax))
        done = np.zeros(m, dtype=bool)
        fallback = 0
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
            sv = np.linalg.svd(V, compute_uv=False)
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
            P = np.linalg.pinv(V[good])  # (mg, nc, kk)
            if weight_power:
                P = P * sw[good][:, None, :]
            inds[idx] = inds_all[idx]
            w_val[idx, :kk] = P[:, 0, :]
            w_dR[idx, :kk] = P[:, 1, :] / self.h[0]
            w_dZ[idx, :kk] = P[:, 2, :] / self.h[1]
            if eval_offsets is not None:
                off = eval_offsets[idx]
                mono = _monomials(off[..., 0] / self.h[0], off[..., 1] / self.h[1], p)
                w_eval[idx, :, :kk] = mono @ P
            done[idx] = True
            if p < order:
                fallback += len(idx)
        if fallback:
            warnings.warn(
                f"{fallback} of {m} stencils dropped below order {order}",
                StencilFallbackWarning,
            )
        u_all0 = u_all[:, 0]
        v_all0 = v_all[:, 0]

        snapped = np.zeros(m, dtype=bool)
        if snap_tol is not None:
            nearest = np.sqrt(u_all0**2 + v_all0**2)
            snapped = nearest < snap_tol
            w_val[snapped] = 0.0
            w_val[snapped, 0] = 1.0
        info = {"fallback": fallback, "snapped": int(np.sum(snapped))}
        out = (inds, w_val)
        if gradient:
            out += (w_dR, w_dZ)
        if eval_offsets is not None:
            out += (w_eval,)
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
    """A stack of identical-topology poloidal planes along y.

    ``poloidal`` is either one :class:`UnstructuredPoloidalGrid` (used for
    every plane) or a list of ``ny`` grids with the same ``n_nodes``.
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
            raise ValueError(
                f"{self.n_interior} evolving nodes is not a multiple of {self.multiple_of}"
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


def trace_maps(grid, field, rtol=1e-10, field_tracer=None, domain_tol=1e-9):
    """Trace field lines from every node to the neighbouring planes.

    Returns a dict with per-plane arrays of shape ``(ny, n_nodes)``:
    ``forward_R``, ``forward_Z``, ``forward_hit`` (0 inside, -1 inner wall,
    +1 outer wall), ``forward_half_R``/``forward_half_Z`` (positions half
    way to the next plane), and the same for ``backward``.

    End points outside the target plane's mesh are wall hits. On planes with
    marked ``walls`` the hit is on the wall with the nearest wall node,
    otherwise on the outer wall if R is above the mesh centre. End points
    within about ``domain_tol`` node spacings outside the mesh count as
    inside (their stencils extrapolate slightly); this plays the role of
    the manual correction of ``xt_prime`` next to traced flux-surface walls
    in structured stellarator grids.
    """
    tracer = field_tracer or fieldtracer.FieldTracer(field)
    ny, n = grid.ny, grid.n_nodes
    dy = grid.dy()
    out = {}
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
    grid, maps, order=4, strict=True, snap_tol=1e-10, extra=16, weight_power=3.0
):
    """Cell numbering, leg metadata and CSR operators for ``PetscOperators``.

    Uses the numbering of :func:`zoidberg.weights.assign_cell_space_numbers`:
    interior cells first (loop order x, y), then (none) X-boundary cells,
    then for each interior cell the backward and forward parallel boundary
    virtual cells. Legs are single-sample.
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


def _plane_dagp(
    pol,
    order=3,
    rule="gauss2",
    strict=True,
    stabilisation=1.0,
    weight_power=0.0,
    extra=16,
    isotropic=True,
    max_stabilisation=8.0,
    stability_tol=1e-10,
    cylindrical=False,
    stabilisation_form="polynomial",
    reconstruction="polynomial",
):
    """Per-plane CSR blocks (local node / face-point numbering).

    Each face contributes ``nq`` quadrature points, where the flux is
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
    Distance weighting (``weight_power``) was tested and makes stability
    worse; it is off by default.

    With ``cylindrical=True`` (fields with a major radius) volumes and face
    fluxes are weighted by R, giving (1/R) d/dR (R a df/dR) + d/dZ (a df/dZ).

    With ``isotropic=True`` (default) the least-squares fits use physical
    distances. Scaling R and Z separately by the node spacings (as for the
    parallel maps) amplifies errors on strongly anisotropic meshes, where
    the gradient across thin layers is divided by the small spacing.
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
    if isotropic:
        pol.h = np.full(2, np.sqrt(saved_h[0] * saved_h[1]))
        pol._kdtree = None
    try:
        f = pol.faces()
        vol = pol.dual_volumes(cylindrical=cylindrical)
        tq, wq = _FACE_RULES[rule]
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
            "weight_power": weight_power,
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
        # Least-squares gradients at the unknowns (for upwind reconstruction)
        n_inds, _, n_dR, n_dZ, n_info = pol.lsq_weights(pol.dof, **lsq)
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
        fallback = info["fallback"] + n_info["fallback"]
        beta = 0.0
        max_re = None

        if stabilisation:
            mid = pol.wrap(xi + 0.5 * edge) if mid_s is None else mid_s
            c = np.sum(normal**2, axis=1) / np.sum(edge_s * normal, axis=1)
            if cylindrical:
                c = c * mid[:, 0]
            if stabilisation_form == "polynomial":
                # Prediction P(x_j) - P(x_i) of the order-p fit around the
                # midpoint: the bracket is O(h^(p+1))
                half = 0.5 * edge_s[:, None, :]
                m_inds, m_val, m_dR, m_dZ, m_eval, m_info = pol.lsq_weights(
                    mid, eval_offsets=np.concatenate([-half, half], axis=1), **lsq
                )
                m_grad = -c[:, None] * (m_eval[:, 1, :] - m_eval[:, 0, :])
            elif stabilisation_form == "midpoint":
                m_inds, m_val, m_dR, m_dZ, m_info = pol.lsq_weights(mid, **lsq)
                m_grad = -c[:, None] * (edge_s[:, :1] * m_dR + edge_s[:, 1:] * m_dZ)
            else:
                raise ValueError(f"Unknown stabilisation_form '{stabilisation_form}'")
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

        # Upwind reconstruction of a at each face point from either side
        if reconstruction == "polynomial":
            # Order-p fit around the side's unknown, evaluated at the face
            # point: O(h^(p+1)). (The linear form below is only O(h^2) per
            # face point, which on irregular meshes gives an O(h)
            # truncation error once upwinding is on.)
            lsq_v = dict(lsq, gradient=False)

            def recon(node, offset):
                r_inds, _, w_e, r_info = pol.lsq_weights(
                    pol.dof[node], eval_offsets=offset[:, None, :], **lsq_v
                )
                return r_inds, w_e[:, 0, :], r_info["fallback"]

        elif reconstruction == "linear":
            # a_side + (x_f - x_side) . grad a(side)
            def recon(node, offset):
                cols = np.column_stack([node, n_inds[node]])
                w = np.column_stack(
                    [
                        np.ones(len(node)),
                        offset[:, :1] * n_dR[node] + offset[:, 1:] * n_dZ[node],
                    ]
                )
                return cols, w, 0

        else:
            raise ValueError(f"Unknown reconstruction '{reconstruction}'")
        # Offsets from the unknowns (nodes or dual-cell centroids)
        off_l, off_r = pos_i, pos_i - edge_fp
        if shift is not None:
            off_l, off_r = off_l + shift[fi], off_r + shift[fj]
        left_inds, left_w, fb_l = recon(fi, off_l)
        right_inds, right_w, fb_r = recon(fj, off_r)
        fallback += fb_l + fb_r
    finally:
        pol.h = saved_h
        pol._kdtree = None
    owner_node = np.minimum(fi, fj)

    return {
        "n_fp": len(fi),
        "inds": inds,
        "w_val": w_val,
        "w_grad": w_grad,
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
    rule="gauss2",
    strict=True,
    **plane_options,
):
    """Global CSR matrices for ``FV::Div_a_Grad_perp_FCI_unstructured``.

    Only evolving nodes get a divergence row. Face points between two
    Dirichlet wall nodes are dropped; faces between a wall node and an
    evolving node use the wall value, which BOUT++ holds in the guard cell.
    """
    interior = grid.interior
    cache = {}
    blocks = []
    for y in range(grid.ny):
        pol = grid.plane(y)
        if id(pol) not in cache:
            cache[id(pol)] = _plane_dagp(
                pol, order=order, rule=rule, strict=strict, **plane_options
            )
        blocks.append(cache[id(pol)])

    keeps = [interior[b["fi"]] | interior[b["fj"]] for b in blocks]
    n_fp_total = int(sum(np.sum(k) for k in keeps))
    gr, gc, gv = [], [], []
    vr, vc, vv = [], [], []
    dr, dc, dv = [], [], []
    lr, lc, lv, rr, rc, rv = [], [], [], [], [], []  # upwind reconstructions
    fr, fc, fR, fZ = [], [], [], []  # cell-averaged flux vector
    owner = np.empty(n_fp_total, dtype=int)
    volume = np.zeros((grid.ny, grid.n_nodes))
    offset = 0
    for y, (b, keep) in enumerate(zip(blocks, keeps)):
        cells = cell_number[grid.node_x, y, 0]
        nfp = int(np.sum(keep))
        rows = offset + np.arange(nfp)
        inds, fi, fj = b["inds"][keep], b["fi"][keep], b["fj"][keep]
        k = inds.shape[1]
        gr.append(np.repeat(rows, k))
        gc.append(cells[inds].ravel())
        gv.append(b["w_grad"][keep].ravel())
        vr.append(np.repeat(rows, k))
        vc.append(cells[inds].ravel())
        vv.append(b["w_val"][keep].ravel())
        mi, mj = interior[fi], interior[fj]
        dr.append(cells[fi][mi])
        dc.append(rows[mi])
        dv.append(1.0 / b["vol"][fi][mi])
        dr.append(cells[fj][mj])
        dc.append(rows[mj])
        dv.append(-1.0 / b["vol"][fj][mj])
        kl = b["left_inds"].shape[1]
        lr.append(np.repeat(rows, kl))
        lc.append(cells[b["left_inds"][keep]].ravel())
        lv.append(b["left_w"][keep].ravel())
        rr.append(np.repeat(rows, kl))
        rc.append(cells[b["right_inds"][keep]].ravel())
        rv.append(b["right_w"][keep].ravel())
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
        owner[rows] = cells[owner_node]
        volume[y] = b["vol"]
        offset += nfp

    def build(r, c, v, n_rows):
        r, c, v = np.concatenate(r), np.concatenate(c), np.concatenate(v)
        keep = np.abs(v) > 1e-300
        return _csr(n_rows, r[keep], c[keep], v[keep])

    out = {"dagp_us_n_facepoints": n_fp_total, "dagp_us_facepoint_owner": owner}
    for name, (r, c, v, nrow, ncol) in {
        "gradp": (gr, gc, gv, n_fp_total, total_cells),
        "value": (vr, vc, vv, n_fp_total, total_cells),
        "div": (dr, dc, dv, total_cells, n_fp_total),
        "left": (lr, lc, lv, n_fp_total, total_cells),
        "right": (rr, rc, rv, n_fp_total, total_cells),
        "flowR": (fr, fc, fR, total_cells, n_fp_total),
        "flowZ": (fr, fc, fZ, total_cells, n_fp_total),
    }.items():
        ptr, cols, vals = build(r, c, v, nrow)
        out[f"dagp_us_{name}_rows"] = ptr
        out[f"dagp_us_{name}_columns"] = cols
        out[f"dagp_us_{name}_weights"] = vals
        out[f"dagp_us_{name}_ncols"] = ncol
    out["dagp_us_volume"] = grid.to3d(volume)
    stats = {
        "fallback": sum(b["fallback"] for b in cache.values()),
        "beta": max(b["beta"] for b in cache.values()),
        "max_re": max((b["max_re"] or 0.0) for b in cache.values()),
    }
    return out, stats


# ----------------------------------------------------------------------
# Metric


def is_cylindrical(field):
    """True for fields in cylindrical coordinates (y is the toroidal angle)."""
    return field.Rfunc(np.ones(1), np.zeros(1), 0.0) is not None


def slab_metric(grid, field, maps, volume):
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
        if cylindrical:
            Byh = Byh * maps[f"{name}_half_R"]
        metric[f"By{post}"] = grid.to3d(Byh)
        metric[f"g_22{post}"] = grid.to3d((Bh / Byh) ** 2)

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
# Top level


def build_grid_data(
    grid,
    field,
    map_order=4,
    dagp_order=4,
    rule="gauss2",
    strict=True,
    rtol=1e-10,
    domain_tol=1e-9,
    map_extra=16,
    map_weight_power=3.0,
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
    (see :meth:`UnstructuredPoloidalGrid.lsq_weights`).
    """
    maps = trace_maps(grid, field, rtol=rtol, domain_tol=domain_tol)
    som, som_stats = som_weights(
        grid,
        maps,
        order=map_order,
        strict=strict,
        extra=map_extra,
        weight_power=map_weight_power,
    )
    dagp, dagp_stats = dagp_unstructured(
        grid,
        som["cell_number"],
        som["total_cells"],
        order=dagp_order,
        rule=rule,
        strict=strict,
        cylindrical=is_cylindrical(field),
    )
    volume = np.array([grid.plane(j).dual_volumes() for j in range(grid.ny)])
    data = {}
    data.update(slab_metric(grid, field, maps, volume))
    data.update(som)
    data.update(dagp)

    # Positions of the unknowns (the nodes, or the dual-cell centroids)
    R = np.array([grid.plane(j).dof[:, 0] for j in range(grid.ny)])
    Z = np.array([grid.plane(j).dof[:, 1] for j in range(grid.ny)])
    data["R"] = grid.to3d(R)
    data["Z"] = grid.to3d(Z)
    data["Rxy"] = data["R"]
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
    data["unstructured"] = 1
    data["MXG"] = grid.MXG
    data["MYG"] = grid.MYG
    data["n_nodes"] = grid.n_nodes
    stats = {
        "n_nodes": grid.n_nodes,
        "ny": grid.ny,
        "som": som_stats,
        "dagp": dagp_stats,
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
    rule="gauss2",
    strict=True,
    multiple_of=4,
    dirichlet_walls=False,
    domain_tol=1e-9,
):
    """Convenience wrapper: build the 3D grid, compute and write all data.

    ``poloidal`` is one :class:`UnstructuredPoloidalGrid` for every plane or
    a list of ``ny`` of them (e.g. from :func:`gmsh_planes`).
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
        rule=rule,
        strict=strict,
        domain_tol=domain_tol,
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
):
    """Mesh every poloidal cross-section independently with gmsh.

    ``outers[j]`` (and ``inners[j]``, or ``None`` for discs) are the closed
    wall curves of plane ``j`` (see :meth:`UnstructuredPoloidalGrid.from_gmsh`).
    The grid format stores nodes along x, so all planes need the same number
    of nodes and of wall nodes. The wall-node counts are fixed from the
    longest curve of all planes; the planes are meshed, and those with fewer
    nodes than the largest are re-meshed with the missing nodes added in
    their largest triangles. The total is rounded up so that the number of
    evolving (non-wall) nodes is a multiple of ``multiple_of``. ``relax``
    and ``centroid_dofs``: see :meth:`UnstructuredPoloidalGrid.from_gmsh`.
    """
    ny = len(outers)
    if inners is None:
        inners = [None] * ny
    n_outer = int(np.ceil(max(_curve_length(c) for c in outers) / mesh_size))
    n_inner = None
    if inners[0] is not None:
        n_inner = int(np.ceil(max(_curve_length(c) for c in inners) / mesh_size))
    kw = {"mesh_size": mesh_size, "n_outer": n_outer, "n_inner": n_inner}
    kw["centroid_dofs"] = False  # set after relaxing
    planes = [
        UnstructuredPoloidalGrid.from_gmsh(o, i, relax=False, **kw)
        for o, i in zip(outers, inners)
    ]
    n_walls = n_outer + (n_inner or 0)
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


# Cylindrical test grids: pure toroidal field (Screwpinch without shear) on
# an annulus R in [1, 2] with Dirichlet walls, periodic in Z (Lz = 1).
BOUT_MMS_CYL = {"R0": 1.0, "Lx": 1.0, "Lz": 1.0, "ny": 4, "jitter": 0.2, "seed": 4321}


def make_test_grid(family, n, filename=None, **kwargs):
    """Shared entry point for test grids (used by BOUT-dev and Hermes-3).

    Families: ``"bout_mms"`` (MMS-spatial-fci-unstructured, Slab field) and
    ``"bout_mms_cyl"`` (cylindrical case of that test). Other families are
    built by the callers with :func:`make_unstructured_grid`.
    """
    if family == "bout_mms_cyl":
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
        return make_unstructured_grid(
            filename,
            pol,
            field,
            ny=p["ny"],
            Ly=2.0 * np.pi,
            yperiodic=True,
            dirichlet_walls=True,
        )
    if family != "bout_mms":
        raise ValueError(f"Unknown test-grid family '{family}'")
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
    return make_unstructured_grid(
        filename, pol, field, ny=n, Ly=p["Ly"], yperiodic=True
    )


__all__ = [
    "BOUT_MMS",
    "BOUT_MMS_CYL",
    "StencilFallbackWarning",
    "UnstructuredGrid",
    "UnstructuredPoloidalGrid",
    "build_grid_data",
    "dagp_unstructured",
    "gmsh_planes",
    "is_cylindrical",
    "make_test_grid",
    "make_unstructured_grid",
    "num_coefs",
    "slab_metric",
    "som_weights",
    "trace_maps",
    "write_unstructured",
]
