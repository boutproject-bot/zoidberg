import warnings
from pathlib import Path

import numpy as np
import pytest

from . import field as zbfield
from . import unstructured as us


def periodic_slab(n=12, jitter=0.2, seed=3):
    return us.UnstructuredPoloidalGrid.slab(
        n,
        n,
        1.0,
        1.0,
        R0=0.0,
        Z0=0.0,
        jitter=jitter,
        jitter_dirs="rz",
        periodic_R=True,
        periodic_Z=True,
        seed=seed,
    )


def strip(nR=8, nZ=16, seed=5):
    # Same shape as the BOUT-dev MMS grids: few fixed R layers, periodic Z
    return us.UnstructuredPoloidalGrid.slab(
        nR,
        nZ,
        0.1,
        1.0,
        R0=-0.05,
        Z0=0.0,
        jitter=0.2,
        jitter_dirs="z",
        periodic_R=False,
        periodic_Z=True,
        seed=seed,
    )


def poly(R, Z, deg):
    rng = np.random.default_rng(42)
    total = np.zeros_like(R)
    for d in range(deg + 1):
        for j in range(d + 1):
            total += rng.normal() * R ** (d - j) * Z**j
    return total


@pytest.mark.parametrize("make", [periodic_slab, strip])
def test_volumes_tile_domain(make):
    pol = make()
    vol = pol.dual_volumes()
    assert np.all(vol > 0)
    assert np.isclose(vol.sum(), pol.total_area(), rtol=1e-12)


def test_euler_characteristic():
    assert periodic_slab().euler_characteristic() == 0  # torus
    assert strip().euler_characteristic() == 0  # annulus / periodic strip
    disc = us.UnstructuredPoloidalGrid.slab(
        6, 8, 1.0, 1.0, periodic_Z=False, jitter=0.2, seed=1
    )
    assert disc.euler_characteristic() == 1


def test_face_orientation_closes_cells():
    # Sum of outward normals around every closed (interior) dual cell is 0
    pol = periodic_slab()
    f = pol.faces()
    d = f["end"] - f["start"]
    normal = np.column_stack((d[:, 1], -d[:, 0]))
    total = np.zeros((pol.n_nodes, 2))
    np.add.at(total, f["i"], normal)
    np.add.at(total, f["j"], -normal)
    assert np.max(np.abs(total)) < 1e-12


def test_lsq_cubic_exact_interior_and_seam():
    pol = periodic_slab(n=16)
    rng = np.random.default_rng(0)
    # Targets well inside, and near the periodic seams. Use a polynomial in
    # the minimum-image offset from each target so periodicity is irrelevant.
    targets = np.vstack(
        [rng.uniform(0.0, 1.0, (50, 2)), rng.uniform(-0.02, 0.02, (20, 2)) % 1.0]
    )
    inds, w, info = pol.lsq_weights(targets, order=3, snap_tol=None, strict=True)
    assert info["fallback"] == 0
    off = pol.minimum_image(pol.points[inds] - targets[:, None, :])
    for deg in range(4):
        vals = poly(off[..., 0], off[..., 1], deg)
        exact = poly(np.zeros(len(targets)), np.zeros(len(targets)), deg)
        assert np.max(np.abs(np.sum(w * vals, axis=1) - exact)) < 1e-10


def test_lsq_gradient_exact():
    pol = strip()
    rng = np.random.default_rng(1)
    targets = np.column_stack((rng.uniform(-0.04, 0.04, 30), rng.uniform(0, 1, 30)))
    inds, _w, wR, wZ, _info = pol.lsq_weights(
        targets, order=3, gradient=True, snap_tol=None, strict=True
    )
    off = pol.minimum_image(pol.points[inds] - targets[:, None, :])
    f = 3 * off[..., 0] ** 3 - 2 * off[..., 0] * off[..., 1] ** 2 + off[..., 1] + 0.5
    # d/dR and d/dZ at offset 0: 0 and 1
    assert np.max(np.abs(np.sum(wR * f, axis=1))) < 1e-8
    assert np.max(np.abs(np.sum(wZ * f, axis=1) - 1.0)) < 1e-8


def test_injection_snap():
    pol = strip()
    targets = pol.points[[0, 5, 17]]
    inds, w, info = pol.lsq_weights(targets, order=3)
    assert info["snapped"] == 3
    assert np.all(inds[:, 0] == [0, 5, 17])
    assert np.all(w[:, 0] == 1.0) and np.all(w[:, 1:] == 0.0)


def test_injection_maps_in_grid_file(tmp_path):
    # Straight field: every field line ends on a node of the next plane, so
    # each interior map row written to the grid file must be an exact
    # injection (one entry, weight 1.0) onto the node at the traced end point
    pol = strip(nR=8, nZ=8)
    field = zbfield.Slab(By=1.0, Bz=0.0, Bzprime=0.0)
    _grid, d, _stats = us.make_unstructured_grid(
        str(tmp_path / "g.nc"), pol, field, ny=4, Ly=10.0
    )
    cell = np.asarray(d["cell_number"])[..., 0]
    R, Z = np.asarray(d["R"])[..., 0], np.asarray(d["Z"])[..., 0]
    node = {int(c): xy for xy, c in np.ndenumerate(cell) if c >= 0}
    ny = cell.shape[1]
    for name, step in (("forward", 1), ("backward", -1)):
        rows = np.asarray(d[f"{name}_rows"])
        cols = np.asarray(d[f"{name}_columns"])
        weights = np.asarray(d[f"{name}_weights"])
        legs = np.asarray(d[f"{name}_leg_interior_number"])[..., 0]
        tR, tZ = np.asarray(d[f"{name}_R"])[..., 0], np.asarray(d[f"{name}_Z"])[..., 0]
        assert np.all(np.diff(rows) == 1)
        assert np.all(weights == 1.0)
        n_legs = 0
        for (x, y), leg in np.ndenumerate(legs):
            if leg < 0:
                continue
            n_legs += 1
            xt, yt = node[int(cols[rows[leg]])]
            assert yt == (y + step) % ny
            assert R[xt, yt] == pytest.approx(tR[x, y], abs=1e-12)
            assert Z[xt, yt] == pytest.approx(tZ[x, y], abs=1e-12)
        assert n_legs == np.count_nonzero(cell >= 0)


def test_dagp_laplacian_quadratic():
    # div(grad(R^2 + Z^2)) = 4, with a = 1, on a doubly periodic mesh using a
    # locally defined quadratic (minimum image from each node). Instead,
    # check a periodic function against its analytic Laplacian and require
    # second-order convergence.
    errs = []
    for n in (16, 32):
        pol = periodic_slab(n=n)
        grid = us.UnstructuredGrid(pol, np.array([0.0]), 1.0, multiple_of=4)
        cell = np.arange(pol.n_nodes).reshape(1, -1, 1)
        cell_number = np.full(grid.shape, -1)
        cell_number[1:-1, 0, 0] = np.arange(pol.n_nodes)
        d, stats = us.dagp_unstructured(grid, cell_number, pol.n_nodes, strict=True)
        assert stats["fallback"] == 0
        from scipy.sparse import csr_matrix

        def mat(name, shape, d=d):
            return csr_matrix(
                (
                    d[f"dagp_us_{name}_weights"],
                    d[f"dagp_us_{name}_columns"],
                    d[f"dagp_us_{name}_rows"],
                ),
                shape=shape,
            )

        nfp = d["dagp_us_n_facepoints"]
        G = mat("gradp", (nfp, pol.n_nodes))
        Vv = mat("value", (nfp, pol.n_nodes))
        D = mat("div", (pol.n_nodes, nfp))
        R, Z = pol.R, pol.Z
        k = 2 * np.pi
        f = np.sin(k * R) * np.cos(k * Z)
        a = 1.0 + 0.3 * np.cos(k * R)
        res = D @ ((Vv @ a) * (G @ f))
        # div(a grad f) = a lap f + grad a . grad f
        lap = -2 * k**2 * f
        dadR = -0.3 * k * np.sin(k * R)
        dfdR = k * np.cos(k * R) * np.cos(k * Z)
        exact = a * lap + dadR * dfdR
        errs.append(np.sqrt(np.mean((res - exact) ** 2)))
        # Conservation: sum V * result = 0 on a closed domain
        V = pol.dual_volumes()
        assert abs(np.sum(V * res)) < 1e-10 * np.sum(V * np.abs(res))
        del cell
    order = np.log(errs[0] / errs[1]) / np.log(2)
    assert order > 1.8, (errs, order)


def test_bout_mms_grids_no_fallback():
    # The anisotropic BOUT-dev MMS grids must not need lower-order stencils
    for n in (8, 16):
        with warnings.catch_warnings():
            warnings.simplefilter("error", us.StencilFallbackWarning)
            _grid, _data, stats = us.make_test_grid("bout_mms", n, filename=None)
        assert stats["som"]["fallback"] == 0
        assert stats["dagp"]["fallback"] == 0
        # No growing modes, even on the coarse, strongly anisotropic grids
        assert stats["dagp"]["max_re"] <= 1e-10
        assert stats["forward_hits"] == 0 and stats["backward_hits"] == 0


def test_bout_mms_iso_grids():
    # Isotropic family: nR = n/8 (at least 4), jitter in R and Z, nodes on the
    # R walls; no stencil fallbacks, no growing modes, no wall hits
    for n in (32, 48):
        with warnings.catch_warnings():
            warnings.simplefilter("error", us.StencilFallbackWarning)
            grid, _data, stats = us.make_test_grid("bout_mms_iso", n, filename=None)
        pol = grid.plane(0)
        R = np.unique(np.round(pol.R, 12))
        p = us.BOUT_MMS_ISO
        assert np.isclose(pol.R.min(), p["R0"]) and np.isclose(
            pol.R.max(), p["R0"] + p["Lx"]
        )
        # interior nodes are jittered in R (not on nR fixed lines)
        assert len(R) > max(4, n // 8)
        assert pol.n_nodes == max(4, n // 8) * n
        assert stats["som"]["fallback"] == 0 and stats["dagp"]["fallback"] == 0
        assert stats["dagp"]["max_re"] <= 1e-10
        assert stats["forward_hits"] == 0 and stats["backward_hits"] == 0


def test_contract_variables_finite(tmp_path):
    pol = strip(nR=8, nZ=8)
    field = zbfield.Slab(Bz=0.05, Bzprime=0.1)
    grid, data, _stats = us.make_unstructured_grid(
        str(tmp_path / "g.nc"), pol, field, ny=4, Ly=10.0
    )
    required = [
        "unstructured",
        "MXG",
        "MYG",
        "R",
        "Z",
        "forward_R",
        "backward_Z",
        "forward_xt_prime",
        "backward_zt_prime",
        "cell_number",
        "total_cells",
        "forward_cell_number",
        "forward_leg_interior_number",
        "forward_rows",
        "forward_columns",
        "forward_weights",
        "forward_leg_weights",
        "dagp_us_div_rows",
        "dagp_us_gradp_weights",
        "dagp_us_value_ncols",
        "dagp_us_facepoint_owner",
        "dagp_us_volume",
        "dx",
        "dy",
        "dz",
        "g_22",
        "Bxy",
        "forward_g_22",
        "backward_Bxy",
        "forward_dy",
        "By",
        "By_cell_ylow",
        "By_cell_yhigh",
        "g_22_cell_ylow",
        "g_22_cell_yhigh",
    ]
    for name in required:
        assert name in data, name
        v = np.asarray(data[name])
        assert np.all(np.isfinite(v)), name
    assert data["cell_number"].shape == grid.shape
    # BOUT++ inserts CSR entries with INSERT_VALUES: columns must be unique
    for name in (
        "forward",
        "backward",
        "dagp_us_div",
        "dagp_us_gradp",
        "dagp_us_value",
    ):
        rows, cols = data[f"{name}_rows"], data[f"{name}_columns"]
        for r in range(len(rows) - 1):
            c = cols[rows[r] : rows[r + 1]]
            assert len(np.unique(c)) == len(c), f"{name} row {r}"
    assert np.all(data["cell_number"][0] == -1) and np.all(
        data["cell_number"][-1] == -1
    )
    from boututils.datafile import DataFile

    with DataFile(str(tmp_path / "g.nc")) as f:
        assert int(f["nz"]) == 1
        assert int(f["nx"]) == pol.n_nodes + 2
        assert int(f["unstructured"]) == 1


def test_walls_y_marks_boundaries():
    pol = us.UnstructuredPoloidalGrid.slab(
        6, 8, 1.0, 1.0, R0=1.5, Z0=-0.5, jitter=0.2, periodic_Z=False, seed=2
    )
    field = zbfield.Slab(Bz=0.0, Bzprime=0.0)
    grid, data, stats = us.make_unstructured_grid(
        None, pol, field, ny=8, Ly=100.0, walls_y=True
    )
    assert stats["forward_hits"] == pol.n_nodes
    assert stats["backward_hits"] == pol.n_nodes
    assert np.all(data["forward_xt_prime"][1:-1, -1, 0] == grid.nx)
    assert stats["som"]["snapped"] == 2 * pol.n_nodes * 7


def test_dagp_stable():
    # The discrete perpendicular Laplacian must not have growing modes
    # (eigenvalues with positive real part), otherwise time stepping diverges
    from scipy.sparse import csr_matrix

    for n in (16, 24):
        pol = periodic_slab(n=n)
        grid = us.UnstructuredGrid(pol, np.array([0.0]), 1.0)
        cell_number = np.full(grid.shape, -1)
        cell_number[1:-1, 0, 0] = np.arange(pol.n_nodes)
        d, _ = us.dagp_unstructured(grid, cell_number, pol.n_nodes)
        nfp = d["dagp_us_n_facepoints"]

        def mat(name, shape, d=d):
            return csr_matrix(
                (
                    d[f"dagp_us_{name}_weights"],
                    d[f"dagp_us_{name}_columns"],
                    d[f"dagp_us_{name}_rows"],
                ),
                shape=shape,
            )

        L = mat("div", (pol.n_nodes, nfp)) @ mat("gradp", (nfp, pol.n_nodes))
        ev = np.linalg.eigvals(L.toarray())
        assert ev.real.max() < 1e-8 * np.abs(ev.real).max()


def test_dirichlet_walls_layout_and_order():
    # Wall nodes go into the x guard cells and get no divergence row; with
    # exact wall values the operator is second order on the evolving nodes
    from scipy.sparse import csr_matrix

    errs = []
    for n in (16, 32):
        pol = us.UnstructuredPoloidalGrid.slab(
            n + 1,
            n,
            1.0,
            1.0,
            R0=0.0,
            jitter=0.2,
            jitter_dirs="rz",
            periodic_R=False,
            periodic_Z=True,
            seed=7 + n,
            r_on_walls=True,
        )
        grid = us.UnstructuredGrid(pol, np.array([0.0]), 1.0, dirichlet_walls=True)
        assert grid.MXG == n and grid.n_interior == (n - 1) * n
        assert grid.nx == grid.n_interior + 2 * n
        field = zbfield.Slab(Bz=0.0, Bzprime=0.0)
        data, stats = us.build_grid_data(grid, field)
        assert stats["dagp"]["max_re"] <= 1e-10
        cn = data["cell_number"][:, 0, 0]
        assert np.all(cn[: grid.MXG] >= 0) and np.all(cn[-grid.MXG :] >= 0)
        nc, nfp = data["total_cells"], data["dagp_us_n_facepoints"]

        def mat(name, shape, data=data):
            return csr_matrix(
                (
                    data[f"dagp_us_{name}_weights"],
                    data[f"dagp_us_{name}_columns"],
                    data[f"dagp_us_{name}_rows"],
                ),
                shape=shape,
            )

        G, V, D = (
            mat("gradp", (nfp, nc)),
            mat("value", (nfp, nc)),
            mat("div", (nc, nfp)),
        )
        node_cells = cn[grid.node_x]
        k = 2 * np.pi
        f = np.zeros(nc)
        f[node_cells] = np.sin(k * pol.R) * np.cos(k * pol.Z)
        res = (D @ (V @ np.ones(nc) * (G @ f)))[node_cells]
        ex = -2 * k * k * np.sin(k * pol.R) * np.cos(k * pol.Z)
        assert np.all(res[~grid.interior] == 0.0)  # walls are not evolved
        e = (res - ex)[grid.interior]
        errs.append(np.sqrt(np.mean(e**2)))
    assert np.log2(errs[0] / errs[1]) > 1.8, errs


def test_dagp_flows_and_conservation():
    # Flux conservation and the cell-averaged flux vector (exact for linear f)
    from scipy.sparse import csr_matrix

    pol = periodic_slab(n=12)
    grid = us.UnstructuredGrid(pol, np.array([0.0]), 1.0)
    cell_number = np.full(grid.shape, -1)
    cell_number[1:-1, 0, 0] = np.arange(pol.n_nodes)
    d, _ = us.dagp_unstructured(grid, cell_number, pol.n_nodes)
    n, nfp = pol.n_nodes, d["dagp_us_n_facepoints"]

    def mat(name, shape):
        return csr_matrix(
            (
                d[f"dagp_us_{name}_weights"],
                d[f"dagp_us_{name}_columns"],
                d[f"dagp_us_{name}_rows"],
            ),
            shape=shape,
        )

    G, Vv, D = mat("gradp", (nfp, n)), mat("value", (nfp, n)), mat("div", (n, nfp))
    L, Rt = mat("left", (nfp, n)), mat("right", (nfp, n))
    FR, FZ = mat("flowR", (n, nfp)), mat("flowZ", (n, nfp))
    V = pol.dual_volumes()
    rng = np.random.default_rng(3)
    f, a = rng.normal(size=n), 1.0 + rng.uniform(size=n)
    flux = (Vv @ a) * (G @ f)
    # Conservation: every face-point flux leaves one cell and enters another
    assert abs(np.sum(V * (D @ flux))) < 1e-10 * np.sum(V * np.abs(D @ flux))
    # Left/right reconstructions are exact for constants and (locally) linear a
    assert np.allclose(L @ np.ones(n), 1) and np.allclose(Rt @ np.ones(n), 1)
    # Linear f with a = 1 (local, so use minimum-image offsets from node 0):
    # the flux vector -grad f = (-2, 3) is reproduced at every node
    off = pol.minimum_image(pol.points - pol.points[0])
    mask = np.all(np.abs(off) < 0.25, axis=1)  # away from the periodic seam
    flin = 2.0 * off[:, 0] - 3.0 * off[:, 1]
    flux = Vv @ np.ones(n) * (G @ flin)
    qR, qZ = FR @ flux, FZ @ flux
    inner = mask & np.all(np.abs(off) < 0.1, axis=1)
    assert np.allclose(qR[inner], -2.0, atol=1e-9)
    assert np.allclose(qZ[inner], 3.0, atol=1e-9)
    # Both reconstructions of a linear a at a face point are exact, so they
    # agree wherever their stencils avoid the periodic seam
    alin = 1.0 + 0.5 * off[:, 0] + 0.25 * off[:, 1]
    smooth = np.all(np.abs(off) < 0.45, axis=1)  # linear away from the seam
    outside = (~smooth).astype(float)
    rows = np.flatnonzero((abs(L) @ outside == 0) & (abs(Rt) @ outside == 0))
    assert len(rows) > 0
    assert np.allclose((L @ alin)[rows], (Rt @ alin)[rows], atol=1e-9)


def test_cylindrical_operator_and_metric():
    # Pure toroidal field B = 1/R (field lines at constant R, Z): the metric
    # is g_22 = R^2 and the perpendicular operator is the cylindrical one,
    # (1/R) d/dR (R a df/dR) + a d2f/dZ2, second order on evolving nodes
    from scipy.sparse import csr_matrix

    field = zbfield.Screwpinch(xcentre=1.5, shear=0.0, Btor=1.0)
    errs = []
    for n in (12, 24):
        pol = us.UnstructuredPoloidalGrid.slab(
            n + 1,
            n,
            1.0,
            1.0,
            R0=1.0,
            jitter=0.2,
            jitter_dirs="rz",
            periodic_R=False,
            periodic_Z=True,
            seed=11 + n,
            r_on_walls=True,
        )
        grid = us.UnstructuredGrid(
            pol,
            np.linspace(0, 2 * np.pi, 4, endpoint=False),
            2 * np.pi,
            dirichlet_walls=True,
        )
        data, stats = us.build_grid_data(grid, field)
        assert stats["dagp"]["max_re"] <= 1e-10
        R3 = data["R"]
        assert np.allclose(data["g_22"], R3**2)
        assert np.allclose(data["forward_g_22"], data["forward_R"] ** 2)
        # Staggered metric (field lines at constant R): g_22 at the half-step
        # points equals g_22 = R^2, and By_cell is B_y = 1/R itself
        for post in ("_cell_yhigh", "_cell_ylow"):
            assert np.allclose(data[f"g_22{post}"], data["g_22"])
            assert np.allclose(data[f"By{post}"], 1.0 / R3)
        nc, nfp = data["total_cells"], data["dagp_us_n_facepoints"]

        def mat(name, shape, data=data):
            return csr_matrix(
                (
                    data[f"dagp_us_{name}_weights"],
                    data[f"dagp_us_{name}_columns"],
                    data[f"dagp_us_{name}_rows"],
                ),
                shape=shape,
            )

        G, V, D = (
            mat("gradp", (nfp, nc)),
            mat("value", (nfp, nc)),
            mat("div", (nc, nfp)),
        )
        cells = data["cell_number"][grid.node_x, 0, 0]
        k = 2 * np.pi
        R, Z = pol.R, pol.Z
        fv = np.sin(k * R) * np.cos(k * Z)
        av = 1.0 + 0.3 * np.cos(k * R)
        f, a = np.zeros(nc), np.zeros(nc)
        f[cells], a[cells] = fv, av
        res = (D @ ((V @ a) * (G @ f)))[cells]
        fR = k * np.cos(k * R) * np.cos(k * Z)
        fRR = -k * k * fv
        aR = -0.3 * k * np.sin(k * R)
        ex = (a[cells] * fR) / R + aR * fR + av * fRR - av * k * k * fv
        e = (res - ex)[grid.interior]
        errs.append(np.sqrt(np.mean(e**2)))
    assert np.log2(errs[0] / errs[1]) > 1.8, errs


def _curved_annulus(shift=0.0, n=200):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    outer = np.column_stack(
        (2.0 + 0.45 * np.cos(th) + 0.04 * np.cos(3 * th), (0.38 + shift) * np.sin(th))
    )
    inner = np.column_stack(
        (2.05 + shift + 0.15 * np.cos(th), 0.02 + 0.12 * np.sin(th))
    )
    return outer, inner


def test_gmsh_curved_walls_order():
    # gmsh annulus with two curved walls: walls ordered inner first, outer
    # last, unmoved by relaxation, Dirichlet in the guard cells; the operator
    # is second order on the evolving nodes and has no growing modes
    pytest.importorskip("gmsh")
    from scipy.sparse import csr_matrix

    outer, inner = _curved_annulus()
    errs = []
    hs = (1 / 12, 1 / 24)
    for h in hs:
        pol = us.UnstructuredPoloidalGrid.from_gmsh(outer, inner, mesh_size=h)
        w_in, w_out = pol.walls
        assert np.array_equal(w_in, np.arange(len(w_in)))
        assert np.array_equal(w_out, pol.n_nodes - len(w_out) + np.arange(len(w_out)))
        # Inner-wall nodes on the inner (elliptic) wall curve
        r_in = np.hypot((pol.R[w_in] - 2.05) / 0.15, (pol.Z[w_in] - 0.02) / 0.12)
        assert np.allclose(r_in, 1.0, atol=1e-4)
        relaxed = us.UnstructuredPoloidalGrid.from_gmsh(
            outer, inner, mesh_size=h, relax=True
        )
        walls = np.concatenate(pol.walls)
        assert np.array_equal(pol.points[walls], relaxed.points[walls])
        # Unknowns at the dual-cell centroids, wall nodes on the wall
        assert pol.centroid_dofs
        assert np.array_equal(pol.dof[walls], pol.points[walls])
        grid = us.UnstructuredGrid(
            pol, np.array([0.0]), 1.0, dirichlet_walls=True, multiple_of=1
        )
        assert grid.MXG == max(len(w_in), len(w_out))
        assert not np.any(grid.interior[walls])
        data, stats = us.build_grid_data(grid, zbfield.Slab(Bz=0.0, Bzprime=0.0))
        assert stats["dagp"]["max_re"] <= 1e-10
        assert stats["dagp"]["fallback"] == 0
        cn = data["cell_number"][:, 0, 0]
        nc, nfp = data["total_cells"], data["dagp_us_n_facepoints"]

        def mat(name, shape, data=data):
            return csr_matrix(
                (
                    data[f"dagp_us_{name}_weights"],
                    data[f"dagp_us_{name}_columns"],
                    data[f"dagp_us_{name}_rows"],
                ),
                shape=shape,
            )

        G, V, D = (
            mat("gradp", (nfp, nc)),
            mat("value", (nfp, nc)),
            mat("div", (nc, nfp)),
        )
        node_cells = cn[grid.node_x]
        k = 2 * np.pi
        R, Z = pol.dof[:, 0], pol.dof[:, 1]
        assert np.array_equal(data["R"][grid.node_x, 0, 0], R)
        f = np.zeros(nc)
        f[node_cells] = np.sin(k * R) * np.cos(k * Z)
        res = (D @ (V @ np.ones(nc) * (G @ f)))[node_cells]
        ex = -2 * k * k * np.sin(k * R) * np.cos(k * Z)
        assert np.all(res[~grid.interior] == 0.0)
        e = (res - ex)[grid.interior]
        errs.append(np.sqrt(np.mean(e**2)))
    assert np.log(errs[0] / errs[1]) / np.log(hs[0] / hs[1]) > 1.8, errs


def test_gmsh_planes_equal_counts():
    # Independently meshed planes of different shape get the same node and
    # wall-node counts, as the nodes-along-x layout requires
    pytest.importorskip("gmsh")
    shapes = [_curved_annulus(shift) for shift in (0.0, 0.03, -0.02)]
    planes = us.gmsh_planes(
        [o for o, _ in shapes], [i for _, i in shapes], mesh_size=1 / 16
    )
    assert len({p.n_nodes for p in planes}) == 1
    assert len({tuple(len(w) for w in p.walls) for p in planes}) == 1
    assert not np.allclose(planes[0].points, planes[1].points)
    grid = us.UnstructuredGrid(planes, np.arange(3.0), 3.0, dirichlet_walls=True)
    assert grid.n_interior % 4 == 0
    for p in planes:
        assert np.all(p.dual_volumes() > 0)
        p._check_orientation()


def test_gmsh_planes_node_count_without_dirichlet_walls():
    # Without Dirichlet walls all nodes evolve: gmsh_planes rounds the total
    # (not only the non-wall nodes) to multiple_of
    pytest.importorskip("gmsh")
    shapes = [_curved_annulus(shift) for shift in (0.0, 0.03)]
    outers, inners = [o for o, _ in shapes], [i for _, i in shapes]
    planes = us.gmsh_planes(outers, inners, mesh_size=1 / 16, dirichlet_walls=False)
    grid = us.UnstructuredGrid(planes, np.arange(2.0), 2.0, dirichlet_walls=False)
    assert grid.n_interior == planes[0].n_nodes
    assert grid.n_interior % 4 == 0


def test_node_count_not_multiple_only_warns():
    # A node count that is not a multiple of multiple_of only restricts NXPE:
    # the grid is built, with a warning listing the valid NXPE
    pol = strip(nR=5, nZ=5)  # 25 nodes
    with pytest.warns(UserWarning, match="NXPE in \\[1, 5, 25\\]"):
        grid = us.UnstructuredGrid(pol, np.array([0.0]), 1.0, multiple_of=4)
    assert grid.n_interior == 25


def _plane_laplacian_error(pol):
    from scipy.sparse import csr_matrix

    blk = us._plane_dagp(pol, order=4)
    n, nfp = pol.n_nodes, blk["n_fp"]
    rows = np.repeat(np.arange(nfp), blk["inds"].shape[1])
    G = csr_matrix((blk["w_grad"].ravel(), (rows, blk["inds"].ravel())), shape=(nfp, n))
    D = csr_matrix(
        (
            np.concatenate([1 / blk["vol"][blk["fi"]], -1 / blk["vol"][blk["fj"]]]),
            (np.concatenate([blk["fi"], blk["fj"]]), np.tile(np.arange(nfp), 2)),
        ),
        shape=(n, nfp),
    )
    R, Z = pol.dof[:, 0], pol.dof[:, 1]
    k = 2 * np.pi
    res = D @ (G @ (np.sin(k * R) * np.cos(k * Z)))
    ex = -2 * k * k * np.sin(k * R) * np.cos(k * Z)
    assert blk["max_re"] <= 1e-10
    return np.sqrt(np.mean((res - ex) ** 2))


def test_centroid_dofs_unrelaxed_order():
    # On an unrelaxed irregular mesh node unknowns give a first-order
    # truncation error; unknowns at the dual-cell centroids give second order
    def mesh(n, centroid):
        pol = us.UnstructuredPoloidalGrid.slab(
            n,
            n,
            1.0,
            1.0,
            jitter=0.2,
            jitter_dirs="rz",
            periodic_R=True,
            periodic_Z=True,
            seed=5 + n,
            relax=False,
        )
        return pol.use_centroid_dofs(centroid)

    ns = (24, 48)
    for centroid, lo, hi in ((True, 1.9, 3.0), (False, 0.5, 1.5)):
        errs = [_plane_laplacian_error(mesh(n, centroid)) for n in ns]
        order = np.log2(errs[0] / errs[1])
        assert lo < order < hi, (centroid, errs)
    pol = mesh(24, True)
    offsets = pol.dual_centroid_offsets()
    assert np.max(np.abs(offsets)) > 0.01 * pol.h[0]  # really unrelaxed
    assert np.allclose(pol.dof, pol.wrap(pol.points + offsets))


def test_lsq_eval_offsets_exact():
    # Polynomial evaluation weights reproduce polynomials of the fit order
    pol = us.UnstructuredPoloidalGrid.slab(
        12, 12, 1.0, 1.0, jitter=0.2, jitter_dirs="rz", periodic_R=False, seed=3
    )
    rng = np.random.default_rng(1)
    targets = np.column_stack((rng.uniform(0.3, 0.7, 20), rng.uniform(0.3, 0.7, 20)))
    offsets = rng.uniform(-0.05, 0.05, (20, 2, 2))
    inds, _, w_eval, _ = pol.lsq_weights(targets, order=4, eval_offsets=offsets)

    def poly(R, Z):
        return 1 + R - 2 * Z + R**2 * Z - 3 * Z**4 + R**3 * Z

    vals = poly(pol.R, pol.Z)[inds]
    got = np.einsum("mqk,mk->mq", w_eval, vals)
    pts = targets[:, None, :] + offsets
    assert np.allclose(got, poly(pts[..., 0], pts[..., 1]), atol=1e-10)


def test_upwind_reconstruction_order():
    # Upwinding with the polynomial reconstruction stays second order at
    # fine resolution (the linear reconstruction drops to ~1.5 at n = 128 on
    # this anisotropic strip) and matches the central value closely
    from scipy.sparse import csr_matrix

    k = 2 * np.pi
    errs = {"value": [], "upwind": []}
    for n in (64, 128):
        pol = us.UnstructuredPoloidalGrid.slab(
            8, n, 0.1, 1.0, R0=-0.05, jitter=0.2, jitter_dirs="z", seed=1234 + n
        )
        grid = us.UnstructuredGrid(pol, np.array([0.0]), 10.0)
        cn = np.full(grid.shape, -1)
        cn[1:-1, 0, 0] = np.arange(pol.n_nodes)
        d, _ = us.dagp_unstructured(grid, cn, pol.n_nodes, strict=False, order=4)
        nfp, nn = d["dagp_us_n_facepoints"], pol.n_nodes

        def mat(name, shape, d=d):
            return csr_matrix(
                (
                    d[f"dagp_us_{name}_weights"],
                    d[f"dagp_us_{name}_columns"],
                    d[f"dagp_us_{name}_rows"],
                ),
                shape=shape,
            )

        G, V, D = (
            mat("gradp", (nfp, nn)),
            mat("value", (nfp, nn)),
            mat("div", (nn, nfp)),
        )
        left, right = mat("left", (nfp, nn)), mat("right", (nfp, nn))
        z = k * pol.Z
        f = np.sin(np.pi - 2 * z) + np.sin(np.pi - z)
        a = np.cos(np.pi - z) + 2.0
        fz = -2 * np.cos(np.pi - 2 * z) - np.cos(np.pi - z)
        fzz = -4 * np.sin(np.pi - 2 * z) - np.sin(np.pi - z)
        ex = k * k * (a * fzz + np.sin(np.pi - z) * fz)
        g = G @ f
        for name, af in (
            ("value", V @ a),
            ("upwind", np.where(g > 0, right @ a, left @ a)),
        ):
            errs[name].append(np.sqrt(np.mean((D @ (af * g) - ex) ** 2)))
    order = np.log2(errs["upwind"][0] / errs["upwind"][1])
    assert order > 1.9, errs
    assert np.allclose(errs["upwind"], errs["value"], rtol=0.05), errs


def test_map_interpolation_amplification():
    # The parallel-map weights must not amplify much: large sum|w| spreads
    # grid-scale noise into stiff parallel conduction (more solver steps)
    from scipy.sparse import csr_matrix

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, data, _ = us.make_test_grid("bout_mms", 16)
    for name in ("forward", "backward"):
        nl = data[f"n_{name}_legs"]
        w = csr_matrix(
            (data[f"{name}_weights"], data[f"{name}_columns"], data[f"{name}_rows"]),
            shape=(nl, data["total_cells"]),
        )
        amp = np.asarray(abs(w).sum(axis=1)).ravel()
        interior = np.diff(w.indptr) > 1  # not boundary (single-entry) legs
        assert np.median(amp[interior]) < 1.7, np.median(amp[interior])
        assert amp.max() < 4.0, amp.max()


def test_value_fit_order_and_size():
    # The separate order-2 fit for the face values of a keeps second order
    # with a variable coefficient and needs far fewer nonzeros than the
    # order-4 gradient fit
    pytest.importorskip("gmsh")
    from scipy.sparse import csr_matrix

    outer, inner = _curved_annulus()
    k = 2 * np.pi
    errs = []
    for h in (1 / 16, 1 / 32):
        pol = us.UnstructuredPoloidalGrid.from_gmsh(outer, inner, mesh_size=h)
        grid = us.UnstructuredGrid(
            pol, np.array([0.0]), 1.0, dirichlet_walls=True, multiple_of=1
        )
        data, _ = us.build_grid_data(grid, zbfield.Slab(Bz=0.0, Bzprime=0.0))
        cn = data["cell_number"][:, 0, 0]
        nc, nfp = data["total_cells"], data["dagp_us_n_facepoints"]

        def mat(name, shape, data=data):
            return csr_matrix(
                (
                    data[f"dagp_us_{name}_weights"],
                    data[f"dagp_us_{name}_columns"],
                    data[f"dagp_us_{name}_rows"],
                ),
                shape=shape,
            )

        G, V, D = (
            mat("gradp", (nfp, nc)),
            mat("value", (nfp, nc)),
            mat("div", (nc, nfp)),
        )
        assert V.nnz < 0.4 * G.nnz
        node = cn[grid.node_x]
        R, Z = pol.dof[:, 0], pol.dof[:, 1]
        f, a = np.zeros(nc), np.zeros(nc)
        f[node] = np.sin(k * R) * np.cos(k * Z)
        a[node] = 1 + 0.5 * np.sin(k * R + 0.3) * np.cos(k * Z - 0.7)
        fR, fZ = k * np.cos(k * R) * np.cos(k * Z), -k * np.sin(k * R) * np.sin(k * Z)
        aR = 0.5 * k * np.cos(k * R + 0.3) * np.cos(k * Z - 0.7)
        aZ = -0.5 * k * np.sin(k * R + 0.3) * np.sin(k * Z - 0.7)
        ex = a[node] * (-2 * k * k * f[node]) + aR * fR + aZ * fZ
        res = (D @ ((V @ a) * (G @ f)))[node]
        errs.append(np.sqrt(np.mean((res - ex)[grid.interior] ** 2)))
    assert np.log2(errs[0] / errs[1]) > 1.8, errs


def test_operator_order_four_resolutions():
    # Hidden O(h) terms (the earlier stabilisation bracket and upwind
    # reconstruction) only showed beyond n = 64, so check
    # every pair of four resolutions up to n = 96 on an unrelaxed mesh with
    # centroid unknowns
    def mesh(n):
        return us.UnstructuredPoloidalGrid.slab(
            n,
            n,
            1.0,
            1.0,
            jitter=0.2,
            jitter_dirs="rz",
            periodic_R=True,
            periodic_Z=True,
            seed=5 + n,
            relax=False,
        ).use_centroid_dofs()

    errs = np.array([_plane_laplacian_error(mesh(n)) for n in (12, 24, 48, 96)])
    orders = np.log2(errs[:-1] / errs[1:])
    assert np.all(orders > 1.9), orders


def test_grid_generation_memory():
    # Grid generation competes for memory with simulations; keep the peak
    # memory of a test-grid build bounded.
    # VmHWM of the child process (ru_maxrss would include the pytest
    # parent's high-water mark, which Linux keeps across exec)
    import subprocess
    import sys

    if not Path("/proc/self/status").exists():
        pytest.skip("needs /proc/self/status")
    code = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "from zoidberg import unstructured as us\n"
        "us.make_test_grid('bout_mms', 16)\n"
        "line = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0]\n"
        "print(int(line.split()[1]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    peak_mb = int(out.stdout.split()[-1]) / 1024
    assert peak_mb < 1024, f"peak memory {peak_mb:.0f} MB"


def test_grid_generation_memory_scaling():
    # Peak memory of grid generation per cell, at two sizes, in fresh
    # processes (VmHWM minus the resident memory after the imports). The
    # perpendicular-operator assembly works plane by plane and the
    # least-squares fits in blocks, so the peak is about 34 kB per cell
    # (before: 77 kB per cell, 4.0 GB for the 3 mm Dommaschk grid) and
    # does not grow faster than the number of cells.
    import subprocess
    import sys

    if not Path("/proc/self/status").exists():
        pytest.skip("needs /proc/self/status")
    code = (
        "import sys, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "from zoidberg import unstructured as us\n"
        "def st(key):\n"
        "    return [int(l.split()[1]) for l in open('/proc/self/status')\n"
        "            if l.startswith(key)][0]\n"
        "base = st('VmRSS')\n"
        "_, data, _ = us.make_test_grid('bout_mms_cyl_sheared', int(sys.argv[1]))\n"
        "print(int(data['total_cells']), st('VmHWM') - base)\n"
    )
    per_cell = {}
    for n in (16, 24):
        out = subprocess.run(
            [sys.executable, "-c", code, str(n)],
            capture_output=True,
            text=True,
            check=True,
        )
        cells, kb = (int(v) for v in out.stdout.split()[-2:])
        per_cell[n] = kb / cells
    assert max(per_cell.values()) < 50.0, per_cell
    assert per_cell[24] < 1.2 * per_cell[16], per_cell


def test_lsq_blocks_and_repeated_targets_identical():
    # lsq_weights solves each target independently: processing the targets
    # in blocks and fitting repeated targets once give the same weights
    pol = us.UnstructuredPoloidalGrid.slab(
        12, 12, 1.0, 1.0, jitter=0.2, jitter_dirs="rz", seed=3
    )
    rng = np.random.default_rng(1)
    targets = pol.dof[rng.integers(0, pol.n_nodes, 300)]  # many repeats
    offsets = 0.05 * rng.standard_normal((300, 2, 2))
    kwargs = {"order": 3, "gradient": True, "eval_offsets": offsets}
    ref = pol.lsq_weights(targets, **kwargs)
    pol.lsq_block_size = 37
    blocked = pol.lsq_weights(targets, **kwargs)
    # One target at a time (no blocks, no repeats)
    single = [
        pol.lsq_weights(t[None], order=3, gradient=True, eval_offsets=o[None])
        for t, o in zip(targets, offsets)
    ]
    for i in range(5):
        assert np.array_equal(ref[i], blocked[i])
        assert np.array_equal(ref[i], np.concatenate([s[i] for s in single]))
    assert ref[5] == blocked[5]


def test_parallel_planes_identical():
    # dagp_unstructured with worker processes gives the same arrays
    planes = [
        us.UnstructuredPoloidalGrid.slab(
            8, 8, 1.0, 1.0, jitter=0.2, jitter_dirs="z", seed=20 + j
        )
        for j in range(4)
    ]
    field = zbfield.Slab(Bz=0.05, Bzprime=0.1)
    results = []
    for processes in (1, 3):
        _, data, _ = us.make_unstructured_grid(
            None, planes, field, ny=4, Ly=1.0, processes=processes
        )
        results.append(data)
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        assert np.array_equal(np.asarray(results[0][key]), np.asarray(results[1][key]))


# ----------------------------------------------------------------------
# K2: flux-reconstructed conservative parallel half-divergences


def _csr_from(data, name, shape):
    from scipy.sparse import csr_matrix

    return csr_matrix(
        (data[f"{name}_weights"], data[f"{name}_columns"], data[f"{name}_rows"]),
        shape=shape,
    )


def _support_half_divergences(data):
    """Div_minus and Div_plus of BOUT++'s PetscOperators::getParallel,
    rebuilt from the grid variables (unit leg weights)."""
    from scipy.sparse import csr_matrix, diags

    nc = int(data["total_cells"])
    cn = data["cell_number"][..., 0]
    dl_f = (data["dy"] * np.sqrt(data["g_22"]))[..., 0]
    dV_f = (data["dy"] * data["dagp_us_volume"])[..., 0]
    dl, dV = np.zeros(nc), np.zeros(nc)
    for numbers in (cn, data["forward_cell_number"][..., 0]):
        m = numbers >= 0
        dl[numbers[m]], dV[numbers[m]] = dl_f[m], dV_f[m]
    m = data["backward_cell_number"][..., 0] >= 0
    dl[data["backward_cell_number"][..., 0][m]] = dl_f[m]
    dV[data["backward_cell_number"][..., 0][m]] = dV_f[m]
    out = {}
    for name, sign in (("forward", 1), ("backward", -1)):
        nl = int(data[f"n_{name}_legs"])
        F = _csr_from(data, name, (nl, nc))
        rows, cols = [], []
        for kind in ("interior", "boundary"):
            legs = data[f"{name}_leg_{kind}_number"][..., 0]
            m = (legs >= 0) & (cn >= 0)
            rows.append(legs[m])
            cols.append(cn[m])
        inj = csr_matrix(
            (np.ones(nl), (np.concatenate(rows), np.concatenate(cols))), shape=(nl, nc)
        )
        interp = 0.5 * (F + inj)
        grad = diags(sign / (interp @ dl)) @ (F - inj)
        out[name] = (diags(-1 / dV) @ grad.T @ diags(interp @ dV)).tocsr()
    return out, dV


def test_k2_overlap_partition_and_moments():
    # The pieces of the dual cells with the pulled-back destination cells
    # (another mesh, sheared) tile both planes exactly, and the polygon
    # moments integrate quadratics exactly
    def plane(n, seed):
        return us.UnstructuredPoloidalGrid.slab(
            n,
            n + 2,
            1.0,
            1.0,
            jitter=0.2,
            jitter_dirs="rz",
            seed=seed,
            periodic_R=False,
            periodic_Z=True,
            r_on_walls=True,
        )

    src, dst = plane(10, 3), plane(11, 4)
    td, od = us._dual_triangles(dst)
    td = td.copy()
    td[..., 1] -= 0.13 + 0.3 * td[..., 0]  # affine, area-preserving shear
    o, k, mom, contains, pulled = us._k2_pieces(src, td, od)
    A_src = us._k2_cell_moments(src)[:, 0]
    assert np.allclose(np.bincount(o, mom[:, 0]), A_src, rtol=1e-12, atol=0)
    assert np.allclose(np.bincount(k, mom[:, 0]), dst.dual_volumes(), rtol=1e-12)
    assert np.allclose(pulled, dst.dual_volumes(), rtol=1e-12)
    # Every source dof lies in (at least) one of its pieces
    assert np.all(np.bincount(o, contains) >= 1)
    # Moments against a 3-point (degree-2 exact) rule on random triangles
    rng = np.random.default_rng(0)
    T = rng.standard_normal((50, 3, 2))
    T[us._tri_area_signed(T) < 0] = T[us._tri_area_signed(T) < 0][:, ::-1]
    P = np.concatenate([T, np.repeat(T[:, :1], 6, axis=1)], axis=1)
    m = us._poly_moments(P)
    A = us._tri_area_signed(T)
    q = 0.5 * (T + np.roll(T, -1, axis=1))  # edge midpoints
    x, y = q[..., 0], q[..., 1]
    for col, f in enumerate((x**0, x, y, x * x, x * y, y * y)):
        assert np.allclose(m[:, col], A * f.mean(axis=1), rtol=1e-12, atol=1e-14)


def test_k2_reconstruction_exact():
    # Quadratic (linear) flux reconstructions reproduce quadratic (linear)
    # densities; cells without an interior leg are excluded from every stencil
    pol = strip(nR=8, nZ=24)
    cand = np.ones(pol.n_nodes, dtype=bool)
    cand[pol.R < pol.R.min() + 1e-9] = False
    for order in (1, 2):
        nb, P, used = us._k2_reconstruction(pol, cand, order)
        assert np.all(used[cand] == order) and np.all(used[~cand] == -1)
        assert np.all(cand[nb[cand]])
        assert us._k2_check_reconstruction(pol, nb, P, used) < 1e-12


def test_k2_order0_equals_support_operator():
    # Convention check: with a straight field (injection maps, identity
    # pull-back) the constant reconstruction gives exactly BOUT++'s
    # Div_minus / Div_plus, including the boundary legs of a y-bounded grid
    pol = strip(nR=8, nZ=8)
    field = zbfield.Slab(By=1.0, Bz=0.0, Bzprime=0.0)
    _, data, stats = us.make_unstructured_grid(
        None, pol, field, ny=4, Ly=10.0, yperiodic=False, k2_order=0
    )
    assert stats["k2"]["order"] == 0
    so, _ = _support_half_divergences(data)
    nc = int(data["total_cells"])
    for name in ("forward", "backward"):
        nl = int(data[f"n_{name}_legs"])
        assert np.any(data[f"{name}_leg_boundary_number"] >= 0)
        D = _csr_from(data, f"k2_{name}", (nc, nl))
        diff = abs(D - so[name]).max()
        assert diff <= 1e-13 * abs(so[name]).max(), name


def test_k2_reference_matrices():
    # Order-2 matrices against reference matrices from an independent
    # vectorised implementation of the same scheme (exact analytic maps),
    # fingerprinted as D v for a fixed random vector
    # (zoidberg/test_data/k2_reference.npz); zoidberg traces the
    # field lines, which agree with the analytic maps to round-off here
    ref = np.load(Path(__file__).parent / "test_data" / "k2_reference.npz")
    for family in ("bout_mms", "bout_mms_cyl_sheared"):
        _, data, stats = us.make_test_grid(family, 16, k2_refine=0)
        assert stats["k2"]["tube_volume"] == "dV"
        nc = int(data["total_cells"])
        cn = data["cell_number"][..., 0]
        v = np.random.default_rng(0).standard_normal(nc)
        for name in ("forward", "backward"):
            nl = int(data[f"n_{name}_legs"])
            legs = data[f"{name}_leg_interior_number"][..., 0]
            assert np.all((legs >= 0) == (cn >= 0))
            v_leg = np.zeros(nl)
            v_leg[legs[legs >= 0]] = v[cn[legs >= 0]]
            Dv = _csr_from(data, f"k2_{name}", (nc, nl)) @ v_leg
            r = ref[f"{family}_{name}"]
            assert np.max(np.abs(Dv - r)) <= 1e-12 * np.max(np.abs(r)), family


def test_k2_gmsh_walls():
    # Independently meshed gmsh planes with curved Dirichlet walls and a
    # sheared field whose lines cross the walls (boundary legs, tubes that
    # partly leave the mesh, curved images): every column of dV D sums to
    # zero, boundary-leg columns are those of Div_minus / Div_plus, most
    # pair reconstructions are quadratic, and the matrices have one row per
    # cell
    pytest.importorskip("gmsh")
    shapes = [_curved_annulus(shift) for shift in (0.0, 0.03, -0.02)]
    planes = us.gmsh_planes(
        [o for o, _ in shapes], [i for _, i in shapes], mesh_size=1 / 12
    )
    field = us.ShearedCylinderField(0.2, 0.3, 2.0)
    _, data, stats = us.make_unstructured_grid(
        None, planes, field, ny=3, Ly=0.6, dirichlet_walls=True
    )
    st = stats["k2"]
    assert st["refine"] == 1 and st["unique_pairs"] == 6
    assert st["n_deficit_src"] > 0 and st["n_deficit_dst"] > 0
    assert st["order_counts"].get(2, 0) > 0.9 * sum(st["order_counts"].values())
    so, dV = _support_half_divergences(data)
    nc = int(data["total_cells"])
    for name in ("forward", "backward"):
        nl = int(data[f"n_{name}_legs"])
        D = _csr_from(data, f"k2_{name}", (nc, nl))
        colsum = dV @ D
        assert np.max(np.abs(colsum)) <= 1e-12 * np.max(abs(D).T @ dV)
        bnd = data[f"{name}_leg_boundary_number"][..., 0]
        bnd = bnd[bnd >= 0]
        assert len(bnd) > 10
        assert abs(D[:, bnd] - so[name][:, bnd]).max() <= 1e-14 * abs(D).max()
        # Rows only for cells (evolving, Dirichlet wall, boundary virtual)
        assert D.shape[0] == nc


def test_k2_variables_in_grid_file(tmp_path):
    pol = strip(nR=8, nZ=8)
    field = zbfield.Slab(Bz=0.05, Bzprime=0.1)
    us.make_unstructured_grid(str(tmp_path / "g.nc"), pol, field, ny=4, Ly=10.0)
    us.make_unstructured_grid(
        str(tmp_path / "n.nc"), pol, field, ny=4, Ly=10.0, k2=False
    )
    from boututils.datafile import DataFile

    with DataFile(str(tmp_path / "g.nc")) as f:
        assert int(f["k2_order"]) == 2
        assert int(f["k2_version"]) == us.K2_VERSION
        nc, nl = int(f["total_cells"]), int(f["n_forward_legs"])
        for name in ("forward", "backward"):
            rows = f[f"k2_{name}_rows"]
            cols = f[f"k2_{name}_columns"]
            assert len(rows) == nc + 1 and rows[-1] == len(cols)
            assert len(f[f"k2_{name}_weights"]) == len(cols)
            assert cols.min() >= 0 and cols.max() < nl
            for r in range(nc):
                c = cols[rows[r] : rows[r + 1]]
                assert len(np.unique(c)) == len(c)
    with DataFile(str(tmp_path / "n.nc")) as f:
        assert not any(k.startswith("k2_") for k in f.list())


def test_add_k2_to_file(tmp_path):
    # Existing grid files (without K2, or with another order) get the same
    # k2 variables as a fresh build; a grid that does not match is refused
    from boututils.datafile import DataFile

    path = str(tmp_path / "g.nc")
    _, data0, _ = us.make_test_grid("bout_mms", 8, filename=path, k2=False)
    grid, field = us.family_grid("bout_mms", 8)
    for order in (1, 2):
        us.add_k2_to_file(path, grid, field, order=order)
        _, ref, _ = us.make_test_grid("bout_mms", 8, k2_order=order)
        with DataFile(path) as f:
            assert int(f["k2_order"]) == order
            for d in ("forward", "backward"):
                for part in ("rows", "columns", "weights"):
                    key = f"k2_{d}_{part}"
                    assert np.array_equal(f[key], ref[key]), key
            assert np.array_equal(f["forward_weights"], data0["forward_weights"])
    other, field = us.family_grid("bout_mms", 8, seed=99)
    with pytest.raises(ValueError):
        us.add_k2_to_file(path, other, field)


def test_k2_tiny_grids():
    # Regression (BOUT-dev test-petsc-operators-unstructured, 5 x 4 nodes,
    # periodic in Z): the nearest points include further periodic images of
    # one node; the stencils must use distinct nodes, and planes with too few
    # nodes for the quadratic fit fall back to smaller stencils or the
    # linear reconstruction
    field = zbfield.Slab(By=1.0, Bz=0.3, Bzprime=0.0)
    for (nR, nZ), order, kw in (
        ((5, 4), 2, {}),
        ((3, 4), 2, {"map_order": 1, "dagp_order": 1, "strict": False}),
        ((2, 4), 1, {"map_order": 1, "dagp_order": 1, "strict": False}),
    ):
        pol = us.UnstructuredPoloidalGrid.slab(
            nR, nZ, 1.0, 1.0, R0=1.5, jitter=0.2, jitter_dirs="z", seed=7
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, data, stats = us.make_unstructured_grid(
                None, pol, field, ny=16, Ly=10.0, walls_y=True, **kw
            )
        st = stats["k2"]
        assert list(st["order_counts"]) == [order], (nR, nZ, st["order_counts"])
        assert st["reconstruction_error"] < 1e-12
        assert "k2_forward_weights" in data


def test_same_plane_twice_gives_the_same_grid():
    """Building the perpendicular operator scales the plane isotropically
    for a while; the plane's search trees must not keep that scaling, or a
    second grid built from the same plane classifies interior field-line end
    points as wall hits."""
    pol = us.UnstructuredPoloidalGrid.slab(
        8, 16, 0.1, 1.0, R0=-0.05, jitter=0.2, seed=1
    )
    field = zbfield.Slab(Bz=0.05, Bzprime=0.1)
    results = []
    for _ in range(2):
        _, data, stats = us.make_unstructured_grid(
            None, pol, field, ny=4, Ly=10.0, k2=False
        )
        results.append(
            (stats["forward_hits"], stats["backward_hits"], data["total_cells"])
        )
    assert results[0] == results[1]
    assert pol.in_domain(pol.points).all()


def test_wall_options_keep_wall_nodes_out_of_the_maps():
    # Field lines that cross the walls (constant B_R): map_walls=False keeps
    # the Dirichlet wall nodes out of the parallel-map stencils, and
    # near_wall_hits=True makes end points whose nearest node is a wall node
    # wall hits
    class Tilted(zbfield.Slab):
        def __init__(self):
            super().__init__(By=1.0, Bz=0.05, Bzprime=0.0)

        def Bxfunc(self, x, z, phi):
            return np.full(np.shape(x), 0.4)

    pol = us.UnstructuredPoloidalGrid.slab(
        9,
        8,
        1.0,
        1.0,
        R0=1.5,
        jitter=0.2,
        jitter_dirs="z",
        periodic_R=False,
        periodic_Z=True,
        seed=3,
        r_on_walls=True,
    )
    pol.walls = (
        np.flatnonzero(np.isclose(pol.R, 1.5)),
        np.flatnonzero(np.isclose(pol.R, 2.5)),
    )
    ny = 6
    grid = us.UnstructuredGrid(
        pol,
        np.linspace(0, 2 * np.pi, ny, endpoint=False),
        2 * np.pi,
        dirichlet_walls=True,
    )
    field = Tilted()
    maps = us.trace_maps(grid, field)
    near = us.trace_maps(grid, field, near_wall_hits=True)
    wall = np.concatenate(pol.walls)
    plasma = np.setdiff1d(np.arange(pol.n_nodes), wall)
    for name in ("forward", "backward"):
        hit, hit_near = maps[f"{name}_hit"] != 0, near[f"{name}_hit"] != 0
        assert np.all(hit_near[hit]) and hit_near.sum() > hit.sum()
        # No interior end point is closer to a wall node than to a plasma node
        pos = np.column_stack(
            (near[f"{name}_R"][~hit_near], near[f"{name}_Z"][~hit_near])
        )
        d_wall = np.min(
            np.linalg.norm(pos[:, None] - pol.points[wall][None], axis=2), axis=1
        )
        d_plasma = np.min(
            np.linalg.norm(pos[:, None] - pol.points[plasma][None], axis=2), axis=1
        )
        assert np.all(d_plasma <= d_wall)
    # Wall cells are numbered after the evolving cells
    n_evolving = int(np.sum(grid.interior)) * ny
    n_wall = (pol.n_nodes - int(np.sum(grid.interior))) * ny
    for map_walls in (True, False):
        som, _ = us.som_weights(grid, near, map_walls=map_walls)
        cols = np.concatenate([som["forward_columns"], som["backward_columns"]])
        in_wall = (cols >= n_evolving) & (cols < n_evolving + n_wall)
        assert in_wall.any() == map_walls
