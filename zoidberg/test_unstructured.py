import warnings

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
