"""Regression tests for the boundary-orthogonality corrections in
`grid_elliptic`:

* the algebraic near-boundary correction (`boundary_orthogonal=True`)
* the GRAPE/Sorenson-style control-function iteration (`control_functions=True`)
* the `min_sin_alpha` option (which auto-enables the correction)
* the `StructuredPoloidalGrid.orthogonality` / `minSinAlpha` helpers

Expected values below were established against the baseline elliptic grids
(no correction requested): the default path must be unchanged, and each
correction must satisfy its documented contract.

"""

import numpy as np
import pytest

from zoidberg import rzline
from zoidberg.poloidal_grid import StructuredPoloidalGrid, grid_elliptic, sin_alpha


def _signed_jacobian(R, Z):
    """One-sided radial, central poloidal signed Jacobian ``dRdx*dZdz -
    dRdz*dZdx`` (the sign pattern of the baseline grid)."""
    R = np.asarray(R)
    Z = np.asarray(Z)
    dRdx = np.empty_like(R)
    dZdx = np.empty_like(Z)
    dRdx[1:-1, :] = R[2:, :] - R[:-2, :]
    dZdx[1:-1, :] = Z[2:, :] - Z[:-2, :]
    dRdx[0, :] = R[1, :] - R[0, :]
    dZdx[0, :] = Z[1, :] - Z[0, :]
    dRdx[-1, :] = R[-1, :] - R[-2, :]
    dZdx[-1, :] = Z[-1, :] - Z[-2, :]
    dRdz = np.roll(R, -1, axis=1) - np.roll(R, 1, axis=1)
    dZdz = np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)
    return dRdx * dZdz - dRdz * dZdx


def _grids():
    """A small dictionary of (inner, outer, kwargs) stress cases."""
    inner = rzline.shaped_line(R0=3.0, a=0.5, elong=1.3, triang=0.2, indent=0.6, n=40)
    outer = rzline.shaped_line(R0=2.8, a=1.5, elong=1.3, triang=0.2, indent=0.4, n=40)
    tl = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    misaligned_inner = rzline.RZline(
        inner.Rvalue(tl + 0.5), inner.Zvalue(tl + 0.5)
    )
    cusp_inner = rzline.shaped_line(
        R0=3.0, a=0.5, elong=1.3, triang=0.2, indent=1.0, n=40
    )
    return {
        "aligned": (inner, outer, {}),
        "misaligned": (misaligned_inner, outer, {"inner_ort": False}),
        "cusp": (cusp_inner, outer, {}),
    }


def _run(inner, outer, nx=40, nz=40, **kw):
    return grid_elliptic(
        inner, outer, nx, nz, show=False, return_coords=True, **kw
    )


@pytest.fixture(scope="module")
def baselines():
    """Baseline grids with no correction requested, and their min sin(alpha)."""
    result = {}
    for name, (inn, out, kw) in _grids().items():
        R, Z = _run(inn, out, **kw)
        result[name] = {"R": R, "Z": Z, "min": sin_alpha(R, Z).min()}
    return result


def test_default_path_matches_baseline(baselines):
    """The default grid_elliptic path (no correction options set) must be
    unchanged from before the correction options were added."""
    expected = {"aligned": 0.4787, "misaligned": 0.3399, "cusp": 0.3008}
    for name, exp in expected.items():
        assert baselines[name]["min"] == pytest.approx(exp, abs=5e-4)
    # The cusp case has no negative Jacobians on the default path
    for name in expected:
        assert np.all(_signed_jacobian(baselines[name]["R"], baselines[name]["Z"]) > 0)


@pytest.mark.parametrize("name", ["aligned", "misaligned", "cusp"])
def test_algebraic_sharpens_boundary_rings(baselines, name):
    """boundary_orthogonal=True makes the grid lines leave both boundaries
    orthogonally (ring sin(alpha) close to 1) without NaN or new folds."""
    inn, out, kw = _grids()[name]
    R, Z = _run(inn, out, boundary_orthogonal=True, **kw)
    s = sin_alpha(R, Z)
    assert np.all(np.isfinite(s))
    # Some ring next to each boundary must have been substantially improved
    assert max(s[0].min(), s[-1].min()) > 0.9
    # And the far-boundary handlers leave the opposite ring orthogonal too
    if name != "cusp":
        assert min(s[0].min(), s[-1].min()) > 0.8
    # No new folds vs. the baseline sign pattern
    base = baselines[name]
    assert np.all(np.sign(_signed_jacobian(R, Z)) == np.sign(_signed_jacobian(base["R"], base["Z"])))
    # Correction must not be a strict regression of the global floor for the
    # healthy cases.
    if name != "cusp":
        assert s.min() >= base["min"] - 0.05


@pytest.mark.parametrize("name", ["aligned", "misaligned", "cusp"])
def test_control_functions_raise_floor(baselines, name):
    """The GRAPE-style iteration raises the global floor of sin(alpha)."""
    inn, out, kw = _grids()[name]
    R, Z = _run(inn, out, control_functions=True, **kw)
    s = sin_alpha(R, Z)
    assert np.all(np.isfinite(s))
    assert s.min() > baselines[name]["min"]
    assert np.all(np.sign(_signed_jacobian(R, Z)) == np.sign(_signed_jacobian(baselines[name]["R"], baselines[name]["Z"])))
    per_case = {"aligned": 0.52, "misaligned": 0.40, "cusp": 0.33}
    assert s.min() >= per_case[name]


@pytest.mark.parametrize("name", ["aligned", "misaligned", "cusp"])
def test_min_sin_alpha_auto_enables(baselines, name):
    """min_sin_alpha given without method flags auto-enables the correction,
    here the GRAPE-style iteration, and its floor is respected."""
    inn, out, kw = _grids()[name]
    R, Z = _run(inn, out, min_sin_alpha=0.2, **kw)
    assert sin_alpha(R, Z).min() >= 0.2


def test_min_sin_alpha_reachable_thresholds():
    """Where geometrically possible the requested threshold is met."""
    inn, out, kw = _grids()["aligned"]
    R, Z = _run(inn, out, min_sin_alpha=0.5, control_functions=True, **kw)
    assert sin_alpha(R, Z).min() >= 0.5

    inn, out, kw = _grids()["misaligned"]
    R, Z = _run(inn, out, min_sin_alpha=0.4, control_functions=True, **kw)
    assert sin_alpha(R, Z).min() >= 0.4


def test_min_sin_alpha_warns_when_unreachable():
    """A cusp (X-point) inner boundary aligned to its outer one is not
    geometrically fixable: the correction warns and returns a finite grid."""
    cusp_inner = rzline.shaped_line(
        R0=3.0, a=0.5, elong=1.3, triang=0.2, indent=1.0, n=40
    )
    outer = rzline.shaped_line(
        R0=2.8, a=1.5, elong=1.3, triang=0.2, indent=0.4, n=40
    )
    with pytest.warns(UserWarning):
        R, Z = _run(
            cusp_inner,
            outer,
            nx=20,
            nz=20,
            min_sin_alpha=0.5,
            control_functions=True,
            inner_ort=False,
        )
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(Z))


def test_orthogonal_layers_option(baselines):
    """orthogonal_layers controls how many rings the algebraic method uses."""
    inn, out, kw = _grids()["aligned"]
    R1, Z1 = _run(inn, out, boundary_orthogonal=True, orthogonal_layers=1, **kw)
    R2, Z2 = _run(inn, out, boundary_orthogonal=True, orthogonal_layers=2, **kw)
    # More layers affects more rings (the grids really differ)
    assert not np.array_equal(R1, R2)
    # Both keep the boundary grid points fixed
    R0, Z0 = baselines["aligned"]["R"], baselines["aligned"]["Z"]
    assert np.allclose(R1[0], R0[0]) and np.allclose(R1[-1], R0[-1])
    assert np.allclose(Z1[0], Z0[0]) and np.allclose(Z1[-1], Z0[-1])
    assert np.allclose(R2[0], R0[0]) and np.allclose(R2[-1], R0[-1])
    assert np.allclose(Z2[0], Z0[0]) and np.allclose(Z2[-1], Z0[-1])


def test_structured_poloidal_grid_methods():
    """The sin(alpha) helpers on the grid object are consistent with the
    module-level `sin_alpha` function."""
    inner, outer, _ = _grids()["aligned"]
    g = grid_elliptic(inner, outer, 40, 40, show=False)
    assert isinstance(g, StructuredPoloidalGrid)
    ort = g.orthogonality()
    assert ort.shape == g.R.shape
    assert np.allclose(ort, sin_alpha(g.R, g.Z))
    assert g.minSinAlpha() == pytest.approx(ort.min())


def test_finite_for_all_options():
    """All option combinations produce finite coordinates and no NaNs."""
    inn, out, kw = _grids()["aligned"]
    for opts in (
        {"boundary_orthogonal": True},
        {"control_functions": True},
        {"min_sin_alpha": 0.2},
    ):
        R, Z = _run(inn, out, **opts, **kw)
        assert np.all(np.isfinite(R)) and np.all(np.isfinite(Z))
