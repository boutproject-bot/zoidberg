import numpy as np
import pytest
from netCDF4 import Dataset

from . import shiftedmap

# A small single-null grid with 2 x guard cells on each side: interior
# x = 2..5, of which x = 2, 3 are inside the separatrix; inner leg
# y = 0..3, core y = 4..7, outer leg y = 8..11
NX, NY, MXG = 8, 12, 2
IXSEPS1, JYSEPS1_1, JYSEPS2_2 = 4, 3, 7


def shift_angle(x):
    return 2 * np.pi * (0.3 + 0.1 * x)


def write_single_null(path, jyseps2_1=6, jyseps1_2=6):
    x, y = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    variables = {
        "zShift": 0.4 * (x + 1) * y + 0.7 * np.sin(y * 2 * np.pi / NY + x),
        "dx": np.ones(x.shape),
        "dy": 0.5 + 0.02 * y,
        "g_22": 1 + 0.1 * x + 0.01 * y,
        "g22": 1 / (1 + 0.1 * x + 0.01 * y),
        "J": 1 + 0.05 * x + 0.02 * y**2,
        "Bxy": 2 + 0.1 * x - 0.05 * y,
        "Ni0": 1 + x + 0.1 * y,
    }
    with Dataset(path, "w") as d:
        d.createDimension("x", NX)
        d.createDimension("y", NY)
        d.setncattr("parallel_transform", "shiftedmetric")
        integers = {"nx": NX, "ny": NY, "ixseps1": IXSEPS1, "ixseps2": NX}
        integers |= {"jyseps1_1": JYSEPS1_1, "jyseps2_1": jyseps2_1}
        integers |= {"jyseps1_2": jyseps1_2, "jyseps2_2": JYSEPS2_2, "ny_inner": 6}
        for name, value in integers.items():
            d.createVariable(name, "i4")[...] = value
        d.createVariable("Bnorm", "f8")[...] = 2.5
        d.createVariable("ShiftAngle", "f8", ("x",))[...] = shift_angle(np.arange(NX))
        for name, value in variables.items():
            d.createVariable(name, "f8", ("x", "y"))[...] = value
    return variables


@pytest.fixture
def recast(tmp_path):
    """The recast grid (shuffled) as a dictionary, the structured
    variables, and the structured indices of the stored cells"""
    variables = write_single_null(tmp_path / "structured.nc")
    xs, ys = shiftedmap.recast(
        tmp_path / "structured.nc", tmp_path / "recast.nc", mxg=MXG, seed=4
    )
    with Dataset(tmp_path / "recast.nc") as d:
        grid = {name: var[...] for name, var in d.variables.items()}
        grid["parallel_transform"] = d.getncattr("parallel_transform")
        # BOUT++ reads grid fields as doubles
        for name, var in d.variables.items():
            if var.dimensions[:1] == ("x",):
                assert var.dtype == np.float64, name
    return grid, variables, xs, ys


def walk(grid, start, name, nsteps):
    """The cells reached from cell number ``start`` along ``name``
    (forward or backward), and the angles of the steps; stops at a target"""
    cells, angles = [start], []
    for _ in range(nsteps):
        nxt = int(grid[f"pmap_{name}_x"][cells[-1], 0])
        if nxt == grid["nx"]:
            break
        angles.append(grid[f"pmap_{name}_dzeta"][cells[-1], 0])
        cells.append(nxt)
    return cells, angles


def test_field_lines(recast):
    grid, variables, xs, ys = recast
    zshift = variables["zShift"]
    ncells = (NX - 2 * MXG) * NY
    assert grid["nx"] == ncells and grid["ny"] == 1
    cell = {(x, y): u for u, (x, y) in enumerate(zip(xs, ys))}
    for x in range(MXG, NX - MXG):
        if x >= IXSEPS1:
            # Scrape-off layer: from the inner to the outer target
            lines = {0: list(range(NY))}
        else:
            # Private flux region below the X-point, and the closed core
            legs = list(range(JYSEPS1_1 + 1)) + list(range(JYSEPS2_2 + 1, NY))
            lines = {0: legs, JYSEPS1_1 + 1: list(range(JYSEPS1_1 + 1, JYSEPS2_2 + 1))}
        for start, line in lines.items():
            closed = start == JYSEPS1_1 + 1 and x < IXSEPS1
            cells, angles = walk(grid, cell[x, start], "forward", len(line))
            reached = [int(ys[u]) for u in cells]
            assert reached == (line + [line[0]] if closed else line), (x, reached)
            # The angles add up along the line; once round the core they
            # give the twist
            expected = shift_angle(x) if closed else zshift[x, line[-1]] - zshift[x, 0]
            assert np.isclose(sum(angles), expected, rtol=0, atol=1e-12), x
            # Backward retraces the line
            back, _ = walk(grid, cells[-1], "backward", len(line))
            assert back == cells[::-1], x
    # Targets are outer boundaries, after the last y and before the first
    for name, end in (("forward", NY - 1), ("backward", 0)):
        target = grid[f"pmap_{name}_x"][:, 0] == ncells
        assert np.array_equal(target, ys == end)


def test_values(recast):
    grid, variables, xs, ys = recast
    assert grid["parallel_transform"] == "shiftedmap"
    assert grid["Bnorm"] == 2.5
    assert np.array_equal(grid["ShiftAngle"], np.zeros(len(xs)))
    assert np.array_equal(grid["d2x"][:, 0], np.zeros(len(xs)))
    assert np.array_equal(grid["Ni0"][:, 0], variables["Ni0"][xs, ys])
    g_22 = variables["g_22"]
    jxz = variables["J"] / np.sqrt(g_22)
    for name, face in (("forward", "yhigh"), ("backward", "ylow")):
        nxt = grid[f"pmap_{name}_x"][:, 0].astype(int)
        at_target = nxt == len(xs)
        nxt = np.where(at_target, np.arange(len(xs)), nxt)
        for component in ("dy", "g_22", "g22", "Bxy"):
            # The metric at the neighbour, the cell's own at a target
            there = variables[component][xs[nxt], ys[nxt]]
            assert np.array_equal(grid[f"{name}_{component}"][:, 0], there), component
        # The y faces of the field-aligned mesh: g_22 from the mean of
        # sqrt(g_22), and the area (here per dx dz) that BOUT++ takes from By
        # on FCI-like meshes equal to the mean of J / sqrt(g_22)
        root = 0.5 * (np.sqrt(g_22[xs, ys]) + np.sqrt(g_22[xs[nxt], ys[nxt]]))
        assert np.allclose(grid[f"g_22_cell_{face}"][:, 0], root**2, rtol=1e-14, atol=0)
        area = grid["By"][:, 0] / grid[f"By_cell_{face}"][:, 0] * jxz[xs, ys]
        mean = 0.5 * (jxz[xs, ys] + jxz[xs[nxt], ys[nxt]])
        assert np.allclose(area, mean, rtol=1e-14, atol=0), face


def test_order_does_not_matter(tmp_path):
    write_single_null(tmp_path / "structured.nc")
    results = []
    for seed in (None, 11):
        xs, ys = shiftedmap.recast(
            tmp_path / "structured.nc", tmp_path / f"recast_{seed}.nc", seed=seed
        )
        with Dataset(tmp_path / f"recast_{seed}.nc") as d:
            nxt = d["pmap_forward_x"][:, 0].astype(int)
            dzeta = d["pmap_forward_dzeta"][:, 0]
        # The structured cell that each structured cell maps to, and the angle
        found = {}
        for u, (x, y) in enumerate(zip(xs, ys)):
            there = None if nxt[u] == len(xs) else (xs[nxt[u]], ys[nxt[u]])
            found[x, y] = (there, dzeta[u])
        results.append(found)
    assert results[0] == results[1]


def test_double_null_refused(tmp_path):
    write_single_null(tmp_path / "structured.nc", jyseps2_1=5, jyseps1_2=6)
    with pytest.raises(NotImplementedError, match="single null"):
        shiftedmap.recast(tmp_path / "structured.nc", tmp_path / "recast.nc")
