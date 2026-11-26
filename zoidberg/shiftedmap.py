"""Recast a structured BOUT++ tokamak grid for the shiftedmap parallel transform

A structured grid (for example from hypnotoad) stores its cells (x, y) in
field-line order along y, and BOUT++ connects its regions (core, private
flux region, scrape-off layer) by the topology indices ``ixseps1``,
``jyseps1_1``, ``jyseps2_1``, ``jyseps1_2``, ``jyseps2_2``. :func:`recast`
writes the same cells as a list along x (y of size 1), each with its
neighbours along the magnetic field and the toroidal angle the field line
advances to them, for ``[mesh:paralleltransform] type = shiftedmap`` (see
the BOUT++ manual, "Shifted metric with a map"). Fields keep their
structured z, so nothing in the file grows with nz.

Grid variables written, for ``<dir>`` in ``forward`` and ``backward``:

``pmap_<dir>_x``, ``pmap_<dir>_y``
    Global x index of the neighbour in the new grid (which has no x guard
    cells: run with ``MXG = 0``) and its y index (0). At a target
    ``pmap_<dir>_x = nx`` (an outer boundary, which the default y boundary
    iterator visits).
``pmap_<dir>_dzeta``
    Toroidal angle by which the field line advances to the neighbour:
    ``zShift(neighbour) - zShift(cell)``, plus ``ShiftAngle`` where the
    field line crosses the core's branch cut.
``<dir>_<component>``
    The metric at the neighbour (the cell's own at a target), for the
    components BOUT++ reads in the parallel slices with 3D metrics.
``g_22_cell_ylow``, ``g_22_cell_yhigh``, ``By``, ``By_cell_ylow``, ``By_cell_yhigh``
    The y faces, which BOUT++ reads from the grid on FCI-like meshes: g_22
    there, and By at the cell (``By``, or ``Bxy``) and at the faces, chosen
    so that the face areas are those of the field-aligned mesh (the mean of
    J / sqrt(g_22) dx dz of the two cells). Written if the grid has ``J``
    and ``g_22``.

Every other 2D (x, y) variable is written for each cell; ``d2x`` is zero
(index x has no meaning in the new grid) and so is ``d2y`` unless the
grid has it; ``ShiftAngle`` is zero (the twist is part of the angles).
The x guard cells of the structured grid are left out. Only single null
(and limiter or closed) grids without y boundary cells are supported.
"""

import numpy as np

#: Metric components that BOUT++ reads in the parallel slices
SLICED = ("g_11", "g_22", "g_33", "g_13", "g11", "g22", "g33", "g13", "dy", "Bxy")

#: Integers that describe the structured topology (not written)
TOPOLOGY = ("ixseps1", "ixseps2", "jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2")
TOPOLOGY += ("ny_inner", "y_boundary_guards")


def single_null_neighbours(topology, zshift, shift_angle, step):
    """For each cell (x, y) of a single-null grid: the y index of its
    neighbour along y + step in BOUT++'s topology (-1 at a target), and the
    toroidal angle the field line advances to it.

    Inside the separatrix (x < ixseps1) the inner leg (y <= jyseps1_1) and
    the outer leg (y > jyseps2_2) join below the X-point, and the core in
    between is closed, crossing the branch cut between its last and first
    cell; outside, the field lines run from y = 0 to the last y.
    """
    nx, ny = zshift.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = x < topology["ixseps1"]
    first, last = topology["jyseps1_1"] + 1, topology["jyseps2_2"]
    core = inside & (y >= first) & (y <= last)
    neighbour = y + step
    if step == 1:
        cut = core & (y == last)
        neighbour[cut] = first
        neighbour[inside & (y == first - 1)] = last + 1
    else:
        cut = core & (y == first)
        neighbour[cut] = last
        neighbour[inside & (y == last + 1)] = first - 1
    target = (neighbour < 0) | (neighbour >= ny)
    neighbour = np.where(target, -1, neighbour)
    twist = np.where(cut, step * shift_angle[x], 0.0)
    dzeta = zshift[x, np.maximum(neighbour, 0)] - zshift + twist
    return neighbour, np.where(target, 0.0, dzeta)


def recast(source, target, mxg=2, seed=None):
    """Write the grid ``source`` as cells along x for shiftedmap to ``target``.

    Parameters
    ----------
    source, target : str or Path
        Structured BOUT++ grid file, and the file to write
    mxg : int
        Number of x guard cells in ``source`` (left out)
    seed : int, optional
        If given, store the cells in a random order from this seed (the
        result must not depend on the order)

    Returns
    -------
    xs, ys : numpy arrays
        Structured (x, y) indices of the cells in the order stored
    """
    from netCDF4 import Dataset

    with Dataset(source) as d:
        v = d.variables
        nx, ny = int(v["nx"][...]), int(v["ny"][...])
        topology = {
            k: int(v[k][...]) if k in v else default
            for k, default in (
                ("ixseps1", nx),
                ("jyseps1_1", -1),
                ("jyseps2_1", ny // 2),
                ("jyseps1_2", ny // 2),
                ("jyseps2_2", ny - 1),
                ("y_boundary_guards", 0),
            )
        }
        if topology["jyseps2_1"] != topology["jyseps1_2"]:
            raise NotImplementedError("shiftedmap.recast supports single null only")
        if topology["y_boundary_guards"] != 0:
            raise NotImplementedError(
                "shiftedmap.recast needs a grid without y boundary cells"
            )
        zshift = np.asarray(v["zShift"][...], dtype=float)
        shift_angle = (
            np.asarray(v["ShiftAngle"][...], dtype=float)
            if "ShiftAngle" in v
            else np.zeros(nx)
        )
        cell_values = {
            name: np.asarray(var[...])
            for name, var in v.items()
            if var.dimensions[:2] == ("x", "y") and name not in TOPOLOGY
        }
        scalars = {
            name: var[...]
            for name, var in v.items()
            if var.ndim == 0 and name not in TOPOLOGY + ("nx", "ny")
        }
        attributes = {name: d.getncattr(name) for name in d.ncattrs()}
        extra_dims = {name: len(dim) for name, dim in d.dimensions.items()}

    # The interior cells, in structured order or shuffled
    xs, ys = np.meshgrid(np.arange(mxg, nx - mxg), np.arange(ny), indexing="ij")
    xs, ys = xs.ravel(), ys.ravel()
    if seed is not None:
        order = np.random.default_rng(seed).permutation(len(xs))
        xs, ys = xs[order], ys[order]
    ncells = len(xs)
    position = np.full((nx, ny), -1)
    position[xs, ys] = np.arange(ncells)

    values = {name: value[xs, ys][:, None] for name, value in cell_values.items()}
    # Without x guard cells BOUT++ cannot difference dx or dy itself
    values["d2x"] = np.zeros((ncells, 1))
    values.setdefault("d2y", np.zeros((ncells, 1)))
    faces = "J" in cell_values and "g_22" in cell_values
    if faces:
        g_22 = cell_values["g_22"]
        jxz = cell_values["J"] / np.sqrt(g_22)
        by = cell_values.get("By", cell_values.get("Bxy", np.ones(g_22.shape)))
        values["By"] = by[xs, ys][:, None]
    for name, step, face in (("forward", 1, "yhigh"), ("backward", -1, "ylow")):
        neighbour, dzeta = single_null_neighbours(topology, zshift, shift_angle, step)
        neighbour, dzeta = neighbour[xs, ys], dzeta[xs, ys]
        at_target = neighbour < 0
        other = position[xs, np.maximum(neighbour, 0)]
        values[f"pmap_{name}_x"] = np.where(at_target, ncells, other)[:, None]
        values[f"pmap_{name}_y"] = np.zeros((ncells, 1))
        values[f"pmap_{name}_dzeta"] = dzeta[:, None]
        # y of the neighbour, the cell's own at a target
        there = np.where(at_target, ys, neighbour)
        for component in SLICED:
            if component in cell_values:
                values[f"{name}_{component}"] = cell_values[component][xs, there][
                    :, None
                ]
        if faces:
            # The y faces as BOUT++ takes them on a field-aligned mesh: g_22
            # from the mean of sqrt(g_22), and the area J / sqrt(g_22) dx dz
            # averaged over the two cells. On FCI-like meshes BOUT++ reads
            # g_22 at the faces and takes the areas from By at the cell and
            # the face, so By at the face gives the same areas
            root = 0.5 * (np.sqrt(g_22[xs, ys]) + np.sqrt(g_22[xs, there]))
            values[f"g_22_cell_{face}"] = (root**2)[:, None]
            mean = 0.5 * (jxz[xs, ys] + jxz[xs, there])
            values[f"By_cell_{face}"] = (by[xs, ys] * jxz[xs, ys] / mean)[:, None]

    with Dataset(target, "w") as d:
        d.createDimension("x", ncells)
        d.createDimension("y", 1)
        for name, size in extra_dims.items():
            if name not in ("x", "y"):
                d.createDimension(name, size)
        attributes["parallel_transform"] = "shiftedmap"
        for name, value in attributes.items():
            d.setncattr(name, value)
        for name, value in (("nx", ncells), ("ny", 1)):
            d.createVariable(name, "i4")[...] = value
        for name, value in scalars.items():
            d.createVariable(name, np.asarray(value).dtype)[...] = value
        d.createVariable("ShiftAngle", "f8", ("x",))[...] = np.zeros(ncells)
        for name, value in values.items():
            dims = ("x", "y") + tuple(
                dim for dim in extra_dims if dim not in ("x", "y")
            )[: np.ndim(value) - 2]
            # BOUT++ reads grid fields as doubles (also the neighbour indices)
            d.createVariable(name, "f8", dims)[...] = value
    return xs, ys
