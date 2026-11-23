#!/usr/bin/env python3
"""Unstructured meshes of the poloidal cross-sections of an island divertor.

The data in this directory are 16 poloidal cross-sections of a stellarator
vessel (``vessel.json``: one closed contour of (R, Z) points in metres per
cross-section) and of a divertor plate (``d17v3.json``: a closed polygon in
the cross-sections that cut it, empty elsewhere). Every cross-section is
meshed independently with gmsh: the vessel contour is the outer wall and the
plate a hole, both as polygons, so that their corners are kept.

    python island_divertor.py                  # mesh size 2 cm
    python island_divertor.py --mesh-size 0.01 --plot meshes.png

A 3D FCI grid also needs the magnetic field and the toroidal angles of the
cross-sections, which are not part of this example.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from zoidberg.unstructured import UnstructuredPoloidalGrid

HERE = Path(__file__).resolve().parent


def load():
    """Vessel contours and plate polygons (None where there is no plate)."""
    vessel = [
        np.array(c, dtype=float) for c in json.loads((HERE / "vessel.json").read_text())
    ]
    plate = [
        np.array(c, dtype=float) if len(c) else None
        for c in json.loads((HERE / "d17v3.json").read_text())
    ]
    return vessel, plate


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh-size", type=float, default=0.02, help="[m]")
    parser.add_argument("--plot", help="save a plot of the meshes to this file")
    args = parser.parse_args()

    vessel, plate = load()
    planes = []
    for k, (outer, inner) in enumerate(zip(vessel, plate)):
        pol = UnstructuredPoloidalGrid.from_gmsh(
            outer, inner, mesh_size=args.mesh_size, polygon=True
        )
        w_in, w_out = pol.walls
        print(
            f"cross-section {k:2d}: {pol.n_nodes} nodes, "
            f"{len(w_out)} on the vessel, {len(w_in)} on the plate"
        )
        planes.append(pol)

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, (len(planes) + 1) // 2, figsize=(4 * len(planes) // 2, 9)
        )
        for ax, pol, outer, inner in zip(axes.flat, planes, vessel, plate):
            ax.triplot(pol.R, pol.Z, pol.triangles, lw=0.2, color="0.4")
            for curve, colour in ((outer, "b"), (inner, "r")):
                if curve is not None:
                    ax.plot(*np.vstack([curve, curve[:1]]).T, colour, lw=1.0)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.savefig(args.plot, dpi=110, bbox_inches="tight")
        print("Saved", args.plot)


if __name__ == "__main__":
    main()
