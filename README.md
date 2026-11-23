# Zoidberg: Flux-Coordinate Independent Grid Generator for BOUT++

Zoidberg is a tool for generating grids for BOUT++ when using the
Flux-Coordinate Independent (FCI) method of evaluating derivative
parallel to a magnetic field.

## Unstructured grids

`zoidberg.unstructured` generates FCI grids in which every poloidal plane
is an unstructured triangle mesh. It writes the parallel maps, the
perpendicular diffusion operator and, by default, the flux-reconstructed
conservative parallel divergence (K2) for BOUT++'s unstructured FCI
support. Meshing with gmsh needs the optional dependencies:

    pip install zoidberg[unstructured]  # gmsh, meshio

The K2 matrices use shapely (>= 2.0, a core dependency). A minimal
example, an annulus meshed independently in each of 8 planes with gmsh,
with Dirichlet walls:

```python
import numpy as np
from zoidberg import unstructured as us

th = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
outer = np.column_stack((2.0 + 0.5 * np.cos(th), 0.5 * np.sin(th)))
inner = np.column_stack((2.0 + 0.2 * np.cos(th), 0.2 * np.sin(th)))
ny = 8
planes = us.gmsh_planes([outer] * ny, [inner] * ny, mesh_size=0.05)
field = us.ShearedCylinderField(Bz0=0.1, Bzprime=0.1, Rc=2.0)
us.make_unstructured_grid(
    "annulus.fci.nc", planes, field, ny=ny, Ly=2.0 * np.pi, dirichlet_walls=True
)
```

For a jittered slab use `us.UnstructuredPoloidalGrid.slab(nR, nZ, Lx, Lz)`
as `poloidal` instead. The grid-file contract (variables and numbering) is
documented in the docstring of `zoidberg/unstructured.py`.
