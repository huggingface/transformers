# iHELIX

**One architecture for fields on any geometry.** Spheres, images, volumes, periodic boxes, particle
clouds, and whatever you describe next.

```bash
pip install ihelix
```

```python
import torch
from ihelix import Axis, Geometry, FieldGrid, IHelixConfig, IHelixField

geometry = Geometry.atmosphere()                    # (latitude, longitude, pressure)
grid = FieldGrid.from_points(coords, geometry, num_neighbours=64, cluster_size=64)
model = IHelixField(IHelixConfig(in_channels=8), geometry)

forecast = model(fields, grid)                      # (B, T, N, C) in, same out
```

```bash
ihelix info       # what it is and which geometries it covers
ihelix domains    # run every shape of problem through one class
ihelix converge   # read one field at rising resolution
ihelix demo       # build a model and check the invariances
```

---

## The problem

Flatten a 3-D atmosphere into a 1-D sequence and the model has to learn spherical topology from scratch,
because nothing in an index tells it. Rotary embeddings do not help: they encode distance in *array
steps*, and on a latitude-longitude grid an array step means different things in different places.

- Cells one row apart sit `n_lon` positions apart in the sequence.
- The date line becomes a cliff: longitude 359.9° and 0.1° are neighbours on the planet and thousands of
  positions apart in the array.
- The meridians converge. Near the pole, a step of longitude is a few hundred metres; at the equator it is
  a hundred kilometres. The sequence calls both "one step".

A multi-mesh GNN fixes the topology by building the geometry into the wiring, which is honest but rigid:
the mesh is chosen up front, reaching across the domain costs hops, and a new resolution means a new mesh.

## The approach

Declare what your axes *are*. Everything downstream then works in a metric space where the geometry is
already correct, and no part of the model ever sees an index.

| you write | it becomes | which fixes |
| --- | --- | --- |
| `Axis("spherical")` | a point on a sphere in 3-D | the date line stops existing; the poles become ordinary points |
| `Axis("periodic", period=L)` | a point on a circle | a channel-flow box wraps because it *is* wrapped |
| `Axis("log", reference=1000)` | log-pressure | a factor of two in pressure is one distance, everywhere |
| `Axis("linear", scale=s)` | itself, scaled | a kilometre up is not a kilometre sideways, and you say by how much |

Three things follow, and they are what the package actually is:

**Neighbours are the nearest samples on the manifold**, not the adjacent entries in an array. A sample
beside the date line draws neighbours from both sides of it, because they are near.

**Displacements are expressed in each sample's own local frame** — east, north, up. "300 km east" produces
the same input vector at the equator and at 70°N. Polar distortion is not corrected; it is never
introduced.

**Receptive fields are lengths, and attention is weighted by area.** Each head carries a physical radius
with compact support, and every sample is weighted by the share of the domain it stands for. Attention
becomes a quadrature of a continuous operator,

```
out(x)  =  ∫ a(x,y) v(y) dμ(y)  /  ∫ a(x,y) dμ(y)
```

estimated on whatever samples you have. Refine the grid and it converges instead of drifting.

## The three strands

The braid from [`helix-lm`](../helix-lm), with "position in a sequence" replaced by "position on a
manifold" in the two spatial strands:

| strand | reads | cost |
| --- | --- | --- |
| **local** | nearest neighbours in metric space, per-head physical radii | `O(N · K)` |
| **index** | whole regions of the domain, retrieved by content through a hierarchy | `O(N · topk · cluster)` |
| **time** | a gated delta rule along the one axis that has an order | `O(T)`, fixed state per sample |

The **index** strand is the part a fixed mesh cannot do. Message passing couples regions by distance along
edges chosen in advance, so reaching the far side costs hops. Here the route is content-addressed and
learned: a ridge over the Atlantic can read the Pacific in one step, and nothing in the wiring had to
anticipate it. Teleconnections, distant vortices, the other side of an image — same mechanism.

Space has no order, so nothing recurrent runs across it. **Time** does, and that is the only place a
recurrence appears. Its state is a matrix per sample that does not grow with the rollout, so frame 1000
costs what frame 1 costs.

## What has been measured

On untrained models — these are properties of the arithmetic, not of training.

| property | result |
| --- | --- |
| **Sample order carries no information** | **exactly 0** difference under a full shuffle |
| **Longitude shift changes nothing** (local strand) | `1.8e-08` — float64 round-off |
| Refining 16×32 → 24×48 → 32×64, vs the finest | `0.455% → 0.174% → 0.090%`, halving each time |
| One class over image / video / volume / periodic box / point cloud / 1-D / globe / atmosphere | all run, all at full coverage |
| Time is causal; editing frame 5 leaves frames 0–4 | **exactly 0** |
| Temporal state size across 4, 32, 128 frames | identical |

The first row is the formal statement of "no flattening artefact", and it is exact rather than
approximate. A sequence model reading a flattened grid cannot say the same at any precision.

## The one precondition

Fixed-count neighbourhoods and fixed-length receptive fields pull against each other: refine the grid and
a fixed count covers less ground. Every attention module reports whether its neighbour list actually
reaches its radius:

```python
grid = FieldGrid.from_points(coords, geometry, num_neighbours=grid.suggest_neighbours(radius))
model.blocks[0].local.coverage(grid)     # 1.0 = every head decided by its radius, not by the list length
```

At `1.0` the invariances above are exact. Below it they degrade smoothly, and the number tells you so.
The neighbour count is a preprocessing choice — the weights never change.

## Honest limits

- **No trained checkpoints.** Nothing here has been trained on real data. Every claim above is about
  geometry, invariance and complexity, not about forecast skill or sample quality.
- **The region partition is a discretization.** The index strand carves the domain into regions, and no
  finite partition of a sphere commutes with arbitrary rotation — a fixed icosahedral mesh has exactly the
  same property. A longitude shift moves the local strand by round-off and the full model by ~5e-2.
- **Neighbour counts grow with density.** Holding a fixed physical reach at 4× the samples needs ~4× the
  neighbours. That is inherent to fixed-count neighbourhoods; `suggest_neighbours` computes it.
- **Vectors are not parallel-transported.** Components are read in each sample's local frame and the
  frame rotation is supplied to the model, but nothing forces it to transport them correctly.
- **Brute-force neighbour search.** Exact and cached per grid, which is right when the grid is static.
  Pass a precomputed index for very large point sets.

## License

Apache-2.0. The normalization, convolution, delta-rule and pooling kernels carry over from `helix-lm` and
ultimately from [HuggingFace Transformers](https://github.com/huggingface/transformers) (Apache-2.0); see
`NOTICE`. The geometry, grid, geodesic attention, regional index and cross-grid reader are original.

**Made by Nathan.**
