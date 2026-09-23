# How iHELIX and NatureV1 work

This is the detailed version. The short version: **most models reach a grid by flattening it and hoping a
positional encoding puts the geometry back.** iHELIX does the opposite — you declare what your axes are,
and everything downstream works in a space where the geometry is already correct.

---

## 1. The problem with flattening

Take a 240×121 latitude/longitude grid and flatten it row-major. Three things break:

| | what the model sees | what is true |
|---|---|---|
| **Vertical neighbours** | 240 positions apart | adjacent |
| **The date line** | 359.9° and 0.1° are 239 positions apart | adjacent |
| **The poles** | ordinary interior points | all 240 entries are one place |

A positional encoding gives the model an index. Nothing in an index says that cell 12,240 is directly
above cell 12,000, or that the last column touches the first. So the model has to *learn spherical
topology from data* — spending capacity rediscovering something you already knew.

And it never quite succeeds, because the errors are worst exactly where the data is sparsest: near the
poles, where meridians converge and grid cells shrink to nothing.

---

## 2. Geometry: what replaces "position in a sequence"

You declare axes. Each one says what kind of thing it is:

```python
Geometry([Axis("linear"), Axis("linear")])                          # an image
Geometry([Axis("linear")] * 3)                                      # a volume or point cloud
Geometry([Axis("periodic", period=L)] * 2 + [Axis("linear")])       # a channel-flow box
Geometry.globe()                                                    # the Earth's surface
Geometry.atmosphere()                                               # Earth + pressure
```

Each axis maps into a Euclidean space where **straight-line distance already is the distance you care
about**:

- **spherical** — `(latitude, longitude)` → a point on a sphere in 3-D. Chordal distance there is a
  monotone function of great-circle distance. The poles stop being special. The date line **stops
  existing**: 359.9° and 0.1° are near each other because they *are* near each other.
- **periodic** — → a circle. Same trick, for a wind-tunnel box.
- **log** — pressure. The model thinks in *factors* of pressure, not hectopascals, so equal ratios are
  equal distances.
- **linear** — passes through, scaled.

Every axis carries a `scale` in real units (km, say). That's how you declare that a kilometre upward is
not meteorologically the same as a kilometre sideways.

### Local frames — the part that actually kills polar distortion

At every point, the geometry provides an orthonormal frame: **east, north, up**. Neighbour offsets are
expressed in *that point's own frame*.

So "300 km east" produces the **same input vector** at the equator and at 80°N. The distortion isn't
corrected by a factor — it is never introduced, because the representation never had it.

> Measured: frames orthonormal to 1.11e-16 at the poles. Date-line distance 22 km, not 12,742 km.

---

## 3. The grid: data, not architecture

A `FieldGrid` holds two structures, computed once from the sample positions:

**Neighbour graph** — for each point, its nearest neighbours *in metric space*. A point beside the date
line gets neighbours on both sides of it. A polar point gets the ones physically close, not the ones the
array happens to list adjacently.

**Hierarchy** — a tree of nested spatial clusters, built by recursive splitting. The split is along the
**principal axis** (`torch.linalg.eigh` of the covariance), not the widest coordinate axis — coordinate
axes are an arbitrary choice of frame, so splitting on them would make the tree depend on how the domain
happens to be oriented.

The model takes the grid **as an argument**. Swap in a finer grid and the same weights run unchanged.
That is the entire answer to resolution: the model never sees a grid shape, a resolution, or a
dimensionality. It sees points, distances and weights.

---

## 4. The three strands

Every block braids three mixers. This is the core of the architecture.

### Strand I — Geodesic attention (short range)

Each sample attends to its nearest neighbours in metric space. Three things are added to the logits:

1. **A learned geometric bias** from the displacement in the query's own frame, plus log-distance, plus
   how much the neighbour's frame is rotated relative to the query's (which on a curved manifold is what
   you need to parallel-transport a direction).
2. **A per-head physical window.** Each head has its own radius — a *length*, not a count. At 88M the
   eight heads sit on a geometric ladder: **120, 174, 252, 364, 527, 763, 1105, 1600 km**. One layer sees
   eight scales at once, and those scales mean the same thing when the sampling changes.
3. **Quadrature weights** — how much of the domain each sample stands for.

That third term is what makes attention a discretization of a continuous operator:

```
out(x) = ∫ a(x,y) v(y) dμ(y) / ∫ a(x,y) dμ(y)
```

Refine the grid and the estimate **converges** instead of drifting.

> Measured convergence: 0.455% → 0.174% → 0.090% → 0.034% as the grid refines.

**The window has compact support** — exactly zero past 3σ, not merely small. This is not cosmetic. It's
what makes the invariances hold *to the last bit* rather than to a few decimals. An early version
truncated at K neighbours instead, and the permutation error was 3.0e-2; with compact support it is
**exactly 0**.

### Strand II — Index attention (long range)

Local attention reaches 1,600 km. A teleconnection reaches across an ocean.

The domain is summarized bottom-up into a tree of nested regions. Each query region computes a routing
vector, descends the tree keeping a beam, and ends up attending over the samples of the `topk` regions it
selected — which may be **anywhere, at any distance**.

This is the part a fixed mesh cannot do. Message passing on a multi-mesh (GraphCast) couples regions by
distance along fixed edges, so reaching the far side costs hops and the route is decided in advance.
Here the route is **content-addressed and learned**: a ridge over the Atlantic can read the Pacific in one
step, and nothing in the wiring had to anticipate it.

**Making a discrete top-k differentiable.** Selection is argmax — no gradient. So the routing score is
accumulated down the descent path and fed back as an additive bias on the retrieved logits, via
`logsigmoid(path_score)`. Gradient reaches every level of the tree. (An early version didn't accumulate,
and the node pooler received no gradient at all.)

8 of the 17 layers carry the index strand, interleaved every other layer.

### Strand III — Temporal (the gated delta rule)

**Space has no order, so nothing recurrent runs across it.** Time does, and this is the only place a
recurrence runs.

State is a **matrix per head per sample** whose size does not depend on how many frames have passed. A
rollout costs the same at frame 1000 as at frame 1 — which is what makes long autoregressive integrations
affordable. Training uses a chunkwise-parallel UT transform, so it parallelizes rather than stepping.

A per-sample, per-head gate mixes strand I and strand II. Then a feed-forward, with RMSNorm and residuals
throughout.

---

## 5. Encode → process → decode

```
satellite (2 km)   ─┐
analysis  (25 km)  ─┼─► cross-attention ─► Fibonacci mesh (4,096 pts) ─► 17 blocks ─► cross-attention ─► any output grid
environment        ─┘
```

Each source has **its own input projection and its own read radius** — a satellite scene at 2 km and an
analysis field at 25 km are not the same kind of measurement and shouldn't pretend to be. What they share
is the mesh they're read onto, which makes fusing them a **sum** rather than a resampling.

Each encoder carries a learned `absent` vector, so a mesh point with no coverage (the far side of the
planet from a geostationary satellite) produces a clean absence rather than attending to noise.

The mesh is a **Fibonacci sphere** — near-equal-area, so there's no pole to distort and no seam to cross.

---

## 6. Why this beats a ViT/CNN on satellite imagery

| | ViT / CNN | iHELIX |
|---|---|---|
| Pixel size | assumed constant | **measured**: a GOES limb pixel covers 64× the ground of a nadir pixel |
| Location | `image[400, 1200]` says nothing | every sample carries lat/lon + true footprint area |
| Values | normalized to 0–1, physics erased | brightness temperature stays in kelvin |
| Two sources | resample one onto the other, lose information | read both onto one mesh, add |

The projection is solved per-pixel from GOES-R PUG Vol 3 §5.1.2.8.1 — validated against the file's own
metadata at 2.07 km at nadir where nominal C13 resolution is 2 km.

---

## 7. Parameter budget (88M)

```
blocks                84.67M   95.1%   ← the 17 braided blocks
analysis_encoder       0.80M    0.9%
environment_encoder    0.79M    0.9%
satellite_encoder      0.79M    0.9%
decoder                0.79M    0.9%
ri_head                0.40M    0.4%
summary_pool           0.26M    0.3%
track_head             0.14M    0.2%
eyewall_head           0.13M    0.1%
field_head             0.07M    0.1%
...
```

95% of the model is the shared processor. The heads are almost free — which is exactly what makes freezing
the backbone and fine-tuning on 24,585 storm points viable: **0.96M trainable, 1.1%**.

---

## 8. Every output is a distribution

- **Track** — weighted scenarios, each a full trajectory with a *tilting* uncertainty ellipse (two sigmas
  and a correlation, so the ellipse can align with the direction of travel). Reads as "62% recurves
  offshore, 26% tracks the coast, 12% inland" rather than an average track through somewhere the storm was
  never going. *Measured: collapsing to one line scores 116 under the mixture likelihood where keeping both
  branches scores 8.2.*
- **Fields** — per-point mean and variance, per lead time.
- **Eyewall** — peak wind, RMW, and 34/50/64 kt radii per quadrant, each with intervals.
- **RI** — probability per threshold, plus expected 24 h change and likeliest onset.
- **Landfall** — probability per lead time.

Heads start at **Atlantic climatology**, not zero: 65 ± 30 kt peak wind, 985 hPa, RI at the measured 4.11%
base rate, track spread growing 0.6°→12° with lead. Without that, the intensity term alone was 119,191 of
a 60,565 total loss.

---

## 9. The guarantees, and their one precondition

| Property | Status |
|---|---|
| Sample order doesn't matter | **exact** (atol=0) |
| Spinning the globe in longitude | **exact** for the local strand (atol=1e-6) |
| Refining the grid converges | 0.455% → 0.034% |
| Time is causal, space is not | enforced |
| Temporal state doesn't grow | constant |

**The one precondition:** each head's neighbour list must reach 3× its widest radius. `coverage()` reports
whether it does, and the model warns at construction if not. At coverage 1.0 the samples beyond a head's
window contribute *exactly* nothing and the invariances are exact.

**One honest limit:** the index strand is *not* rotation-equivariant, because carving a domain into
regions is a discretization and no finite partition of a sphere commutes with arbitrary rotation. A fixed
icosahedral mesh has the same property. Sub-regions aren't isotropic, so every split below the first is
well determined.

---

## 10. What is not claimed

**No weights here have been trained.** The architecture, losses, ingest and training loop are complete and
exercised end to end on real data, but no accuracy has been measured, and nothing here should inform a
decision about a real storm. The National Hurricane Center is the authoritative source.

The next honest step is baselines: score against **persistence** ("tomorrow = today") and **climatology**
on the same held-out batches. A model that ties climatology has learned nothing; likelihoods alone can't
tell you which.
