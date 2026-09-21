# Reply to the ChatGPT framework review of 2026-09-21

Fable, 2026-09-21, against `CHATGPT_FRAMEWORK_REVIEW_2026-09-21.md` in this directory.

Every item below is marked with what I did about it, not with whether I agreed.
Where I checked a claim myself the check is named, and where I could not, it
says so. Two of the findings correct things I had written earlier the same day.

The review was made against `708ce9e`. Nine commits landed on this branch after
that and before this reply, so some of what it describes has moved; section 0
says what.

## 0. What changed under the review

| commit | what |
| --- | --- |
| `37ffe24` | the kernel profiler is reachable from `gs.init` |
| `8c0820b` | Wang et al. reuse, the `env_failed` buffer guard, insert-time dedup |
| `e65ea08` | a magnitude floor on the edge-edge side test, later removed |
| `d2a4103` | the contact grid is sized off the mesh |
| `4515183` | the cell cap follows it |
| `2a20f83` | the edge-edge crossing test is removed |
| `1035094` | a rigid body's contact block is built from the vertices that have contacts |
| `22c886a` | a contact that touches exactly gets the normal its geometry still has |
| `3f0e6fa` | a vertex sums its contacts in the mesh's order |

Measured on the python head over that span: the broadphase fell from 171.78 ms
a substep to 5.02 ms, a substep from 311.06 ms to 144.66 ms, and the 100 ms
probe went from refusing at 80 ms to completing.

## 1. Confirmed, and checked independently

### 6.2 Shell bending is not objective for a curved rest shape

**This is the most valuable finding in the review.** I verified it two ways.

The code is as described. `vbd_solver.py:549` defines
`E = w/2 |K x - K x_rest|^2` and `vbd_solver.py:1502` computes
`residual = kx - self.bend_info[i_s].kx_rest`, with `kx_rest` a world-space
vector frozen at rest.

The algebra holds: with `sum_i c_i = 0` the stencil kills a translation, so
under `x = R x_rest + t` we get `K x = R K x_rest` and

    E = (w/2) |(R - I) K x_rest|^2

which vanishes for every `R` only when `K x_rest = 0`.

Numerically, over 500 random rigid motions:

| stencil | \|K x_rest\| | worst energy under rigid motion |
| --- | --- | --- |
| coefficients that annihilate their own rest shape | 2e-16 | 3.1e-31 |
| curved rest, `K x_rest` kept in world space | 0.7336 | 1.0755 |

and the second matches the closed form `(w/2)(2|K x_rest|)^2 = 1.0762`, which
is what `R = -I` gives on the `K x_rest` direction.

So the spurious energy grows with the **square of the rest curvature**. Bergou
et al.'s model assumes a stencil that annihilates its rest shape; the
generalisation to a curved rest by subtracting a stored world-space vector is
what breaks it. A digestive tube is the worst case for this, and it is the next
thing we intend to simulate.

I have not fixed it. It is shell work, it wants one of the three remedies the
review lists, and the P0 test it asks for should be written first and be seen
to fail.

### 5.2 The old `nu >= 0.125` finding is a false positive

Verified: `vbd_solver.py:1119` stores `self.elems_info[i_e].lam = lam + mu`, so
the solver's `lam` is `lambda + mu` and the old derivation compared it against
the physical Lamé constant. No restriction should be added. Agreed.

### 10.1 Link to attachment incidence is a scan

Confirmed and measured. `vbd_rigid_attachment.py:260` is

    for i_a in range(attachment.n_attachments):
        if attachment.info[i_a].link == i_l:

run once per free body, on one thread, once per sweep. On the python head that
is 262 attachments x 39 free bodies x 12 sweeps = 122,616 scans a substep, of
which 3,144 do anything: **97.4 per cent waste**.

This is the same shape as the contact defect fixed in `1035094` the same day,
where 15,224 rigid contact vertices were visited a sweep to service 286 that
carried a pair. I had not noticed the attachment loop had it too. A CSR is the
right fix and I intend to write it.

### 12 The contact blind spot after `2a20f83`

Confirmed, independently of this review: an adversarial review run here built
the same counterexample, two crossing strips whose vertices never enter the
other's face footprint, and verified no vertex-face hit at any point on the
sweep.

**Closed, differently from the suggested patch.** `vbd_accd.py:115-130` shows
ACCD already sweeps edge-edge pairs, so the filter prevents exactly the case
the deleted test would have reported. The blocker to turning it on was not cost
but a defect: both contact geometries divided the offset between the closest
points by its own length, so a pair that touched exactly produced a non-finite
normal and the run died before the filter did anything. `22c886a` fixes that;
the app now defaults `contact_ccd` on.

The review's fail-loud oracle remains the right answer for scenes that cannot
use the filter, which is any scene with a prescribed collider or an articulated
body, because the solver refuses it for both. That gap is still open.

## 2. Where the review corrects me

### 13.1 ACCD is a safety clamp, not a complete contact integrator

I reported to the owner that turning the filter on "costs nothing". That was
measured in wall clock only, which is the wrong measure, and the review says
so. What it costs is physical motion:

| frame | min TOI |
| --- | --- |
| 2 to 8 | 1.000000 |
| 9 | 0.373855 |
| 10 | 0.058956 |
| 11 | 0.058956 |

Three of ten frames contain a clamped substep, and one keeps 5.9 per cent of
its solved motion while the clock advances by the whole substep. The clamping
is confined to the last three frames, which is also where the run previously
failed, so the filter may be holding up a configuration rather than resolving
it. "Numerically safe but physically suspect" is the right description and I
withdraw the word free.

The lost-time diagnostics of 13.2 are the correct gate on a setting I have just
turned on by default, and are the next thing I will add.

### 15 Grid cell size and search reach are still coupled

Confirmed, and the coupling is mine, introduced the same day. `d2a4103` made
the cell `max(median_edge, 2 * (max_thickness + margin_max))`, so raising
`margin_max` to enable reuse coarsens the grid and pushes occupancy towards
`contact_cell_cap`. The floor exists so that inflating a box by the reach can
never add more than a cell a side, which is a real constraint, so the two are
not trivially separable. It needs a deliberate answer rather than a deletion.

### 16 and 25.1 The documented contact-grid memory is wrong

Confirmed, and worse than the review states, because `cell_c` was added the
same day. At `hash_cap = 256` and `n_cv = 15,224`:

    cell_v   31.18 MB
    cell_c   93.54 MB
    total   124.72 MB

My commit message for `4515183` says 31 MB, which is `cell_v` alone. A four
fold undercount in my own message. With a gut wall and a meal at B=16 the grid
alone reaches about 4.7 GB.

## 3. Agreed, not yet acted on

- **14, candidate reuse.** Agreed that it is inert, since nothing sets
  `contact_margin_max` above `contact_margin`. The proof obligation stated here
  is sharper than the one I had: the budget is reduced by motion that has
  already happened, so exhaustion may be reported after an uncovered motion
  occurred. That has to be settled before the path is ever enabled, and it is
  currently paying `cell_c`'s memory for nothing.
- **7, warm-start decay.** Agreed that fixed per-substep factors make dual
  memory depend on the substep count. The `exp(-h/tau)` form is right.
- **8, differentiability matrix.** Agreed.
- **10.2, sparse articulation incidence.** Agreed, and it matters once the
  vertebral chain exists.
- **21.1, the cross-repository test oracle.** Agreed. An engine test must not
  depend on `snakeSimWithAstra`.
- **13.3 and 17, split-step CCD and Offset Geometric Contact.** Both are R&D
  sized and neither is the current bottleneck.

## 4. What the review does not cover, and should know

**This scene is not reproducible.** Two runs of one binary on the python head
were bit-identical for four frames and then diverged, reaching 49 mm at 100 ms.
The seed is that a candidate pair takes its index from an atomic, so each
vertex accumulated its force and Hessian in whatever order the GPU finished in,
and float addition is not associative. `3f0e6fa` sorts each vertex's list on
mesh topology, which moves the first difference from frame 5 to frame 7 and
does not remove it; a second source is still being looked for.

Two consequences for this review's recommendations. Any acceptance criterion
that is a trajectory quantity, which is all of the pin, attachment and
inversion checks, currently has no error bar. And the amplification is about
ten a frame from a seed far below any measurable scale, which says the contact
configuration of one jaw joint, `coronoid.R` against `angular.R`, is close to a
discontinuity in the solver's own logic. That is the same joint that produced
the nine false crossings which motivated `2a20f83`. Determinism would make that
repeatable, not trustworthy, and the modelling question is the larger one.

## 5. One disagreement of emphasis

The review's priority order puts the shell and packet items above the contact
ones. On the evidence of today I would put **13.2 lost-time diagnostics** and
the **fail-loud oracle for scenes that cannot use ACCD** above them, because we
have just changed the default for every bake this project makes and currently
cannot see what that costs beyond one number a frame.

The shell bending finding is nonetheless the one I would want fixed before any
gut work begins, and I would not start that work until its P0 test exists and
passes.
