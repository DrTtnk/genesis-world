# ChatGPT end-to-end framework review — VBD / AVBD / biomechanics extensions

**Date:** 2026-09-21  
**Repository:** `DrTtnk/genesis-world`  
**Reviewed baseline:** `2a20f83e05430fd643ffcbdc2bb9a84655505b0c`  
**Review type:** static architecture, implementation, numerical-method, and research cross-check  
**Reviewer:** ChatGPT (OpenAI), at the explicit request and authorization of the repository owner/user

> **Authorization note.** This review was requested by the repository owner/user, who explicitly authorized ChatGPT to inspect the fork and commit this review document. The code review itself was read-only; this document is the only change made by ChatGPT in this commit.

---

## 0. Scope, provenance, and limitations

This document is an end-to-end review of the custom simulation work layered onto Genesis World for the SnakeSim / biomechanical simulation stack. It covers the major custom subsystems visible on the reviewed development line:

- VBD tetrahedral elasticity
- stable neo-Hookean material handling
- shell membrane and bending elements
- floor and mesh contact
- swept contact broad phase
- additive continuous collision detection (ACCD)
- AVBD-style augmented-Lagrangian hard constraints
- free rigid-body VBD coupling
- articulated rigid coupling
- tissue-to-tissue and tissue-to-rigid attachments
- routed Hill-type muscle-tendon units and ligaments
- differentiable VBD, including converged KKT adjoints and reverse-sweep differentiation
- AVBD packet/model import layer
- relevant MPM and PBD coupling / gradient fixes
- tests, diagnostics, failure semantics, and GPU-oriented data structures

The development line from the upstream merge base contains roughly eighty custom commits and is no longer a small patch set. It is effectively a specialized biomechanics solver built inside Genesis.

This review is **static**. I read the implementation and cross-checked the numerical choices against the related literature, but I did **not** run the GPU test suite in this review session. Therefore:

- statements marked as implementation defects are based on direct code-contract or mathematical inconsistencies;
- statements marked as risks or proof obligations should be confirmed by targeted tests;
- performance recommendations should be validated on an otherwise idle GPU with the existing profiler hooks.

Three owner-provided notes are treated as known facts for this baseline:

1. **Commit `4515183` memory accounting erratum.** The documented “31 MB grid” figure is wrong by approximately 4×. At cap 256 the actual total is about **124.72 MB**, because `cell_c` is a Vector3 of int32 at the same shape as `cell_v`. At `B=16` with the meal mesh, this can reach roughly **4.7 GB**.
2. **Commit `2a20f83` contact blind spot.** Removing the edge-edge crossing test fixed false positives, but leaves a verified class of crossings where two quad strips can pass through each other without a vertex entering the opposite face footprint. Point-triangle crossing tests cannot detect that class at any threshold.
3. **Commit `8c0820b` candidate-reuse path is inert in the current app.** `contact_margin_max` is not set, therefore it defaults to `contact_margin`, and the candidate set rebuilds every moving substep while the memory cost of the reuse machinery is still paid.

---

# 1. Executive summary

The central result of this review is encouraging:

**The mathematically central parts of the VBD/AVBD implementation are stronger than the contact layer surrounding them.**

The VBD core, stable neo-Hookean tetrahedra, graph-colored block Gauss-Seidel structure, augmented-Lagrangian constraints, damping formulation, and differentiation architecture are all broadly consistent with the relevant literature and show unusually good implementation discipline for research code.

The largest remaining risk has shifted to:

1. **codimensional / shell contact completeness,**
2. **contact broad-phase memory scaling,**
3. **candidate reuse semantics,**
4. **shell bending objectivity for curved rest shapes,**
5. **public API / packet capability contracts.**

A concise priority order is:

| Priority | Action |
|---|---|
| P0 | Preserve the reviewed development line with a permanent branch/tag |
| P0 | Add a rigid-rotation test for curved-rest shell bending |
| P0 | Add an edge-edge swept crossing safety oracle for shell/thin-surface contact |
| P0 | Fix packet `CAPABILITIES` so validation cannot advertise unsupported builder features |
| P1 | Separate contact grid cell-size tuning from `contact_margin_max` |
| P1 | Add contact-memory / occupancy / CCD lost-time diagnostics |
| P1 | Replace dense/scan-heavy rigid attachment incidence with CSR/sparse incidence |
| P1 | Add MTU root residual diagnostics and safeguarded fallback |
| P1 | Make augmented-Lagrangian warm-start decay time-step invariant |
| P1 | Remove the cross-repository test dependency on `snakeSimWithAstra` |
| P2 | Add shell geometry to the AVBD packet schema |
| P2 | Replace fixed-cap hash buckets with compact sparse cell storage |
| P2 | Move from post-solve ACCD rescaling toward true split-step CCD |
| R&D | Prototype Offset Geometric Contact (OGC) as a VBD-native contact v2 |

---

# 2. End-to-end architecture as implemented

The custom stack can be viewed as:

```text
anatomical / authored model
        |
        v
genesis.avbd.packet
    load / validate / provenance
        |
        v
genesis.avbd.model
    assemble Genesis entities
        |
        +--> rigid links / joints
        +--> tet or shell tissue runtime entities
        +--> attachments
        +--> MTUs / ligaments / restraints
        +--> collision groups / rules / prescribed colliders
        |
        v
scene.build()
        |
        v
VBD solver build
    rest data
    vertex masses
    element tables
    graph coloring
    incidence CSR
    constraint ownership CSR
    contact structures
        |
        v
physics substep
    rigid/articulation begin
    warm starts
    inertial prediction
    contact search
    MTU state update
        |
        v
colored block sweeps
    3x3 soft vertex blocks
    6x6 free rigid blocks
    scalar articulated-coordinate blocks
    AL constraints
    contact/friction
    routed MTUs
        |
        v
optional ACCD safety rescale
        |
        v
velocity update
diagnostics / failure latch
rigid-state commit
        |
        v
optional differentiation
    converged KKT adjoint
        OR
    exact reverse of executed sweeps
```

This separation is a major strength. In particular, the `packet -> assembler -> runtime solver` split is the beginning of a stable simulation interface rather than an application-specific pile of setup code.

---

# 3. Strong points worth preserving

## 3.1 Scientific / engineering discipline

The project is unusually explicit about the difference between:

- implemented,
- numerically tested,
- physically validated,
- accepted for a specific anatomical scenario.

That distinction should be preserved. It prevents the common simulation-research failure mode where “the code ran without NaN” is accidentally promoted into “the model is correct.”

The same applies to the many explicit failure paths. The code often refuses unsupported states instead of silently inventing semantics. That is exactly the right bias for a biomechanics framework.

## 3.2 VBD decomposition

The VBD implementation follows the block-coordinate structure cleanly:

- local 3×3 vertex solves,
- graph coloring for race-free Gauss-Seidel,
- CSR incidence instead of global scans for tetrahedral/shell neighborhoods,
- a fixed computation budget in the non-converged path,
- optional convergence for the differentiable path.

This structure is highly compatible with GPU execution and with the original VBD motivation.

Reference:

- Anka He Chen, Ziheng Liu, Yin Yang, Cem Yuksel, **Vertex Block Descent**, SIGGRAPH 2024  
  https://graphics.cs.utah.edu/research/projects/vbd/  
  https://graphics.cs.utah.edu/research/projects/vbd/vbd-siggraph2024.pdf

## 3.3 Stable neo-Hookean elasticity

Using the stable neo-Hookean model is a good choice for the intended regime: large rotations, biological soft tissue, and near-incompressibility.

Reference:

- Breannan Smith, Fernando de Goes, Theodore Kim, **Stable Neo-Hookean Flesh Simulation**, ACM TOG 2018  
  https://graphics.pixar.com/library/StableElasticity/paper.pdf

## 3.4 Damping implementation

The tetrahedral Rayleigh-style damping uses the action of the **full rest stiffness row**, including neighbor terms, rather than a diagonal-only approximation. This matters because diagonal-only “stiffness times velocity” damping produces spurious forces under rigid translation.

The current implementation therefore gets an important property right:

**rigid translation is annihilated by the damping operator.**

The shell membrane path similarly uses a rest-Hessian action and deliberately omits bending damping because the current curved-rest bending model does not have the right rigid-rotation nullspace.

That deliberate omission is preferable to adding physically wrong damping for convenience.

## 3.5 Augmented-Lagrangian hard constraints

The AVBD-style constraint machinery is thoughtful:

- per-environment multipliers,
- bounded constraints with clamped dual variables,
- stiffness ramping,
- distinct distance and angle scales,
- explicit effective active stiffness `k_eff`,
- converged Uzawa path for equation-level differentiation,
- low-sweep path for real-time/RL use.

Reference:

- Chris Giles, Elie Diaz, Cem Yuksel, **Augmented Vertex Block Descent**, SIGGRAPH 2025  
  https://graphics.cs.utah.edu/research/projects/avbd/  
  https://graphics.cs.utah.edu/research/projects/avbd/Augmented_VBD-SIGGRAPH25.pdf

## 3.6 Differentiation architecture

The fork supports two importantly different gradient semantics:

### A. Converged forward -> KKT / implicit adjoint

Use this when the forward solve is intended to represent the converged stationarity / constrained solution.

### B. Fixed finite sweeps -> reverse the executed solver

Use this when the actual forward computation is intentionally truncated to a fixed number of sweeps.

This distinction is not cosmetic. A recent 2026 result makes exactly this point: differentiating the converged equation is not the same as differentiating the finite block solver that actually ran.

Reference:

- Lei Shu et al., **Differentiate the Solver, Not the Equation: Reverse-Sweep Adjoints for Block Implicit Simulation**, 2026  
  https://arxiv.org/abs/2608.08559

This is one of the most forward-looking aspects of the implementation. Do not collapse these two modes into one “gradient” mode; document them as separate semantics.

## 3.7 Failure latching and diagnostics

The pattern of:

- per-environment error state,
- failed-substep capture,
- freeze failed environments while the batch continues,
- opt-in global raising,
- persistent inversion diagnostics,

is very good for batched simulation.

This is particularly important for RL, where one bad environment should not necessarily terminate every environment in the batch.

## 3.8 Routed MTUs as a separate subsystem

The MTU implementation correctly separates:

- route geometry,
- activation state,
- fibre state,
- tendon equilibrium,
- force application.

The choice to advance activation/fibre state **once per substep**, while re-evaluating force against the moving VBD iterate during sweeps, is especially good. It avoids making physiology depend on the numerical iteration count.

---

# 4. VBD core review

## 4.1 Local block structure

For the stable neo-Hookean tetrahedra, the local vertex system has the expected structure:

```text
H_i = M_i / h^2
    + elastic PSD / local curvature
    + damping block
    + constraint blocks
    + contact / friction blocks
    + MTU blocks
```

The use of local inverse solves is exactly the VBD design target.

The graph coloring includes tetrahedra, shell triangles, bending stencils, distance constraints, and angle constraints, which is important: a bending stencil couples all four vertices, not just the shared edge.

## 4.2 Accumulation precision

`accumulate_f64` is a sensible knob.

Near-incompressible materials, small/sliver elements, and mixed stiffness ratios can lose substantial digits in float32 assembly even when state storage remains float32.

Recommendation:

- keep float64 accumulation available;
- record in experiment manifests whether it is enabled;
- add a performance/accuracy benchmark specifically for the Python anatomy at representative Poisson ratios.

## 4.3 Energy diagnostics

`compute_energy` correctly refuses shell scenes because it does not include shell energy. This is much better than returning a plausible but incomplete scalar.

Longer-term, consider exposing a structured energy diagnostic:

```text
inertia
tet_elastic
shell_membrane
shell_bending
constraints
contact_penalty
attachments
MTU_tendon
MTU_fibre
damping_dissipation
```

This would be extremely useful during validation and parameter fitting.

---

# 5. Re-check of the older Opus Review 3

The older `snakeSim/docs/reviews/OPUS_REVIEW_3.md` contained several valuable findings. A comparison against the reviewed baseline indicates that many of them have already been corrected.

## 5.1 Findings that appear fixed

The reviewed code now contains fixes for the following older issues:

- checkpoint constraint state now uses a start-of-window snapshot (`_cons_window_start`);
- constraint history recording is performed explicitly when the converged solve returns, avoiding stale history when zero sweeps are needed;
- `k_eff` is recorded explicitly rather than inferring constraint activity only from multiplier value;
- angle adjoint residuals are weighted where their gradient actually acts;
- stationarity and constraint violation are no longer collapsed into one absolute tolerance with incompatible physical units;
- `_held_actu` is cleared on gradient/reset setup;
- dual-step and sweep budgets are separated;
- dual-adjoint buffers are reset from Python scope.

These are substantive improvements.

## 5.2 Important correction: the old `nu >= 0.125` K0 finding appears to be a false positive

The old review argued that the rest stiffness used for damping is PSD only for Poisson ratio `nu >= 0.125`.

However, the runtime element setup stores:

```python
self.elems_info[i_e].lam = lam + mu
```

so the internal variable called `lam` in the VBD solver is actually:

```text
lambda' = lambda + mu
```

for the stable neo-Hookean formulation.

The quadratic rest form therefore becomes:

```text
2 mu ||S||^2 + (lambda' - mu) tr(S)^2
= 2 mu ||S||^2 + lambda tr(S)^2
```

The condition `lambda' >= mu/3` becomes:

```text
lambda >= -2 mu / 3
```

and using

```text
lambda / mu = 2 nu / (1 - 2 nu)
```

this reduces to approximately:

```text
nu >= -1
```

which is already the admissible lower domain of the material.

Therefore, **do not add a `nu >= 0.125` restriction based on the older review.**

The old derivation appears to have confused physical Lamé `lambda` with the solver’s stored `lambda' = lambda + mu`.

---

# 6. Shells

## 6.1 Membrane path

The membrane element is a clean extension of the stable neo-Hookean idea onto a 3×2 deformation gradient.

The test strategy is good:

- independent math oracle,
- direct force comparison,
- random configurations,
- explicit float32/float64 tolerance handling.

## 6.2 Bending path: likely objectivity problem for curved rest shapes

This is the most important new mathematical concern found in this review.

The current bending energy is effectively:

```text
E_b = (k w / 2) || K x - K x_rest ||^2
```

with scalar stencil coefficients `K` fixed at rest and `K x_rest` stored as a world-space vector.

For a pure rigid transformation:

```text
x = R x_rest + t
```

and because a curvature stencil satisfies translational invariance:

```text
K 1 = 0
```

we obtain:

```text
K x = R K x_rest
```

so:

```text
E_b = (k w / 2) || (R - I) K x_rest ||^2
```

For a **curved** rest shape, `K x_rest != 0`, so a generic rigid rotation produces:

```text
E_b > 0
```

despite zero physical deformation.

This means the issue is not only that the bending **damping** Hessian would brake a rotating curved shell; the **elastic curved-rest bending energy itself** is likely not objective.

### P0 test to add

A four-vertex curved-rest bending stencil should be subjected to a pure rigid rotation and translation:

```text
1. construct a non-flat rest stencil;
2. compute force at rest;
3. rotate every vertex by a generic 3D R and add translation;
4. compute bending force;
5. require force ~= 0 and energy unchanged.
```

Expected outcome from the current formula: this test is likely to fail.

### Possible fixes

Options include:

- dihedral-angle bending relative to a rest dihedral;
- a corotated rest-curvature representation;
- transporting the rest curvature vector into a local/current frame before differencing.

The current quadratic model is attractive because of its constant Hessian, but for an anatomical gut wall a rigid-motion-invariant energy is more important than retaining a constant block.

Reference for the quadratic bending family:

- Miklós Bergou, Max Wardetzky, David Harmon, Denis Zorin, Eitan Grinspun, **A Quadratic Bending Model for Inextensible Surfaces**, 2006  
  https://diglib.eg.org/items/edfc6fd8-504e-4c0c-89d6-35579d1c8b39  
  DOI: https://doi.org/10.2312/SGP/SGP06/227-230

---

# 7. Augmented-Lagrangian warm start

The current hard-constraint warm start uses fixed per-substep factors approximately of the form:

```text
lambda <- 0.95 * 0.99 * lambda
k      <- max(k_start, 0.99 * k)
```

There is an empirical reason for applying this every physics substep: in the tested ladder/Python regime, less-frequent decay allowed stale dual state to produce significantly larger stretch.

The remaining concern is **discretization dependence**.

If the same physical simulation is run with:

```text
10 substeps
```

versus:

```text
40 substeps
```

per outer step, fixed decay per substep changes the amount of dual memory retained per second of simulated time.

Recommended formulation:

```text
alpha(h) = exp(-h / tau_lambda)
gamma(h) = exp(-h / tau_k)
```

Choose `tau_lambda` and `tau_k` so the current default timestep reproduces existing behavior.

This converts the warm-start memory from a numerical artifact into a time-scale parameter.

---

# 8. Differentiability matrix should become explicit

The solver is now rich enough that a single boolean `requires_grad` no longer communicates the real differentiation contract.

A public capability table should be maintained, for example:

| Feature | Forward | Reverse fixed sweeps | Converged adjoint |
|---|---:|---:|---:|
| Tet elasticity | yes | yes | yes |
| Fibre reinforcement | yes | yes | yes |
| Shell membrane | yes | incomplete / verify | incomplete / verify |
| Shell bending | yes | incomplete / verify | incomplete / verify |
| Floor contact/friction | yes | yes | yes |
| Mesh contact | yes | partial | partial |
| Self collision | yes | no | no |
| Free rigid attachment | yes | partial | partial |
| Articulation | yes | partial | partial |
| Tissue attachment | yes | no | no |
| MTU | yes | partial | partial |
| ACCD | yes | no | no |
| Prescribed collider motion | yes | state-dependent | state-dependent |

The exact cells should be filled from tests, but the important idea is to make differentiation a **feature matrix**, not a global adjective.

---

# 9. Muscle-tendon units

## 9.1 What is good

The MTU is an engineering Hill-type equilibrium model with:

- activation dynamics,
- active force-length,
- force-velocity,
- eccentric branch,
- passive fibre force,
- series tendon,
- implicit fibre equilibrium,
- route anchors on world / rigid / tissue points.

This is a reasonable complexity level for the current project. It is better to use a small interpretable model with validated parameters than a more “physiological” model with many invented constants.

Relevant references:

- Matthew Millard, Thomas Uchida, Ajay Seth, Scott Delp, **Flexing Computational Muscle: Modeling and Simulation of Musculotendon Dynamics**, 2013  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3705831/
- Hartmut Geyer, Hugh Herr, **A Muscle-Reflex Model that Encodes Principles of Legged Mechanics Produces Human Walking Dynamics and Muscle Activities**, 2010  
  https://pubmed.ncbi.nlm.nih.gov/20378480/

## 9.2 Add a fibre-equilibrium residual diagnostic

The current fibre solve uses a fixed number of Newton iterations.

Even if six iterations are sufficient in all tested cases, the solver should expose:

```text
max_mtu_equilibrium_residual
n_mtu_nonconverged
```

and optionally a safeguarded fallback.

Because this is a 1D root, a bracketed Newton/bisection fallback is very cheap relative to the rest of the substep.

The important requirement is:

**failure to solve the muscle equilibrium should never be silent.**

---

# 10. Rigid and articulated coupling

The architecture is convincing:

- free bodies are solved as local 6×6 blocks;
- fixed links share the same attachment representation but have no dynamic block;
- articulated systems use coordinate descent;
- the rigid mass matrix is frozen for a substep;
- forward kinematics is refreshed after coordinate changes;
- contact and MTU terms participate in the same local rigid/coordinate solve.

This is aligned with AVBD’s goal of treating rigid, articulated, stiff, and soft systems in one local-iteration framework.

## 10.1 Low-hanging performance improvement: link -> attachment CSR

Some free-body attachment operations still conceptually follow:

```text
for each link:
    scan all attachments:
        if attachment belongs to link:
            ...
```

The codebase already uses CSR incidence extensively.

Add:

```text
link_attachment_offset
link_attachment
```

to reduce this to the actual incident attachments.

## 10.2 Sparse articulation incidence

The articulated path stores / uses a dense notion of whether each DOF affects each attachment.

For a long anatomical chain this relationship is sparse: an attachment is affected only by the ancestor DOFs between its link and the root.

Prefer either:

```text
DOF -> affected attachment CSR
```

or:

```text
attachment -> ancestor DOFs
```

over a dense `n_dof × n_attachment` table.

This will matter when the vertebral chain becomes large.

---

# 11. Contact system

The current contact system is significantly more sophisticated than a simple penalty implementation.

It includes:

- tetrahedral boundary extraction,
- shell triangles,
- rigid collision meshes,
- collision groups / rules,
- point-triangle pairs,
- edge-edge pairs,
- swept broad phase,
- spatial hashing,
- signed face-side semantics for point-triangle contact,
- smoothed Coulomb friction,
- rigid reaction accumulation,
- candidate reuse machinery,
- per-environment diagnostics,
- ACCD safety filtering.

This is a good foundation.

The main remaining issue is **continuous-contact completeness for codimensional geometry**.

---

# 12. Confirmed contact blind spot after `2a20f83`

The old edge-edge “crossing” test was removed because the sign of the edges’ common-perpendicular triple product does not identify which side of a physical surface the edges occupy. On articulated convex geometry it generated false positives under ordinary sliding.

Removing it was reasonable.

However, the replacement state is incomplete.

A pair of thin quad strips can cross edge-first such that:

- no vertex enters the opposite triangle interior,
- therefore signed point-triangle crossing never fires,
- yet the surfaces have passed through each other.

This matters especially for the **single-sided digestive shell**.

Important distinction:

**A swept broad phase does not make the narrow phase continuous.**

The broad phase can correctly identify an edge-edge candidate along the sweep, while the final-time unsigned distance response still fails to prevent an actual topological crossing.

## 12.1 P0 safety patch

Before inventing a new signed edge-side heuristic, add an explicit **swept edge-edge safety oracle** for shell / thin-surface pair classes.

Suggested behavior:

```text
if an edge-edge swept crossing is detected:
    if every participating dynamic object can be rescaled:
        use ACCD / safe-time response
    else:
        fail the environment loudly
        report:
            pair ids
            primitive ids
            groups
            TOI / conservative TOI
            substep
```

Even a fail-loud implementation is a major improvement over silent tunnelling.

---

# 13. ACCD review

The ACCD implementation is in the same conceptual family as the additive conservative-advancement method introduced for codimensional IPC.

Reference:

- Minchen Li, Danny M. Kaufman, Chenfanfu Jiang, **Codimensional Incremental Potential Contact**, SIGGRAPH 2021  
  https://ipc-sim.github.io/C-IPC/  
  https://ipc-sim.github.io/C-IPC/file/paper.pdf

The implementation removes common translation, derives a relative displacement bound, and advances conservatively toward a contact gap.

That part is reasonable.

## 13.1 Current ACCD is a safety clamp, not a complete contact integrator

Current semantics are approximately:

```text
solve full h
find conservative TOI t
rescale final pose to t
compute velocity over original h
advance simulation clock by h
```

If `t = 0.35`, the system consumes a full time step while only executing 35% of the solved displacement.

This is intentionally dissipative.

That is acceptable as a rare safety intervention, but if it fires often it changes the physical interpretation of the simulation.

## 13.2 Add lost-time diagnostics

Expose at least:

```text
ccd_min_toi
ccd_clamped_substeps
ccd_lost_time_fraction = sum(1 - toi) / n_substeps
```

An experiment that loses a substantial fraction of simulated time to CCD clamps should be flagged as numerically safe but physically suspect.

## 13.3 Long-term: split-step CCD

A more faithful integrator would:

```text
solve h
find t
commit t*h
resolve/contact
solve remaining (1-t)*h
```

with a bounded number of event splits per original substep.

This is more expensive, so it is a P2/R&D item rather than an immediate fix.

---

# 14. Candidate reuse: useful idea, but prove safety before enabling it

The `D_min / D_max` candidate reuse mechanism is inspired by continuous collision-handling work that amortizes collision searches while maintaining a safe proximity bound.

Reference:

- Tianyu Wang, Jiong Chen, Dongping Li, Xiaowei Liu, Huamin Wang, Kun Zhou, **Fast GPU-Based Two-Way Continuous Collision Handling**, 2022  
  https://arxiv.org/abs/2211.04045

At the reviewed baseline the reuse path is effectively disabled because:

```text
contact_margin_max == contact_margin
```

unless the application explicitly overrides it.

This is fortunate, because there is a proof obligation to settle before enabling it aggressively:

> Does the solver know **before** executing a reused-candidate substep that no omitted pair can enter the active contact layer during that substep?

The current budget is reduced from the motion that actually occurred. If this is only known after the substep, then the mechanism may tell us that the set is exhausted **after** a motion not covered by the previous set has already happened.

Do not call this a confirmed bug while the path is disabled.

Treat it as:

**a correctness proof/test requirement before setting `margin_max > margin` in production.**

---

# 15. Grid-cell tuning and search reach are still coupled in code

The local `useful_knowledge.md` correctly records the lesson:

> hash-grid cell size and search reach are separate tuning knobs.

However, the reviewed contact constructor still defaults to a cell size based on approximately:

```text
max(
    median_edge,
    2 * (max_thickness + margin_max)
)
```

Therefore increasing `margin_max` to enable reuse can automatically make the grid coarser.

That increases occupancy and can reintroduce `contact_cell_cap` pressure.

This is not inherently incorrect, but it couples two parameters that the design notes correctly argue should be independent.

## Recommendation

When enabling reuse, explicitly benchmark:

```text
margin_max = wider search horizon
cell_size  = mesh-scale value held fixed
```

Then choose the cell-size heuristic independently from candidate lifetime.

---

# 16. Contact-grid memory scaling

The current fixed-cap structure has a dominant memory term approximately:

```text
hash_buckets ~= 2 * N_contact_vertices

cell_v : int32
cell_c : int32[3]
```

Per occupied capacity slot:

```text
4 bytes + 12 bytes ~= 16 bytes
```

Therefore the allocated storage scales roughly as:

```text
M_grid ~= 2 * Ncv * C * B * 16
       ~= 32 * Ncv * C * B bytes
```

where:

- `Ncv` = contact vertices,
- `C` = per-bucket capacity,
- `B` = batch size.

This is the key reason the memory footprint grows so fast.

The owner-provided corrected measurement is:

- about **124.72 MB** at cap 256 for the measured single-env scene;
- approximately **4.7 GB** for the described `B=16` meal-mesh case.

## 16.1 Cheap partial reduction: pack the cell key

`cell_c` exists because an ordinary hash value does not uniquely identify the actual integer cell.

A packed 64-bit exact cell key could potentially reduce a slot from approximately:

```text
4-byte vertex id + 12-byte xyz cell = 16 bytes
```

to:

```text
4-byte vertex id + 8-byte packed cell key = 12 bytes
```

subject to Quadrants layout/alignment.

This is only a partial win, but potentially ~25% on that dominant slot storage.

## 16.2 Real fix: compact sparse cells

Long-term, replace:

```text
bucket_count × worst_case_capacity × B
```

with storage proportional to **actual insertions**.

For example:

```text
emit (env, cell_key, primitive_id)
sort / radix-sort by (env, cell_key)
compact
build:
    unique keys
    offsets
    primitive ids
```

Then memory becomes closer to:

```text
O(actual rasterized cell visits)
```

rather than:

```text
O(bucket_count * cap * batch)
```

For high batch counts and a meal mesh, this is likely the eventual architecture.

---

# 17. Offset Geometric Contact is worth a serious prototype

The current contact system can continue to be improved incrementally, but there is a recent method whose problem statement is unusually close to the one this project is encountering.

Reference:

- Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, Cem Yuksel, **Offset Geometric Contact**, SIGGRAPH 2025  
  https://graphics.cs.utah.edu/research/projects/ogc/  
  https://graphics.cs.utah.edu/research/projects/ogc/Offset_Geometric_Contact-SIGGRAPH2025.pdf

OGC specifically targets:

- codimensional objects,
- penetration-free simulation,
- large contact radii without the usual offset artifacts,
- vertex-specific displacement bounds,
- local massively-parallel operations,
- avoiding expensive global CCD,
- integration with VBD.

This maps directly onto the digestive-shell problem.

I would not replace the current contact implementation immediately.

I would prototype OGC on a deliberately small benchmark:

```text
single-sided curved shell
+
moving ellipsoid / thin opposing shell
+
large tangential sliding
+
edge-first crossing attempts
```

Compare:

- candidate count,
- memory,
- kernel time,
- maximum allowed substep,
- penetration / crossing,
- energy dissipation,
- batch scaling.

If OGC wins cleanly, it is a plausible “contact v2” direction.

---

# 18. AVBD packet / model layer

The packet layer is one of the strongest architectural decisions in the fork.

It gives the project:

- stable IDs,
- units,
- provenance,
- explicit ownership,
- tissue / rigid / route records,
- collision groups,
- contact materials,
- required capabilities,
- deterministic validation.

This is the correct direction for reproducible biomechanics.

## 18.1 Contract bug: `elastic_point_attachment` is advertised but rejected by the builder

`genesis.avbd.packet.CAPABILITIES` includes:

```text
elastic_point_attachment
```

but `genesis.avbd.model` explicitly rejects `elastic_point` attachments.

Therefore a packet may:

```text
validate_packet() -> PASS
```

and later:

```text
build_model() -> unsupported feature
```

This violates the intuitive meaning of `required_capabilities`.

### Fix

Either:

- remove `elastic_point_attachment` from advertised capabilities until implemented; or
- make the capability set builder-specific and only advertise features the active builder can honor.

## 18.2 `contact_penalty_ccd` is also too broad as a capability

CCD is controlled by runtime `VBDOptions`, and the current rescale implementation cannot simply handle every prescribed/articulated combination.

Therefore `contact_penalty_ccd` should not mean “the repository contains some CCD code.”

It should mean:

**this packet + runtime configuration can actually execute the requested contact semantics.**

Recommended approach:

- capability validation at static packet level;
- capability realization check at `build_model`;
- configuration-dependent capability check at `scene.build`.

## 18.3 Add one build test per advertised capability

For every public capability string:

```text
minimal valid packet
-> validate
-> build_model
-> scene.build
```

should succeed, or the capability should not be advertised.

---

# 19. Packet validation should include more semantic ranges

Current validation is already strong in topology and referential integrity:

- unique IDs,
- positive-oriented tets,
- exact outward boundary orientation,
- closed boundary,
- valid dynamic inertia,
- anchor weight sums,
- route references,
- required collision pair checks,
- finite arrays.

Add explicit physical range validation for quantities such as:

```text
E > 0
-1 < nu < 0.5
rho > 0

f_max > 0
l_opt > 0
l_slack >= 0
v_max > 0
0 <= activation0 <= 1

ligament stiffness >= 0
slack length >= 0

friction coefficients >= 0
normal compliance >= 0
restitution in supported range

quaternion norm ~= 1
```

The goal is to strengthen the meaning of:

```python
validate_packet(packet)
```

so a validated packet is not merely structurally parseable but physically admissible for the supported model.

---

# 20. Packet schema should learn about shells

The runtime now supports `VBD.Shell`, but the static packet is still tet-centric:

```text
Tissue:
    rest_positions
    tets
    boundary_faces
```

The digestive wall is exactly the subsystem most likely to need shell-native representation.

Suggested evolution:

```text
TissueGeometry:
    kind = "tet" | "shell"

tet:
    tets
    boundary_faces

shell:
    triangles
    mechanical_thickness
    bending law / parameter
    side semantics
```

For a single-sided digestive wall, **side/lumen semantics should be explicit metadata**, not something inferred indirectly from authored winding.

---

# 21. Testing architecture

The test suite is a major strength. In particular:

- independent mathematical references,
- direct force parity tests,
- finite-difference/adjoint checks,
- momentum checks,
- rest-validity tests,
- diagnostics tests,
- failure semantics,
- shell-specific tests,
- ACCD tests,
- contact regression tests.

The main testing issue found is a repository coupling.

## 21.1 Remove hardcoded sibling-repository oracle dependency

`tests/vbd/test_vbd_shell.py` imports the shell reference implementation from a sibling path similar to:

```text
../snakeSimWithAstra/spikes/verify_avbd_shell_math.py
```

This is excellent as an independent oracle during development, but a standalone Genesis fork should not require another repository to be cloned beside it with a particular directory name.

Copy the pure mathematical oracle into something like:

```text
tests/vbd/reference_shell.py
```

and record the original source commit/hash in a comment.

The test remains independent from the engine implementation while the repository becomes self-contained.

---

# 22. MPM and PBD changes

The MPM changes reviewed are small but well targeted:

- actuation gradients are explicitly cleared;
- one control input is copied to every substep frame that consumes it;
- input gradients sum across those frames;
- grid frame allocation reflects actual indexing;
- forward-only sparse reset does not pretend to work under gradients.

These are the right kind of “boring correctness” fixes.

The PBD physical attachment path also has a reasonable two-way solve based on the rigid mass matrix.

As more solvers gain custom coupling, avoid prematurely unifying their internal implementations.

Instead, unify their **contract tests**:

```text
momentum conservation
action/reaction symmetry
reset behavior
batch independence
prescribed-motion behavior
gradient semantics
failure semantics
```

---

# 23. Reproducibility recommendation: experiment manifests

This is slightly outside the Genesis fork itself, but it is important enough to record here.

SnakeSim is now dependent on:

- anatomy revision,
- Genesis fork commit,
- SnakeSim commit,
- solver options,
- contact parameters,
- mesh versions,
- material parameters,
- acceptance criteria.

Every serious run should emit a small versioned manifest such as:

```yaml
experiment:
  genesis_commit: 2a20f83e05430fd643ffcbdc2bb9a84655505b0c
  snakesim_commit: ...
  anatomy_revision: ...
  geometry_packet: ...
  backend: gpu
  float: f32
  vbd:
    dt: ...
    substeps: ...
    n_iterations: ...
    accumulate_f64: ...
  contact:
    margin: ...
    margin_max: ...
    cell_size: ...
    cell_cap: ...
    ccd: ...
  acceptance:
    rest_geometry: PASS
    gravity: PASS
    feeding: NOT_RUN
```

This is a very cheap way to prevent future archaeology.

---

# 24. Suggested diagnostics to expose

The solver already has useful diagnostics. The next useful set would be:

## Contact

```text
allocated_contact_grid_bytes
actual_cell_insertions
max_bucket_occupancy
mean_bucket_occupancy
hash_collision_rate
candidate_pt_count
candidate_ee_count
active_pt_count
active_ee_count
rebuild_count
d_budget
ccd_min_toi
ccd_clamped_substeps
ccd_lost_time_fraction
```

## VBD / constraints

```text
max_vertex_residual
relative_vertex_residual
max_distance_strain_violation
max_angle_violation
dual_updates
sweeps
max_constraint_k / k0
```

## Tissue

```text
min_J
inverted_tets
inversion_streak
```

## MTU

```text
max_equilibrium_residual
failed_roots
max_tension
min_fibre_length / l_opt
max_fibre_length / l_opt
```

These would make performance and physical-validation work much faster.

---

# 25. Low-hanging fruit

If the goal is maximum return per implementation day, the following are especially attractive.

## 25.1 Correct the documented contact-grid memory

Update the `VBDOptions.contact_cell_cap` documentation so it no longer says “31 MB” for the measured case.

Use the full `cell_v + cell_c` footprint.

## 25.2 Preserve the dev line

Create a permanent branch or tag for the VBD/AVBD development line, because it is not a trivial patch set anymore.

## 25.3 Add shell rigid-rotation test

Very small test, potentially large physical consequence.

## 25.4 Add fail-loud edge-edge swept crossing detection

Even before full response is implemented, prevent silent digestive-wall topology corruption.

## 25.5 Fix capability advertisements

Small code change; makes the AVBD interface truthful.

## 25.6 Add link-attachment CSR

Straightforward, low-risk performance improvement.

## 25.7 Add MTU root residual

Straightforward, high diagnostic value.

## 25.8 Make the shell oracle local to the repo

Straightforward CI/reproducibility improvement.

---

# 26. Medium-term engineering roadmap

A reasonable medium-term order would be:

### Phase A — correctness gates

1. curved-shell rigid-motion objectivity test;
2. shell EE swept crossing test;
3. capability contract tests;
4. MTU equilibrium residual;
5. candidate-reuse safety test.

### Phase B — instrumentation

1. contact memory accounting;
2. bucket occupancy;
3. actual rasterized insert count;
4. candidate-to-active ratio;
5. CCD lost-time metrics;
6. per-kernel isolated profiling protocol.

### Phase C — easy scaling

1. link->attachment CSR;
2. sparse articulation incidence;
3. decouple cell size from candidate horizon;
4. packed cell key if worthwhile.

### Phase D — architecture

1. shell packet representation;
2. compact sparse contact grid;
3. split-step CCD;
4. OGC prototype.

---

# 27. Open research questions

These are not bugs; they are questions worth making explicit.

## 27.1 What is the intended physical meaning of shell bending stiffness?

If the digestive wall obtains most of its effective bending resistance from:

- layered tissue,
- fibre architecture,
- muscle tone,
- incompressibility,
- folds,

then a separate numerical bending energy may only need to regularize the membrane.

If so, a simpler objective bending law may be enough.

If the bending coefficient is expected to represent measurable tissue mechanics, it needs a more direct calibration story.

## 27.2 How often is ACCD expected to fire in accepted runs?

If “almost never,” post-solve rescaling is a good safety mechanism.

If it fires routinely, the integrator should move toward split-step handling.

## 27.3 Is contact penalty stiffness a numerical parameter or a material parameter?

At present it is encoded through contact regularization and stiffness ramps.

For scientific interpretation, it should be clear which part controls:

- physical compliance,
- contact-layer resolution,
- solver conditioning.

## 27.4 What gradient semantics are required for training?

If RL only needs the forward solver, this is irrelevant.

If future optimization uses differentiable anatomy/contact, decide per feature whether the target is:

- derivative of the converged physical model; or
- derivative of the finite simulation algorithm.

The framework can support both, but callers should choose intentionally.

## 27.5 How much of the anatomy should remain rigid-link based versus tissue based?

The AVBD packet is already flexible enough to support mixed representations. A future model may benefit from using rigid links only where the skeleton truly behaves rigidly, while using soft or shell structures where the distinction is artificial.

---

# 28. Paper / reference reading list

## Core VBD / AVBD

### Vertex Block Descent
Anka He Chen, Ziheng Liu, Yin Yang, Cem Yuksel, SIGGRAPH 2024  
Project: https://graphics.cs.utah.edu/research/projects/vbd/  
Paper: https://graphics.cs.utah.edu/research/projects/vbd/vbd-siggraph2024.pdf

### Augmented Vertex Block Descent
Chris Giles, Elie Diaz, Cem Yuksel, SIGGRAPH 2025  
Project: https://graphics.cs.utah.edu/research/projects/avbd/  
Paper: https://graphics.cs.utah.edu/research/projects/avbd/Augmented_VBD-SIGGRAPH25.pdf

## Elasticity

### Stable Neo-Hookean Flesh Simulation
Breannan Smith, Fernando de Goes, Theodore Kim, 2018  
Paper: https://graphics.pixar.com/library/StableElasticity/paper.pdf

## Differentiable block solvers

### Differentiate the Solver, Not the Equation: Reverse-Sweep Adjoints for Block Implicit Simulation
Lei Shu et al., 2026  
https://arxiv.org/abs/2608.08559

## Contact

### Codimensional Incremental Potential Contact (C-IPC)
Minchen Li, Danny M. Kaufman, Chenfanfu Jiang, SIGGRAPH 2021  
Project: https://ipc-sim.github.io/C-IPC/  
Paper: https://ipc-sim.github.io/C-IPC/file/paper.pdf

### Fast GPU-Based Two-Way Continuous Collision Handling
Tianyu Wang et al., 2022  
https://arxiv.org/abs/2211.04045

### Offset Geometric Contact
Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, Cem Yuksel, SIGGRAPH 2025  
Project: https://graphics.cs.utah.edu/research/projects/ogc/  
Paper: https://graphics.cs.utah.edu/research/projects/ogc/Offset_Geometric_Contact-SIGGRAPH2025.pdf

## Shell bending

### A Quadratic Bending Model for Inextensible Surfaces
Miklós Bergou, Max Wardetzky, David Harmon, Denis Zorin, Eitan Grinspun, 2006  
Eurographics: https://diglib.eg.org/items/edfc6fd8-504e-4c0c-89d6-35579d1c8b39  
DOI: https://doi.org/10.2312/SGP/SGP06/227-230

## Muscle modeling

### Flexing Computational Muscle: Modeling and Simulation of Musculotendon Dynamics
Matthew Millard, Thomas Uchida, Ajay Seth, Scott Delp, 2013  
https://pmc.ncbi.nlm.nih.gov/articles/PMC3705831/

### A Muscle-Reflex Model that Encodes Principles of Legged Mechanics Produces Human Walking Dynamics and Muscle Activities
Hartmut Geyer, Hugh Herr, 2010  
https://pubmed.ncbi.nlm.nih.gov/20378480/

---

# 29. Final assessment

The fork has crossed the threshold from “application patches to Genesis” into a real simulation subsystem.

The strongest parts are:

- the VBD core decomposition;
- stable neo-Hookean integration;
- AL constraint machinery;
- explicit failure semantics;
- the two-mode differentiation architecture;
- the model-packet direction;
- the test culture.

The largest technical debt is concentrated rather than diffuse.

The two issues I would place the strongest red boxes around are:

1. **curved-rest shell bending objectivity**, and
2. **continuous contact safety / memory for the digestive shell.**

The first can be tested quickly.

The second is the actual hard research problem, and should be attacked as its own subsystem rather than by indefinitely adding special cases.

The current implementation is a good enough foundation that a contact v2 based on ideas such as OGC can be evaluated without rewriting the rest of the biomechanics stack.

---

# 30. Immediate checklist

- [ ] Tag or permanently branch the VBD development baseline
- [ ] Correct the 31 MB contact-grid documentation
- [ ] Add curved-rest rigid-rotation shell bending test
- [ ] Add edge-edge swept crossing safety test
- [ ] Add fail-loud path for unsupported CCD participants
- [ ] Fix `CAPABILITIES` / builder mismatch
- [ ] Add capability realization tests
- [ ] Add MTU Newton residual diagnostic
- [ ] Add link->attachment CSR
- [ ] Make articulation incidence sparse
- [ ] Decouple `contact_cell_size` from reuse horizon
- [ ] Add contact memory / occupancy metrics
- [ ] Validate candidate reuse before enabling `margin_max > margin`
- [ ] Move shell math oracle into this repository
- [ ] Add shell geometry to packet schema
- [ ] Prototype compact sparse contact cells
- [ ] Prototype OGC on a digestive-shell microbenchmark
- [ ] Add experiment manifests to application runs

---

*Prepared and committed by ChatGPT at the repository owner/user's explicit request and authorization.*
