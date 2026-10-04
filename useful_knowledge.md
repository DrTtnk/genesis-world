# Useful knowledge

Lessons from wrong assumptions, recorded as they were found.

## Test suite time (2026-09-30)

- **The slow VBD tests were compile-bound, not physics-bound.** I assumed that the 541 s adjoint test was slow
  because of its finite-difference rollouts. Its rig has 8 vertices. Measured alone, its physics was a small part.
  Three kernels unrolled `for sweep in qd.static(range(n_iterations))`: `kernel_sweeps_articulation`,
  `_kernel_primal_sweeps` and `_kernel_adjoint_sweeps`. Each inlined copy of the sweep body costs about 7 s of
  Quadrants front end, and LLVM then compiles the large result. The fix is the same as in `_kernel_sweeps`:
  one sweep per kernel, with the sweep index as a runtime argument and a Python loop. Results stay bitwise
  equal. The adjoint test went from 394 s to 38 s alone. Do not unroll a sweep loop with `qd.static`.
- **Measure the front end per kernel before you optimise a slow test.** cProfile makes the Quadrants front end
  look about 3 times slower than it is. Time `quadrants.lang.kernel.Kernel.materialize` for each kernel on its
  first call instead. In the hinge MTU test one kernel took 28.7 s of the 30.3 s front end.
- **A new worktree or a changed kernel source misses the offline cache.** The first run then includes the LLVM
  compile of every changed kernel. Compare test times only between runs with the same cache state.

## Merging the perf branch under the WIP (2026-09-30)

- **Check the function list after a scripted conflict resolution.** I replaced a function by slicing from its
  `def` to the next blank-line pair after a line I searched for. The slice also took the next function,
  `func_contact_dof_terms`, and every test then failed at `import genesis`. After a merge, compare the `def`
  names of HEAD, the stash and the result; each name that is missing needs a reason. Here
  `func_sweep_overlaps` was missing too, but for a good reason: the WIP removed its only caller.

## Rigid colouring (2026-09-30)

- **Do not unroll a loop over colours with `qd.static` when the body is a whole block solve.** I did this first
  for the free-body colours (cap 8). Four small tests then took 517 s. Launch one kernel for each colour from
  Python, the same fix as for the sweeps.
- **Quadrants refuses an alias of a solver object inside kernel scope.** `contact = solver.contact` and
  `colouring = self.rigid_colouring` fail with "Invalid constant scalar data type". Pass the object as a
  `qd.template()` argument, or write the full attribute path. `qd.static(x is not None)` fails too ("Operator
  "is not" ... not supported"): give the object a `has_...` property, as `has_joint` does.
- **Kernels cannot capture a Genesis solver or its fields by closure.** The fields are ndarrays ("Ndarray ...
  used in kernel scope but not registered"). Pass them as template arguments.
- **A sum in the same order is not always bitwise the same.** The entry pass stores each contact term and sums
  it later, and the serial block sums it where it computes it. The blocks differ by one unit in the last place,
  most probably because fast math fuses the last multiply into the add only in the serial path. Compare such
  paths with an absolute tolerance scaled to the largest element (1e-14), and keep a whole-trajectory test
  where both runs use the same code.

## VBD contact search / margin (2026-09-21, GPU perf task on vbd_contact.py)

- **Never scale a search reach by a multiple of a user-chosen `margin`.** `margin` can already be sized
  generously by whoever set it (a test used 25 mm on purpose). A blind `4 * margin` default for a wider rebuild
  reach (`margin_max`) reached far enough to touch the far side of a 200 mm plate and added 218 spurious
  edge-edge candidates that had nothing to do with the approach being tracked, over-constraining a block that
  should have sunk 1.3 mm and instead stopped at 0.1 mm. The safe default for an opt-in "search wider than you
  strictly need" knob is *no widening at all* (`margin_max = margin`), with a comment telling the caller to size
  it against their own scene's gaps if they want the wider reach.

- **A hash-grid cell size and a search reach are two different knobs and must not be tied to the same value.**
  Sizing the grid cell off the wider `margin_max` (so a rebuild's search box lines up with the cell) coarsened
  the grid and packed far more contact vertices into one hash bucket for the same geometry, overflowing
  `contact_cell_cap` on scenes that never touched `margin_max` at all. The grid only has to be fine enough that
  two primitives within one `margin` (not `margin_max`) of each other share a canonical cell; a rebuild that
  wants to search further just walks more of the smaller cells, which costs cycles, not correctness.

- **AVBD-style dual state (a pair's `lam` and its stiffness ramp `k`) is scoped to one substep, not to how long a
  candidate pair happens to stay in the candidate list.** It used to reset implicitly, because every substep
  rebuilt the whole candidate list from scratch and insertion always zeroed it. The moment candidate pairs can
  be *reused* across substeps without rebuilding, that implicit reset stops firing for the reused pairs, and `k`
  keeps ramping substep after substep toward its cap instead of restarting — this produced both a NaN contact
  distance (degenerate geometry from an over-driven pair) and a block that could not be pushed across a plate by
  friction it should have shrugged off. Fix: reset `lam`/`k` for every *currently live* pair every substep,
  unconditionally, in a pass separate from (and in addition to) the one that resets them at insertion time.

- **Never run a "before vs after" GPU kernel-profiler comparison concurrently with an unrelated CPU-heavy
  process (e.g. a 9-worker pytest run), even if the two don't share the GPU.** Host-side contention (process
  scheduling, memory bandwidth) inflated unrelated kernel times by 2-3x in one measured run — kernels neither
  change touched came back 3x slower than the same kernels measured alone. Always profile on an otherwise idle
  machine, and note when a comparison run was contended so the numbers are not trusted.

- **When editing a `@qd.kernel`/`@qd.func` body, do not run a pytest suite that imports and JIT-compiles that
  same module in the background at the same time.** A freshly spawned xdist worker importing a half-edited file
  segfaulted the whole run partway through. Let a background test run finish (or kill it) before touching the
  file it is testing again.

- **`~expr` on a positive Python `int` gives the correct two's-complement bit pattern for "every bit but this
  one" and is safe to embed as a kernel literal; `0xFFFFFFFF & ~expr` is not** — the masked value can exceed the
  literal range of a kernel's default `i32` and raises `QuadrantsTypeError: Integer literal ... exceeded the
  range of default_ip`.

- **A top-level `ndrange` nested inside runtime control flow is not parallelised; it runs on one thread.**
  `func_contact_dual_update` iterates `ndrange(contact.pair_cap, B)` and was called under
  `if sweep < self._n_iterations - 1:` inside `_kernel_sweeps`. Quadrants parallelises only top-level loops, so
  the whole 65,536-slot pair capacity was walked serially on eleven of every twelve sweeps. Measured on a scene
  with 248 actual pairs: 64.42 ms a substep at the default capacity, exactly linear in `contact_pair_cap`
  (1024 to 4.00 ms, 4096 to 6.88, 16384 to 18.40), and 2.51 ms flat once the guard was carried into the loop
  body instead. The profiler's own `min 0.004 / avg 5.164 / max 5.719 ms` gave the diagnosis away: the minimum
  is the one sweep a substep where the condition is false and the loop is skipped. Carry such a condition into
  the loop as a value; never wrap the loop in it. A static `qd.static(...)` guard is fine, because it is
  resolved at compile time and the loop stays top-level.

- **A profile line's name is not its cause.** The 5.25 ms serial kernel in the head36 profile was reported as
  the free-body Gauss-Seidel loop in both the performance plan and the review reply. A shell-and-one-collider
  scene with *zero* free bodies pays the same 5.16 ms, which is what showed the label was guessed rather than
  read. Attribute a kernel by removing the suspected work and re-measuring, not by matching the name to the
  nearest plausible loop in the source.

## A `maxvolume` above the shape's own volume refines nothing

`gs.morphs.Box(size=(0.01, 0.01, 0.01), maxvolume=4e-7)` produces **6 tetrahedra and 8 vertices**: the box is
1e-6 m^3, tetgen's six default tets are already 1.67e-7 each, and the constraint is satisfied without a single
refinement. The entity then has no interior vertex at all, so a test that attaches four vertices and pins four
more is exercising a rigid cube, not tissue.

`tests/vbd/test_vbd_attachment_stiffness.py` uses that value, so its "tiny tissue block" is those six tets, and
its 0.062 mm result says less about the attachment penalty than it appears to. `maxvolume=5e-9` on the same box
gives 383 tets.

Check the tetrahedralisation log (`Mesh tetrahedra:`) rather than trusting the parameter.

## The head36 attachment failure is not the per-sweep dual update

`func_update_attachment_dual` advances the augmented-Lagrangian multiplier *and* the stiffness ramp once per
sweep on the forward path, which looked like an outer-loop step wired as an inner one -- especially beside the
gradient path's own comment that "a dual update on an unconverged iterate overshoots at the stiffness cap and
limit-cycles instead of converging". Astra's head36 traces fit it: the attachments sitting at the stiffness cap
rose 259 -> 322 -> 341 of 474 as sweeps went 12 -> 48 -> 192.

Measured, it is not the cause. `tests/vbd/test_vbd_attachment_convergence.py` hangs a pinned-tissue, free-bone,
tissue, free-bone chain at head36's 0.25 ms substep for 100 ms and reports the peak raw attachment gap:

    CPU float64   12 sweeps 0.0034 mm | 48 sweeps 0.0001 mm | 192 sweeps 0.0000 mm
    GPU float32   12 sweeps 0.0066 mm | 48 sweeps 0.0001 mm | 192 sweeps 0.0000 mm

More sweeps converges, monotonically, on both backends. A plausible mechanism plus a matching trend in the
failing scene is not evidence: the trend has to be reproduced in isolation before the mechanism is believed.

## VBD tet vertex mass is uniform per entity, not lumped per element

`vbd_entity.py` sets `mass = rho * total_rest_volume / n_vertices` and hands that one scalar
to `_kernel_add_elements`, so every vertex of a tet entity carries the same mass. It is not
the usual `rho * V_tet / 4` lumped onto each corner.

Reconstructing it the lumped way still reproduces `_k_start` exactly, because both schemes
give an entity the same total mass and `_k_start` is a mean over all vertices. The agreement
is therefore not evidence that the reconstruction is right. Only the extremes differ, and
they differ a lot: on head36 the lumped reconstruction reports a lightest vertex of
2.46e-12 kg and a spread of 190,987 to 1, while the engine's actual formula gives 4.37e-08 kg
and 461 to 1.

Check a reconstruction against a statistic that the two candidate formulas disagree about,
not one they share.

## A per-vertex mass is a meshing number, not a physical one

Because the mass is `rho * V / n_vertices`, two objects of identical density differ in vertex
mass by whatever their resolutions differ by. On head36, with `rho = 1050` everywhere:

    fascia27.geniohyoideus...  232 verts, 342 tets, V 9.66e-9 m^3  ->  4.37e-08 kg a vertex
    palatine_maxilla_tie.R       6 verts,   4 tets, V 1.15e-7 m^3  ->  2.01e-05 kg a vertex

461 to 1, entirely from how finely each was meshed. Anything scaled off a mean vertex mass -
`_k_start` is - therefore inherits the mesh resolution of whichever population dominates the
vertex count. On head36 that is the fine fascia, which carry no attachments at all, while
every one of the 262 attachment-carrying tissues has exactly 6 vertices and is among the
heaviest per vertex in the model.

## An augmented Lagrangian's per-substep gain does not depend on how often the dual is updated

On head36 the attachment multipliers diverge: their sum reaches 311.90 N against 0.0432 N
of tissue weight, a factor of 7,200, growing by a median 1.0783 a substep against an
`alpha * gamma` decay of 0.9405, so a net 1.0142 compounding over 400 substeps.

`func_update_attachment_dual` runs once a sweep, which looked like the cause: twelve dual
updates a substep where the gradient path does exact Uzawa with one. Moving it to one a
substep changed the gain from 1.07833 to 1.07852 - by 0.02 percent. Each update simply
becomes about twelve times larger, because the error it sees has not been relieved by the
intermediate corrections. The gain is set by the error over a substep, and the error adapts
to the cadence.

The change is not neutral, it just does not address the divergence: inversion records fall
from 407 of 866 to 16, the worst tetrahedron from J = -39.73 to -0.571, and attachments at
the stiffness cap from 257 to 39, while attachments over 1 mm rise from 38 to 243 and the
peak gap only falls from 12.208 mm to 7.739 mm. Catastrophic failures are traded for
uniform compliance.

If a dual scheme diverges, measure the gain per outer step before changing the cadence.
The cadence redistributes the same total.

A new rod pose test caught an omitted `morph.pos` translation. Rod sampling must add the requested translation before the existing VBD mesh-centroid rotation/offset transform. Segment frames use the same composed rotation.

A translated/rotated rest rod exposed an optimizer termination bug: Armijo backtracking demanded a decrease from a step below floating-point coordinate precision. The block solver now terminates those converged blocks before line search; eight random rotated/rest cases exercise this boundary.

- Distinguish a closest-contact gap estimate from its conservative dual lower bound. They are different observables; carry both explicitly rather than silently changing the meaning of the distance returned to callers. Run engine tests from genesis-fork so its pytest options are available.

- Rod/tissue and Hill coupling: 12 sweeps leave 10.69 um attachment gap and 0.220 um centre-of-mass drift in the new one-step gates. Check iteration convergence before accepting the coupled reference; do not relax the 10 um/0.1 um assertions.

- Engine pytest defaults CUDA to FP32 while CPU uses FP64. Rod reference CUDA validation must explicitly request precision64; backend selection alone does not preserve precision.

- Native scene build left a warm constraint state in the one-sweep fixture (nonzero multiplier and ramped stiffness). A dual-update oracle must include the existing warm-start decay; assuming initial multipliers zero caused the small post-fix mismatch. Keep the strict accepted-geometry tolerance.

## Rod block stopping metrics

The Cayley angular step has quaternion tangent norm half its angular norm. A machine-precision stopping test must use that stored coordinate metric. Coupled radius nodes use their vector norm. Head40/41 captured rotation and scale rest cases reproduced Armijo failures before these changes; energy and admissibility checks stay unchanged.

- Genesis test CLI has no --precision option. The precision fixture defaults to float64 for CPU correctness tests; use the precision marker for explicit GPU float64 tests.

- Quadrants can compile a variable read after an if/else only when it has an unconditional initial value. In the rigid contact owner gate, initialize the four-vertex vector before the point-triangle/edge-edge branch. Also avoid reusing one local for pair structs of different types across the branches.
- In this checkout, `pytest` is not on the base shell PATH. Use `/home/zefiro/miniforge3/envs/snakesim/bin/python -m pytest` with `PYTHONPATH=$PWD` to run Genesis tests against this sibling checkout.
- A box-on-plate contact scene can generate point-triangle pairs only with the falling bone as the point and the fixed plate as the triangle. Do not assume the search also emits the reverse point-triangle orientation; inspect live pair ownership before writing a coverage assertion. Edge-edge pairs in this scene contain two rigid vertices on each link.
- Moving a pure ownership branch ahead of pair geometry preserved the physical wrench, but the Quadrants compiler changed a few last bits of the matrix arithmetic. An actual-engine oracle measured a 2.19e-17 force difference and a 1.73e-9 Hessian-entry difference; use scale-aware numerical equivalence rather than bitwise equality for this compiled path.

- The old `/home/zefiro/miniconda3/envs/snakesim/bin/python` path is stale on this host. The current test interpreter is `/home/zefiro/miniforge3/envs/snakesim/bin/python`; checking the environment path before a test avoids a false execution failure.
- In a batched Genesis scene, `RigidEntity.get_vel()` returns `(B, 3)`. NumPy `assert_allclose` requires the expected array to have the same shape; use `np.broadcast_to(expected_vec, actual.shape)` when checking all environments.
- The 2026-09-27 stage-0 ownership gate covers 19 focused CPU tests: standalone rigid-only integration, analytical spring balance, two-environment reset, existing rigid coupling, and existing VBD joints. It is not a full engine suite or head-contact acceptance. Full log: `../snakeSimWithAstra/out/unified_avbd_implementation/vbd_owner_nearest_cpu.log`.

## Frozen contact and complete edge candidates (2026-09-27)

- A VBD scene can own free rigid bodies without tissue, but contact construction still allocated `vertex_cv` with a zero shape. Quadrants rejects zero dimensions. Keep one storage cell when `n_vertices == 0`, skip the empty copy, and leave the logical tissue loops at zero; do not add dummy tissue.
- A point-triangle vertex hash is not a complete edge-edge broadphase. Two long perpendicular edges can be closer than the contact layer at their interiors while neither endpoint enters the other's swept search box. A real 60 mm crossed-bar fixture had a 0.13 mm positive gap under a 0.2 mm layer, zero PT pairs and zero EE pairs before the fix. Hash each swept edge AABB for EE, then accept only the canonical shared cell and exact hash-cell match. The PT and EE searches run sequentially, so the same hash buckets can be cleared and reused instead of allocating a second grid; I initially missed that storage reuse.
- Candidate pair counts or tiny final reactions do not prove an active contact law. A proposed angular-momentum fixture counted pairs but produced at most tens of micronewtons after a spin; that mostly measured free integration. Require a resolved contact load or measured impulse before interpreting a conservation or refinement result. The symbolic fixed-normal torque defect remains a separate proof, not a native physical acceptance gate.

- The new edge insertion initially applied `contact_sweep_cell_cap` to the whole edge AABB. That was the wrong owner for the limit: it historically bounds a vertex's *travel*, while PT triangle and EE query boxes already traverse full static primitive extents. Six contact regressions with a large fixed table failed on this new check. Remove only the edge-extent check; retain the moving-vertex cap and bucket/pair overflow. The six failed cases then passed, and dedicated tests prove both static >512-cell edges and moving-vertex overflow.

- A swept-contact regression assumed that predicted-path deviation above the margin was the rebuild trigger. That is stale: `max_tissue_deviation` is diagnostic, while raw maximum vertex motion spends `d_budget`. With the complete EE grid, the native impact deviates 0.832 mm under a 1 mm margin but still spends its budget and rebuilds on the next step (counts 1 then 2, no environment failure). Test the budget and rebuild contract, not one response trajectory.
- A cell-size invariance test compared candidate sets even when its 0.5 mm grid overflowed a 256-entry hash bucket: max occupancy was 433, errno 4224 included the fatal cell-overflow bit 128, and 14 EE pairs were absent. The helper silently clipped pair counts and ignored `env_status`. Require a healthy environment before comparing sets and choose an explicit capacity for the test's smallest cell; retain a separate real overflow test that expects failure. This is a capacity limit, not evidence that a valid grid changes physics.

### 2026-09-27: standalone VBD reset must use solver activity

A solver with zero owned entities can still be active: VBD may own free rigid links. `Simulator.reset` used `n_entities > 0`, so it skipped VBD state restoration, including the reset after the build warmup. The rigid residual test exposed this as a stale completed-solve marker. Use `solver.is_active` to select reset targets, as stepping does; test before-first-step refusal and reset replay on a coupled standalone scene.

That reset also exposed a logical-size error: `kernel_set_vertex_state` iterated the minimum-one allocation size and read from an empty logical vertex snapshot. This caused a native segfault. Its loop must use `pos.shape[1]`, not `vertices.shape[1]`; padding is storage, not an extra physical vertex.

- 2026-09-27 static VBD LDLT gate: the shell default `/home/zefiro/miniforge3/bin/python` has no pytest; use `/home/zefiro/miniforge3/envs/snakesim/bin/python` for engine tests. Quadrants `qd.static(range(row + 1, 6))` rejects `row = 5 - row_offset` because that assignment yields an Expr even when `row_offset` comes from a static loop. Iterate `qd.static(range(5, -1, -1))` directly to retain the 5-to-0 back-substitution order and make nested bounds compile-time constants. The existing 24-case SPD LDLT Torch oracle then passed on CPU and FP64 GPU.

- 2026-09-27 snapshot regression fixture: VBD has no `Elastic` material class. Its supported tet material is `VBD.Muscle` (zero actuation gives the passive solid). Check the material exports before copying an API name from another solver. The first shape-test attempt failed during fixture construction, so it is not evidence of the intended RED condition.

- 2026-09-27 snapshot boundary: copying the logical input length fixes zero-vertex VBD reset, but alone accepts a truncated nonempty snapshot and leaves live vertices stale. The real-scene RED check confirmed this for one and two environments. Validate both full `(B, n_vertices, 3)` shapes in `VBDSolver.set_state` before restoring rods or copying any fields.
