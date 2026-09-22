# Useful knowledge

Lessons from wrong assumptions, recorded as they were found.

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
