# Notes

## Build log

Format — one line per entry:

`YYYY-MM-DD — what I tried, what broke, what fixed it`

2026-08-16 — checkpoints 1 and 2. Avoided one trap the tests set: several public
implementations rescale the linear endpoints by `1000 / T`, and at the `T = 10`
case `test_betas_are_valid_variances` parametrises that puts `beta_end` at 2.0.
The paper's constants are fixed, not relative — no rescaling. Derived
`alpha_bars` once per `(kind, timesteps, device, dtype)` in `_derived` rather
than per call, and kept the derivation off the caller's dtype before casting.
Forward-process strip looks right — front-loaded dissolve, pure static by the
last column.

2026-08-16 — adversarial review of the above, and the corrections it forced.
The float16 justification I wrote for `_derived` was wrong: I claimed a
thousand-term cumprod underflows to zero in float16, and for the linear
schedule it does not — it lands at 4.0e-05 against a smallest subnormal of
6.0e-08. The claim is true for cosine (2.4e-09, flushes to zero), which is the
honest version, and the fix is the same either way: take the square roots
before the cast, since 4.9e-05 is representable where its square is not.
Also wrong in the same sentence — AMP never triggers the path at all, because
autocast leaves the input tensors float32. Derivation moved to float64 while
fixing the wording, which also stops float64 callers silently getting float32
accuracy.

Real bugs the review turned up, none of them in the maths:
- `torch.linspace` and `torch.arange` inherit `torch.set_default_device`, so the
  schedule silently left the CPU, and the float64 cosine branch died outright on
  MPS. Both pinned to `device="cpu"`.
- The `.to()` in `_derived` is a no-op when dtype and device already match, so
  the cached `betas` *was* the `_BETA_CACHE` tensor. One in-place edit anywhere
  would have poisoned both caches process-wide. Cloned at insert.
- `--timesteps 200` used to raise `NotImplementedError`; after checkpoint 2 it
  trained happily on a model that never saw more than 34% noise and then handed
  it to a sampler starting from pure noise. Converting a loud failure into a
  silent one is worse than not implementing the thing. `train.py` now refuses.
- `viz.save_forward_process_grid(total_timesteps=...)` picked columns from one
  schedule and noised them with another, producing a last column at correlation
  0.78 — precisely the bug signature its own docstring advertises catching.
  Now rejected.
- Negative `t` wrapped: `t = -1` was bit-identical to `t = 999`. Integer `x0`
  truncated every coefficient to zero and returned silence. Both now raise.

Worth recording separately: `tests/test_q_sample.py` is behavioural by design,
and mutation testing shows how much that costs. `mean = alpha_bar` instead of
`sqrt(alpha_bar)`, a `+1` index shift, and running `q_sample` on the cosine
schedule while everything else uses linear all pass its thirteen tests. The
green suite was never the evidence I treated it as. `tests/test_derived.py`
pins the closed form and the schedule source; all three mutants die there.

2026-08-16 — checkpoints 3 and 4. `p_losses` is four lines and went in without
incident. `p_sample_loop` needed `alpha_bars_prev` and `posterior_variance`,
which went into `_Derived` rather than being rolled per step — the buffers are
shared process-wide, and `ab_prev = alpha_bars.roll(1); ab_prev[0] = 1.0` is one
missing `clone` away from corrupting `q_sample` for the rest of the run. Took
the exact posterior variance over `beta_t`: it falls to zero at `t = 0`, so the
no-noise-on-the-last-step carve-out comes out of the algebra instead of being
bolted on. Kept the explicit branch anyway, so a draw that gets multiplied by
zero cannot advance the RNG and make reproducibility depend on it.

Two things that actually bit:

The unfolded-posterior cross-check failed first time against `tiny_model`, at a
max deviation of 1.04e-02 — but the untrained chain drifts into the hundreds
over a thousand steps, so that is float32 rounding between two algebraically
identical routes, not a bug. It also made the test useless for its real purpose:
the carve-out it exists to catch adds noise of exactly the same 1e-02 magnitude.
Switched it to the `GaussianOracle`, which holds the chain at unit scale; the
two routes then agree to 1e-06 and the carve-out bug stands out by four orders
of magnitude.

Samples from an 800-step overfit run had the right structure but sat at mean
-1.46 against training data at -0.01, which looked like a recursion bug. It is
not — the oracle test says the recursion is exact. Training longer and measuring
the model's mean eps-prediction bias settles it: bias -0.231 -> -0.0117 ->
-0.0047 -> -0.0023 at 0/1000/2000/4000 steps, and sample mean tracks it
+38.6 -> +0.18 -> -0.062 -> -0.046, converging on the data's -0.008 with std
0.392 against 0.415. A small constant bias in the noise prediction compounds
over a thousand reverse steps. Worth remembering before blaming the sampler.

Mutation testing on the new code, eight injected bugs, all caught — but three
only by `test_derived.py`: the missing carve-out, the loop running forwards, and
`model(x_t, t)` replaced by `model(x_t, zeros)`. That last one is the
uncomfortable one, because `test_gradients_reach_the_timestep_embedding` is
written to catch exactly it and does not: feeding constant zeros still sends
gradient through the embedding.

2026-08-16 — adversarial review of checkpoints 2, 3 and 4. The maths came
through clean on all three: the forward marginal matches Ho eq. 4 to 1.7e-08
against both a hand derivation and a brute-force chain, the objective is Eq. 14
up to the documented 1/D from using a mean instead of a sum, and the reverse
recursion was confirmed against sympy, an independent reimplementation from the
paper, and two known-answer tests on distributions the repo never uses. The
defects were all in the scaffolding I had added around the maths, and the two
worst were self-inflicted by the previous round.

The bad one: the device guard I added to `p_sample_loop` broke the entire GPU
path. `torch.device("mps") != torch.device("mps", 0)`, and `pick_device("auto")`
returns the unindexed form while `.to()` reports the indexed one — so the guard
rejected exactly the device the caller had got right. `python train.py` on this
machine died at the first sample write and `sample.py --device mps` never ran;
only `--device cpu` worked. My own end-to-end check missed it because I had
pinned `device="cpu"` in the synthetic run. A guard that turns a clear failure
into a broken default is worse than no guard; canonicalise with
`torch.empty(0, device=d).device` before comparing.

The worse one: `test_buffers_are_not_aliased_to_the_beta_cache` restored the
shared buffer with `add_(1.0)` then `sub_(1.0)`, which does not round-trip in
float32 — 993 of 1000 betas came back perturbed, `beta[0]` by 2.5e-04 relative.
Because `_derived` hands every caller the same tensors, that poisoned the
schedule for the rest of the process, and the poisoned buffer was then read by
the one test guarding the recursion's coefficients. Measured: a sampler that
skips its final step deviates 2.04e-04 on a clean cache and fails, and 9.06e-05
on the poisoned one and passes. Run as a full suite, the recursion had zero
effective coverage. Restore with `copy_`.

Also fixed: `default_betas` built its cache key before validating, so the same
bad argument raised on a cold cache and silently returned a 1-step schedule on a
warm one; `q_sample` accepted a length-1 `t` against a longer batch and noised
every image to the same timestep; `noise` of a different dtype silently changed
the return dtype; a bool `t` became mask indexing; `p_losses` returned nan on an
empty batch and accepted any broadcast-compatible model output, scoring a
plausible ~1.0 for a model returning a scalar. The range check now does one
device sync instead of two — the pair cost 10.6x on MPS for small tensors.

Two documentation corrections. The `_derived` docstring said the float64
ordering rescues the cosine schedule in float16 and quoted 1.5e-05; the real
figure is 4.93e-05, and the rescue only covers the precomputed square roots —
the `alpha_bars` field itself is still exactly zero, because no ordering makes
an unrepresentable number representable. And `test_sampling.py` claimed its
Gaussian oracle catches swapped alpha terms and off-by-one loop bounds "by a
factor of a hundred or more"; measured, it catches none of seven such mutants,
all landing within its tolerance. N(0,I) is the worst possible probe, since the
paper says `beta_t` is optimal for exactly that distribution and this code uses
`beta-tilde`. I had repeated that claim in the README. Replaced with a
point-mass oracle, which the wrong coefficients miss by orders of magnitude.

The highest-value single edit in the round was one line: the `t` in the
`p_losses` pin was a constant vector, `full((B,), 300)`. Making it varied killed
five real mutants at once — conditioning replaced by a constant, `t` permuted
across the batch, and three clampings that leave part of the schedule untrained.
Every one of them had passed all 98 tests.

