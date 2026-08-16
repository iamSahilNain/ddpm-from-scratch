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

