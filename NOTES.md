# Notes

## Build log

Format — one line per entry:

`YYYY-MM-DD — what I tried, what broke, what fixed it`

2026-08-16 — checkpoints 1 and 2. Avoided one trap the tests set: several public
implementations rescale the linear endpoints by `1000 / T`, and at the `T = 10`
case `test_betas_are_valid_variances` parametrises that puts `beta_end` at 2.0.
The paper's constants are fixed, not relative — no rescaling. Derived
`alpha_bars` once per `(kind, timesteps, device, dtype)` in `_derived` rather
than per call, and kept the derivation in float32 on the CPU before casting: a
thousand-term cumprod underflows to zero in float16, which would have shown up
only once AMP was switched on. Forward-process strip looks right — front-loaded
dissolve, pure static by the last column.

