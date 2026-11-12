# GPU prefill overhead across token counts

The GPU placement table charges overhead as `fixed_us + ntokens * row_us`
once when a batched layer has GPU work. A one-token/vector pass does not use
this charge. The search, candidate ordering, route thresholds and tie-break
remain the existing placement policy.

Begin/finish operations include launch, synchronization and transfer setup as
well as row-dependent work. A single observed token count cannot identify both
terms. For that case, the row slope retains the lane's transfer prior:

```
row_us = (lane.token_bytes + lane.output_bytes) / 1e6 * link_us_per_mb
fixed_us = mean_observed_us - mean_tokens * row_us
```

Both terms must be nonnegative. If the prior alone exceeds the complete
observation, it is capped at `mean_observed_us / mean_tokens` and the fixed
term is zero. With two distinct observed token counts, a nonnegative
least-squares fit estimates both terms. An empty sample window retains the
previous estimate.

The byte geometry comes from `kimi_vram_prefill_lane` in
`engine/ggml/src/ggml-ssd/gpu/vram-tier-api.h`. The grouped engine reports
both input and output row bytes; a lane that reports `token_bytes=0` (as the
retired MXFP4 hand engine did, charging its input per route) has a row prior of
its output bytes alone. The existing observed copy rate, or existing configured link prior
when no rate has been observed, supplies the time per MB. Applying that rate
to both directions is a physical prior, not a separately measured D2H rate.
An explicit `place.costs` value for `gpu_token_us` supplies the row prior instead.
There is no new setting or assumed launch-time constant.

The recorded 2026-10-24 RTX 5090 sample had 2 tokens and 279.656 µs overhead.
Its MXFP4 output row (the hand engine of then) was 20,480 bytes and its measured copy rate was
33.617 GB/s. This produces 278.438 µs fixed plus 0.609216 µs per row, so a
512-row query is 590.356 µs. The previous ratio-only estimate charged
71,591.936 µs. In the recorded 512-token layers that charge exceeded even the
maximum possible gain over their drive-time lower bound. The regression test
replays this bound; it does not establish the speed of an unmeasured GPU path.

Full layer `ntokens` remains the row count used for fitting and placement,
while the grouped engine compacts its rows to the tokens with GPU work.
Correcting that coverage is separate work. The grouped engine's measured
begin interval also includes a small route-list copy already represented by
the route cost; that existing approximation is unchanged here. Sample age,
exploration and transport-direction calibration remain separate work.

`engine/tests/ssd/test-ssd-prefill-cost.cpp` covers the recorded small-to-large
query, repeated same-shape observations, a synthetic second shape, nonnegative
fit boundaries, charging fixed overhead once, and unchanged vector-pass pricing.
The existing placement regression cases run in the same standalone CPU test.
