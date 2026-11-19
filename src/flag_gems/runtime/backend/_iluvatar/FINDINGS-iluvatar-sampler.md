# Iluvatar sampler notes (FlagGems-comp)

Companion e2e numbers live in
`vllm-plugin-FL-comp/experiments/RESULTS-sampler-fused-20261119.md`.

## 2026-11-19 — `top_p_sample`: fused top-p + exponential-noise sampling

### What it replaces

vLLM's post-top-p sampler is

```
ids = (softmax(masked_logits) / q).argmax(-1),   q ~ Exponential(1)
```

That materialises three full-vocabulary float32 tensors per decode step
(`masked_logits`, `probs`, `probs/q`) and runs softmax, a divide and argmax
as separate kernels.  The softmax denominator is a positive per-row constant,
so argmax cannot see it:

```
argmax(softmax(x) / q) == argmax(exp(x - rowmax) / q)
```

### Implementation

`src/flag_gems/ops/top_p_sample.py`, public as `flag_gems.top_p_sample`.

* Reuses the v2 threshold machinery of `top_p_threshold`:
  per-block row max -> histogram of `(x - rowmax)` in log2-probability ->
  cutoff.  Unlike `top_p_threshold`, it does not write masked logits.
* One blocked pass then computes `exp2((x - rowmax) * log2e) / q` for
  `x >= threshold`, per-block max + argmax, writing two small partials.
* A one-program-per-row reduce picks the global argmax and stores an int64
  token id.  Block order is ascending, so ties resolve to the first index,
  matching `torch.argmax`.

Caller contract: `q` must be generated exactly as vLLM's `random_sample`
does (`empty_like(logits)` + `exponential_()` + per-request generators);
`logits`/`q` are fp32 `(rows, vocab)` CUDA tensors; `rows <= 128`
(matching `top_p_threshold`'s v2 crossover).  Above any bound the plugin
patch falls back to the original softmax + `random_sample`.

### Correctness

Against `(top_p_threshold(logits, p).softmax(-1) / q).argmax(-1)` on the
real vocab (130560), fp32:

| rows | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| scale=1, p=0.95 | 0 mismatch | 0 | 0 | 0 | 0 |
| scale=4, p=0.95 | 0 | 0 | 0 | 0 | 0 |
| p=1.0 (rows=32) | — | — | 0 | — | — |

All reference selections are inside the top-p kept set.

### Performance (single step, V=130560, BI-V150)

| rows | old chain | `top_p_sample` | saved |
|---|---|---|---|
| 32 | 1119.3 µs | **374.2 µs** | 745.1 µs (−66.6%) |
| 64 | 1556.0 µs | **538.8 µs** | 1017.2 µs (−65.4%) |

The old chain here is `top_p_threshold` + `softmax` + `probs/q` + `argmax`
on the same logits.  E2E: 4k +1.67% (B+A), 16k +1.01%; MATH-500 L3 98.06%
at write time.

### Env (plugin side)

* `VLLM_FL_FUSED_TOPP_SAMPLE=0` disables the fused path.
* `VLLM_FL_FUSED_TOPP_MAX_ROWS` (default 128) bounds it.
