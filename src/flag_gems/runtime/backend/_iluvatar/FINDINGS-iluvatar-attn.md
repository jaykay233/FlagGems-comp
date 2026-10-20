# Iluvatar fused attention notes (FlagGems-comp)

Companion detail and e2e numbers live in
`vllm-plugin-FL-comp/experiments/FINDINGS-iluvatar.md` §17.x.

## 2026-10-20

### Speculative MTP decode

* `paged_attention_decode.paged_attention_mtp` — KV once for short verify
  batches (`q` in 2..4), used when the plugin enables n-gram / other draft.
* `ixformer_unified_attention` routes short speculative-shaped batches there
  when `VLLM_FL_ILUVATAR_SPEC_MTP` is on (default). Plugin keeps **CPU n-gram
  off by default** (kills async scheduling; random-bench accept rate too low).

### Prefill FA shape gate `(n_seq, min_q, kv_len)`

Calibrated on `/root/bench_results/jagged_prefill_micro.txt`:

| Admit FA when | Else |
|---|---|
| `n≤8` and `min_q≥128` | Triton (`short_skip`) |
| `n≤8` and `min_q≥64` and `kv≥1024` | Triton |
| `n≤16` and `min_q≥256` | Triton |
| mixed (`q_len==1` present) | Triton (`mixed_skip`) |

Cheap sync-free reject for large decode+chunk batches still uses floor
`VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q` (default 256). Disable the table with
`VLLM_FL_ILUVATAR_IXFORMER_PREFILL_SHAPE_GATE=0`.

Many short jagged packs (e.g. 32×64) are **slower** on FA than Triton — do not
lower the gate blindly.
