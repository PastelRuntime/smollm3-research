# EXPERIMENT 1 — RNoPE-SWA on SmolLM3 (pre-registered)

Date: 2026-08-19
Status: **running in background** — Kaggle kernel `haylee00/smollm3-rnope-swa-niah-v2` (fresh restart; v1 kernel `haylee00/smollm3-rnope-swa-niah-eval` abandoned after 8 versions of environment hell)
Git: `experiments/rnope_swa_v2/` (script + kernel-metadata.json)

## Environment post-mortem (why the restart)

Kaggle's GPU image ships torch 2.10+cu128 (sm_70+ only) but assigned us P100s (sm_60,
Pascal). Fix that finally works: `pip uninstall torch torchvision torchaudio` then
`pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124`
(the resolver conflict requires uninstalling torchvision/torchaudio first).
v7/v8 of the old kernel failed due to bugs in *my* CUDA probe (wrong assert constant,
then an fp16 sum overflow: 2097152 > fp16 max 65504) — the torch install itself was
already working. v2 installs unconditionally, no probing.

## Question

Does the RNoPE paper's attention recipe — sliding-window attention (SWA) on the **RoPE** layers only, full attention on the **NoPE** layers — transfer to SmolLM3-3B, an untested model whose hybrid architecture (27 RoPE + 9 NoPE layers) is exactly the design the paper targets?

## Why this is novel

- RNoPE (arXiv:2501.18795) validated on 500M–3B *synthetic* models; SmolLM3 is the first real pretrained hybrid.
- HF's own long-context experiments used SWA on **all** layers (training-time) and it hurt RULER. SWA-on-RoPE-only has never been tested on SmolLM3.
- The HF model tree contains zero SWA variants of SmolLM3 (verified via HF API).

## The config gotcha (documented finding)

transformers' SmolLM3 config auto-derives `layer_types` when `None`
(`configuration_smollm3.py:227-238`): it puts `sliding_attention` on `not has_rope`
(= **NoPE**) layers — the **opposite** of the RNoPE recipe. So a naive
`use_sliding_window=True` flip windows the wrong layers. This experiment passes
`layer_types` explicitly:

```
sliding_attention  where no_rope_layers[i] == 1  (27 RoPE layers, window 8192)
full_attention     where no_rope_layers[i] == 0  (9 NoPE layers)
```

Verified supported end-to-end in transformers 4.57.6:
`modeling_smollm3.py:148-150` (per-layer `sliding_window`),
`:353` (`has_sliding_layers`), `:390-416` (per-type causal masks, shared across layers).

## Phase A (this kernel): inference-only NIAH eval

- Model: `HuggingFaceTB/SmolLM3-3B`, fp16, sdpa, greedy decoding, `max_new_tokens=40`
- Eval: single-needle NIAH, lengths [8k, 16k, 32k, 64k] x depths [0.1, 0.3, 0.5, 0.7, 0.9]
- Metric: exact match of needle answer; peak GPU memory; gen tokens/sec
- Sanity check: at 8k context the 8k window covers everything → both configs must
  produce identical generations (`mask_parity_at_8k_window_covered`). This validates
  the sliding path end-to-end.
- 64k is best-effort (materialized causal masks are ~8.6 GB fp16 on a 16 GB T4; OOM
  is an expected possible outcome and will be recorded, not fatal).

## Pre-registered hypotheses and thresholds

| # | Hypothesis | Falsified if |
|---|-----------|--------------|
| H1 | SWA-8k on RoPE layers cuts peak memory vs baseline; gap grows with context length | peak mem at 32k is NOT >=30% lower |
| H2 | NIAH accuracy at <=32k is not meaningfully hurt (retrieval lives in NoPE layers) | exact-match drop > 5 points at any length |
| H3 | If baseline degrades at 64k, SWA holds at least as well (denoising effect) | SWA worse than baseline at 64k (if baseline runs) |

## Decision rules

- **H1+H2 hold** → Phase B justified: LoRA fine-tune with this config on long-context
  data (SmolTalk2/long docs), then RULER at 32k/64k. This is the publishable claim:
  "memory + retrieval win, config-only, no retrain needed for the inference win."
- **H1 only** → still a clean inference-memory result worth reporting.
- **H2 fails** (accuracy tanks) → negative result is still publishable *because* of the
  layer_types inversion detail; write it up as "the natural implementation does the
  wrong thing, and here is what actually happens."

## Caveat on H1 (recorded before launch)

Peak-memory savings only materialize if the cache evicts windowed positions. With the
default DynamicCache, windowing is enforced by attention *masks* (materialized ~1-8 GB
at 32k-64k in fp16) while the KV cache still stores everything — so H1 may fail on
peak-memory grounds even if attention FLOPs (and thus gen_time_s) improve. Both are
recorded per-run; report what we see, honestly. The fine-tune phase (B) would pair the
windowed config with a StaticCache/SlidingWindowCache for the true memory win.

## Post-run checklist

1. `kaggle kernels status haylee00/smollm3-rnope-swa-niah-v2`
2. `kaggle kernels output haylee00/smollm3-rnope-swa-niah-v2 -p /tmp/opencode/kaggle_out`
3. Inspect `results.json` (meta, mask parity, per-run records, summary)
4. Update this file with results; decide Phase B