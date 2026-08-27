# EXPERIMENT 1 — RNoPE-SWA on SmolLM3 (pre-registered)

Date: 2026-08-19 (v3, clean restart)
Status: **RESULTS IN** — run 1 data archived in `run1_data/` (40/40 runs ok).
Kernel: `haylee00/smollm3-rnope-swa-niah` (version 4 re-running for a clean COMPLETE
status after a cosmetic crash in the summary print; data was already saved).

## RESULTS (Phase A, T4, fp16, chunked prefill)

| length | baseline acc | rnope_swa acc | swa speed | swa peak mem |
|--------|-------------|---------------|-----------|--------------|
| 8192 | 5/5 | 5/5 (parity OK) | ~ -4% time | same |
| 16384 | 5/5 | 0/5 | -21% time | -0.15 GB |
| 32768 | 5/5 | 0/5 | -18% time | -0.37 GB |
| 65536 | 5/5 | 0/5 | -11% time | -0.70 GB |

Interpretation:
- **H2 falsified (cleanly)**: SWA-8k on the RoPE layers at *inference only* destroys
  retrieval beyond the window. The RNoPE recipe evidently requires training with the
  window, not post-hoc application. Baseline stays perfect at 64k (11.53 GB peak,
  105s prefill).
- **H1 partially confirmed**: ~11-21% inference-time speedup and real peak-mem savings
  at long context, but only usable if accuracy is preserved — which requires Phase B
  (fine-tune WITH the window).
- Pipeline validated end-to-end by the 8k parity check (identical generations when
  the window covers the context).
- Note: with DynamicCache the windowed layers still write full KV; a
  SlidingWindowCache in Phase B would cut KV memory ~70% at 64k. The measured
  speedup is therefore a lower bound on the possible win.

## Decision

Phase B is now justified **and has a sharp question to answer**: fine-tune with the
windowed config (LoRA on long-context data), then re-run this exact NIAH harness.
If accuracy recovers while keeping the speed win -> the publishable result. If not,
we have a clean negative: "RNoPE-SWA does not transfer to SmolLM3 even with
fine-tuning" plus the layer_types inversion gotcha.

## Environment lessons (append as discovered)

- Kaggle GPU defaults to P100 without `machine_shape: NvidiaTeslaT4` in metadata;
  the image's torch (2.10+cu128) cannot run Pascal at all.
- On T4 (sm_75), SDPA has no flash kernel; `model.generate` on a long prompt
  materializes the full attention score matrix via the math fallback -> OOM at 8k.
  Fix: manual chunked prefill (1024-token slices) + greedy decode loop. All 40 runs
  then completed, including 64k in 11.5 GB.

## Question

Does the RNoPE paper's attention recipe — sliding-window attention (SWA) on the
**RoPE** layers only, full attention on the **NoPE** layers — transfer to SmolLM3-3B,
the first real pretrained hybrid (27 RoPE + 9 NoPE layers)?

## Why this is novel

- RNoPE (arXiv:2501.18795) validated on 500M–3B *synthetic* models; SmolLM3 is the
  first real pretrained hybrid.
- HF's own long-context experiments used SWA on **all** layers (training-time) and it
  hurt RULER. SWA-on-RoPE-only has never been tested on SmolLM3.
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

Verified supported in transformers 4.57.6: `modeling_smollm3.py:148-150` (per-layer
`sliding_window`), `:353` (`has_sliding_layers`), `:390-416` (per-type causal masks).

## Method (Phase A: inference-only NIAH eval)

- Model: `HuggingFaceTB/SmolLM3-3B`, fp16, sdpa, greedy decoding, `max_new_tokens=40`
- Eval: single-needle NIAH, lengths [8k, 16k, 32k, 64k] x depths [0.1, 0.3, 0.5, 0.7, 0.9]
- Metric: exact match of needle answer; peak GPU memory; generation time
- Sanity check: at 8k context the 8k window covers everything → both configs must
  produce identical generations (`mask_parity_at_8k_window_covered`)
- 64k is best-effort (materialized causal masks are ~8.6 GB fp16 on a 16 GB GPU; OOM
  is an expected possible outcome and will be recorded, not fatal)

## Pre-registered hypotheses and thresholds

| # | Hypothesis | Falsified if |
|---|-----------|--------------|
| H1 | SWA-8k on RoPE layers reduces attention compute at long context; gen_time at 32k improves or holds | gen_time at 32k increases materially (>20%) |
| H2 | NIAH accuracy at <=32k is not meaningfully hurt (retrieval lives in NoPE layers) | exact-match drop > 5 points at any length |
| H3 | If baseline degrades at 64k, SWA holds at least as well (denoising effect) | SWA worse than baseline at 64k (if baseline runs) |

H1 is deliberately framed on **time**, not peak memory: with the default DynamicCache,
windowing is enforced by attention *masks* while the KV cache still stores everything,
so peak-memory savings may not materialize at inference time. Both are recorded per
run; report what we see, honestly. The true memory win would come in Phase B with a
StaticCache/SlidingWindowCache.

## Decision rules

- **H1+H2 hold** → Phase B justified: LoRA fine-tune with this config on long-context
  data, then RULER at 32k/64k. Publishable claim: quality holds, compute drops,
  config-only.
- **H1 only** → still a clean inference-speed result worth reporting.
- **H2 fails** → negative result is still publishable because of the layer_types
  inversion detail: "the natural implementation does the wrong thing, and here is
  what actually happens."

## Environment post-mortem (v1/v2 kernels, for the record)

1. v1: Kaggle CLI 2.2.4 reads `enable_gpu` (not `is_gpu`/`accelerator`) from
   kernel-metadata.json → ran on CPU → "Torch not compiled with CUDA".
2. v2-v8: `enable_gpu` worked but Kaggle assigned **P100** (Pascal, sm_60). The image
   ships torch 2.10+cu128 (sm_70+) → "no kernel image is available". Root cause:
   API pushes without `machine_shape` default to P100. **Fix: `"machine_shape":
   "NvidiaTeslaT4"` in kernel-metadata.json** (values: NvidiaTeslaT4 / NvidiaTeslaP100 /
   Tpu1VmV38, per kagglesdk docs).
3. v7/v8: my CUDA probe was buggy (wrong assert constant, then an fp16 sum overflow:
   2,097,152 > fp16 max 65,504 → inf). The torch 2.4.1+cu124 downgrade itself was
   already working — proven by the fact that CUDA ops executed before the assert fired.
4. v3 script bug caught in review: `swa.use_sliding_window` referenced a local of
   `make_configs()` → NameError before any eval. Fixed to `swa_cfg.use_sliding_window`.

The v3 script keeps a P100 fallback (subprocess probe with correct fp32 math →
uninstall torch trio → install torch==2.4.1+cu124 from the PyTorch cu124 index) so it
survives any GPU assignment.

## Post-run checklist

1. `kaggle kernels status haylee00/smollm3-rnope-swa-niah`
2. `kaggle kernels output haylee00/smollm3-rnope-swa-niah -p /tmp/opencode/kaggle_out`
3. Inspect `results.json` (meta, mask parity, per-run records, summary)
4. Update this file with results; decide Phase B