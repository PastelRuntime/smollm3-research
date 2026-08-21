# EXPERIMENT 2 — RNoPE-SWA Phase B: LoRA fine-tune WITH the window (pre-registered)

Date: 2026-08-20
Status: **pre-registered before push** — kernel `haylee00/smollm3-rnope-swa-phaseb`
Code: `experiments/rnope_swa_phaseb/`

## Motivation (from Phase A results)

Phase A (see `../rnope_swa_v3/EXPERIMENT_1.md`): applying the RNoPE window at
inference only destroys retrieval beyond 8k (0/5 at 16k-64k vs 5/5 baseline) but
delivers 11-21% speed and real memory savings. Hypothesis: the window must be present
*during training*. Phase B tests exactly that.

## Method

- Model: HuggingFaceTB/SmolLM3-3B, fp16, device_map=auto (both T4s), sdpa
- Config: same windowed config as Phase A — `sliding_attention` on the 27 RoPE layers
  (window 8192), `full_attention` on the 9 NoPE layers
- Training: LoRA (r=32, alpha=64, dropout 0.05, targets q/k/v/o projections), adapters
  in fp32, AdamW lr 2e-4, warmup 10 steps, cosine decay, grad accumulation 4,
  gradient checkpointing (non-reentrant), fp16 autocast + GradScaler
- Data: pg19 (public-domain books), streamed, tokenized, cut into 400 blocks of
  16384 tokens (~6.5M tokens). NO synthetic needles — eval data is generated at
  eval time with a random-ish constant; no contamination possible.
- Sequence length 16384 = 2x the window: every training example forces information
  beyond the 8k window on RoPE layers to route through NoPE full-attention layers.
- Chunked cross-entropy (2048-token chunks, each checkpointed) so logits never
  materialize for the full 16k sequence.
- After training: adapter saved to /kaggle/working/adapter_final, then the EXACT
  Phase A NIAH harness re-runs with the adapter active (8k/16k/32k/64k x 5 depths,
  greedy, chunked prefill).

## Pre-registered hypotheses

| # | Hypothesis | Falsified if |
|---|-----------|--------------|
| P1 | Retrieval recovers: NIAH accuracy >= 4/5 at 16k AND 32k with the windowed config | accuracy <= 3/5 at either length |
| P2 | 8k behavior intact (window covers context): accuracy 5/5 at 8k | accuracy <= 4/5 at 8k |
| P3 | Speed win persists: gen_time at 32k within 10% of Phase A's windowed times (28.5s) | gen_time > 31.4s at 32k |

Secondary observations recorded but not gating: loss curve (first vs last block),
chat-format sanity generation, peak memory.

## Decision rules

- **P1 holds** → run the CONTROL kernel (identical LoRA CPT, stock config, no window)
  to prove the window itself — not generic fine-tuning — caused recovery. Then the
  claim is publishable: "train-time windowing recovers RNoPE-SWA for SmolLM3:
  config-only long-context efficiency."
- **P1 fails** → clean negative, publishable with the Phase A result as the full
  story: "RNoPE-SWA does not transfer to SmolLM3, at inference OR with LoRA
  fine-tuning." Adapter released either way.

## Known limitations (stated up front)

- CPT on raw books, not chat SFT — if chat-following collapses, the sanity
  generation will show it and Phase B2 adds chat-format data.
- 400 blocks/100 optimizer steps is an adaptation budget, not convergence; a null
  result is evidence against *cheap* transfer, not against any amount of training.
- Single seed, greedy decode — consistent with Phase A.

## Post-run checklist

1. `kaggle kernels status haylee00/smollm3-rnope-swa-phaseb`
2. `kaggle kernels output haylee00/smollm3-rnope-swa-phaseb -p /tmp/opencode/phaseb`
3. results.json: train_log, sanity_generation, NIAH results vs Phase A table
4. Archive to `run1_data/`, update this file, decide on control kernel