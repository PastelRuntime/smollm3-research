# EXPERIMENT 2 — CONTROL ARM: identical LoRA CPT, stock config (pre-registered)

Date: 2026-08-21
Status: **pre-registered before launch** — kernel `haylee00/smollm3-rnope-swa-control`
Code: `experiments/03-lora-control/` (this folder)

## Amendment to EXPERIMENT_2.md

The original plan launched the control only if Phase B's P1 held. Amended before
either arm produced any results: the control launches **concurrently** with Phase B.
Same account, same GPU type, same data pipeline, same LoRA setup, same eval harness,
same day. This removes time/order confounds and strengthens causal attribution either
way the treatment lands.

## What differs from Phase B (the ONLY differences)

1. Config: stock `AutoConfig.from_pretrained(...)` — no `use_sliding_window`,
   no `sliding_window`, no `layer_types` override. Default SmolLM3 behavior
   (full attention everywhere; verified as baseline in Experiment 1: 5/5 at 64k).
2. Everything else is byte-identical logic: pg19 stream → 400 blocks × 16384 tokens,
   LoRA r=32 α=64 on q/k/v/o, lr 2e-4 cosine + warmup, grad accum 4, grad
   checkpointing, chunked CE, then the same NIAH eval at 8k–64k × 5 depths.

## Pre-registered predictions

| # | Prediction | Falsified if |
|---|-----------|--------------|
| C1 | Control retrieval ≥ treatment retrieval at 16k and 32k IF treatment recovers; i.e., control alone does NOT explain recovery | control shows ≥4/5 at 16k AND 32k AND treatment also ≥4/5 there with no meaningful gap |
| C2 | Control training loss curve ≈ treatment's (both adapt similarly under their own configs) | loss curves differ by >0.15 nats on average over last 20 blocks |

Interpretation matrix:

- Treatment recovers + control doesn't → **window during training caused recovery** (publishable positive)
- Neither recovers → RNoPE-SWA doesn't transfer to SmolLM3 at this budget (publishable negative)
- Both recover → generic LoRA CPT explains it; window adds nothing at this budget (still a result)
- Control recovers, treatment doesn't → window actively harms adaptation (most surprising outcome)

## Post-run checklist

1. `kaggle kernels status haylee00/smollm3-rnope-swa-control`
2. Pull results.json → compare against Phase B arm
3. Fill interpretation matrix above, update README update log