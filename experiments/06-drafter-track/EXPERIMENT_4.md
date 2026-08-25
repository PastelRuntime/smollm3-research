# EXPERIMENT 4 — Drafter Track: speculative decoding for Qwen3.8-27B on 2×T4 (pre-registration skeleton)

Date: 2026-08-24
Status: **RECON PHASE** — hypotheses below are locked; drafter identities pending recon output.
Kernel: `haylee00/qwen38-drafter-recon` (CPU) → `haylee00/qwen38-drafter-bench` (GPU, after recon)

## Background

Speculative decoding: a small fast drafter proposes K tokens, the big target verifies in one
batch pass. Value scales with target cost-per-token → dense targets are the textbook best,
and constrained hardware amplifies gains. Every published benchmark is frontier-pair on fat
GPUs; nobody has measured acceptance/speedup for the current hot open dense model on
PCIe-class consumer hardware. Qwen3.8-27B (dense per community reports, Apache 2.0, days old)
is the ideal specimen: hottest model on Earth right now, dozens of derivative fine-tunes
appearing daily all inheriting its tokenizer — **one drafter accelerates the whole family.**

## Hard exclusions

SmolLM3-3B cannot participate: Llama-3.2 tokenizer vs Qwen tokenizer. Speculation requires
identical vocabulary. (Experiment 3's interrogator track remains the 3B's home.)

## Conditions

| Condition | Drafter | Cost |
|---|---|---|
| vanilla | none | target only |
| n-gram / prompt-lookup | ~free | code-heavy tasks expected strong |
| tiny family draft | 0.6–1.7B same-vocab Qwen (recon picks exact IDs) |
| mid family draft | 2–4B same-vocab Qwen |
| 3B control | only if a vocab-matched ~3B Qwen exists |

Target: INT4 quant (community GGUF if arch supported by llama.cpp, else HF bitsandbytes +
assisted generation). Split across both T4s. Text-only generation (VL image tokens excluded v1).

Tasks: 30 code + 30 prose + 30 chat prompts, ~300 tokens each, fixed seeds.

## Pre-registered hypotheses (locked before bench launch)

| # | Hypothesis | Falsified if |
|---|-----------|--------------|
| H1 | Best family drafter reaches ≥40% mean acceptance on prose | <25% |
| H2 | Best config achieves net ≥1.3× tok/s vs vanilla on ≥2 of 3 domains | no domain ≥1.15× |
| H3 | n-gram wins net speedup on code specifically | loses to model drafters on code |
| H4 (drafter-size law) | Net speedup peaks at ≤2B drafter; larger adds latency without proportional acceptance gain | 3B/4B control wins net |

**Wasted-compute line:** if NO configuration reaches net ≥1.15× anywhere, conclusion =
"speculation does not survive PCIe-class reality for this pairing" — track ends with a
published negative. Exposure capped at ~4–7 quota hours total.

## Risks

- Brand-new architecture may lack llama.cpp support → fallback path (HF assisted generation),
  noted in results either way.
- Quantized-target speculation may behave differently than fp16 literature numbers.
- Community GGUFs may lag the release; we can quantize ourselves via convert script if needed.

## Post-run checklist

1. Recon: confirm dense, lock target ID + matched drafters → fill drafter table above
2. Bench kernel push → results.json archived here
3. Verdict table + README update log + X post if H2 lands
4. If H2 holds: this becomes a standing component of ENGINE.md's stack (drafter-on-FreeToken
   interaction study queued as follow-up)
