# NORTH_STAR.md — The One-Machine Studio

> Where this project is going. Written 2026-08-24 after the strategy sessions that
> followed Experiment 2. Everything below traces back to receipts already in this
> repo or pre-registered experiments.

## The vision

A single machine — cloud or under a desk — that generates full audio+video+image
content end to end, all local, single user. No inference department, no API bills.

```
┌──────────────────────────────────────────────────────┐
│  THE ONE-MACHINE STUDIO                              │
│                                                      │
│  SMALL BRAINS (sub-5B, chat-speed, any GPU):         │
│   - Spec Interrogator: vague idea → engineering doc  │
│     (Experiment 3, pre-registered)                   │
│   - Prompt one-upper: rewrite prompts with           │
│     reward-verified deltas (planned)                 │
│              ↓ specs & prompts                       │
│  THE ENGINE (videng, to be built):                   │
│   - native multi-GPU inference for MoE diffusion     │
│     models — expert pinning, layer pipeline,         │
│     offload, CFG split. No ComfyUI block-swap agony. │
│              ↓ latents                               │
│  BIG MODELS: HunyuanImage-3.0 (80B/13B active),      │
│   MAGI-2-class (114B/6B active), Wan, LTX-2.5...     │
│   the field converged on memory-heavy, compute-light │
│   MoE — perfectly shaped for cheap GPU fleets        │
└──────────────────────────────────────────────────────┘
```

## Why the engine matters (the market gap)

- The 2026 open-model releases are huge and MoE: HunyuanImage-3.0 needs ~181GB
  FP16 / ~90GB INT8; official recommendation is 3-8×80GB GPUs.
- Used-GPU economics: HGX A100 8×80GB ~$85K, 4×A100 PCIe ~$24K, 8×V100 ~$8K.
  INT8 80B fits entirely in a $8-10K junk fleet with headroom.
- Every parallel-diffusion paper (xDiT, DistriFusion, PipeFusion) benchmarks on
  A100/H100/MI300X with fat interconnects. Zero published numbers exist for the
  PCIe/old-silicon/16GB-card class — which is exactly what Kaggle 2×T4 is.
- MoE expert activation locality across diffusion steps has never been measured.
  That's a ~$3 rented-GPU experiment that decides the engine's core design.

## The brain thesis (small-model science)

Competence in the weights, catalog in the context. Small models (SmolLM3-3B class,
runs on an 8th-gen iGPU) trained for *skills* — interrogation, routing within
themselves, prompt craft — with current knowledge injected via context, so they
never go stale as models churn every few weeks.

Core belief (3 years of vibe coding, verified by this repo's own discipline):
**the prompt is the spec, and the spec is the product.** Agents amplify
discipline, not intent. Shit in, shit out; spec in, engineering out.

## Phase ladder (current)

| Phase | What | Status |
|---|---|---|
| 1-2 | RNoPE-SWA: windowed inference kills retrieval; LoRA WITH window restores 5/5 to 64k | ✅ done, needs 2×2 close-out kernel |
| 3 | Spec Interrogator (clarify-aware LoRA on SmolLM3, ClarifyCodeBench) | pre-registered, EXPERIMENT_3.md |
| 4 | Engine Phase 0: CFG split on 2×T4 (Wan 1.3B) | designed |
| 5 | Expert-locality atlas: Wan 2.2-14B experts on 2×T4, then HunyuanImage-3.0 on rented 4×A100 (~$3) | designed |
| 6 | videng v1: INT8 80B native on rented box (~$40) | planned |
| 7 | The studio: brain + engine + big model, one machine | the dream |

## Rules (unchanged)

Nothing runs locally except experiments on free remote compute. No replications.
Pre-register everything. Report what the data says. One verified claim per post.
