# ENGINE.md — videng: native multi-GPU video/image inference for peasant hardware

> **STATUS: PARKED** — do not start until all three activation gates pass.
> This document is the complete blueprint so that when the gates open, execution
> starts immediately with zero re-planning.

## Activation gates (all three required)

1. **Unlimited local compute** — own hardware, no quota clocks.
2. **Deep model-type understanding** — enough architecture fluency to eventually
   build a brand-new model type, not only serve existing ones.
3. **Financial runway** — several weeks of undistracted work possible.

Until then: presence-building continues on free compute (interrogator track,
weekly bounded results). This document ages in the repo, publicly — which is
itself part of the presence strategy: a visible, well-specced plan attracts
collaborators before a single line ships.

---

## The thesis

Open generative media models became huge and MoE-shaped (HunyuanImage-3.0: 80B
total / 13B active; MAGI-2: 114B / 6B active; Wan family; LTX-2.x). MoE means
memory-heavy, compute-light — exactly inverted from what modern GPUs optimize
for. Meanwhile decommissioned datacenter GPUs flood the secondary market
(HGX A100 8x80GB ~ $75-95K; 4xA100 PCIe ~ $24K; V100 fleets ~ $8K/256GB).
Every parallel-inference paper (xDiT, DistriFusion, PipeFusion, ParaAttention,
LongLive-2.0) benchmarks exclusively on NVLink/RoCE clusters. **Zero published
numbers exist for PCIe-only consumer/decommissioned hardware classes**, and the
MoE expert-routing locality that would make cheap-fleet serving efficient has
never been measured for diffusion models.

videng = the native engine + the missing benchmark + the receipts.

## Hardware tiers

| Tier | Rig | VRAM | ~Cost | $/GB |
|---|---|---|---|---|
| Dev | Kaggle 2xT4 (free) | 32GB | $0 | $0 |
| Budget fleet | 8xV100 PCIe (used) | 256GB | ~$8K | $31 |
| Sweet spot | 4xA100 80GB PCIe (used) | 320GB | ~$24K | **$75** |
| Enthusiast | 4xRTX PRO 6000 Blackwell | 384GB | ~$36K | $94 |
| Hero | HGX A100 8x80 SXM (used) | 640GB | ~$85K | $133 |

Flagship demo targets: HunyuanImage-3.0 INT8 (~90GB -> fits budget fleet),
MAGI-2-class INT8 (~114GB), Wan/LTX families for dev iterations.
One-time validation rentals: 4-8xA100 cloud @ ~$12/hr, 2-3 hrs (~$40).

## The rung ladder (each rung: pre-register -> kernel -> receipt -> post)

### R0 — Baseline atlas (dev rig, free)
Single-GPU Wan-1.3B / LTX-2B / Wan-2.2-5B: fps, VRAM curve vs resolution/frames,
deterministic output fingerprints (seeded). Establishes every later speedup's denominator.

### R1 — CFG split (dev rig, free)
Conditional branch on GPU0, unconditional branch on GPU1. Zero inter-GPU comm.
- Pre-reg stub: >=1.7x wall-clock at equal quality (LPIPS <= 0.01 vs single-GPU output).
- Falsifier: <1.4x (loader overhead ate it) or quality divergence.

### R2 — Component placement (dev rig, free)
Text encoder pinned GPU0, DiT GPU1, VAE decode split by frame tiles back to GPU0.
Latents cross the wire once per generation; weights never move.
- Pre-reg stub: >=25% latency win over best single-GPU placement at 720p.

### R3 — Expert-stationary timestep-MoE (dev rig, fp8)
Wan-2.2-A14B: high-noise expert resident GPU0, low-noise expert GPU1. Weights
never travel; latents hand off once at the timestep boundary. **First-ever public
measurement of expert activation patterns across diffusion steps** — routing
stickiness decides everything downstream.
- Pre-reg stub: expert handoff adds <10% overhead vs single-expert baseline;
  routing stability >=90% within denoising phases.
- Falsifier: chaotic routing (handoff cost dominates) -> pivot to offload-first design.

### R4 — Sequence/context parallelism port (rented or dev rig)
USP (Ulysses+Ring) and DistriFusion-style stale-activation patch parallelism,
ported and measured on PCIe-class links where nobody has benchmarked them.
- Pre-reg stub: characterize the crossover — at what comm/compute ratio does each
  strategy beat pipeline placement on 16GB cards?

### R5 — THE MISSING TABLE (the citable artifact)
Consumer/decommissioned-hardware benchmark: strategies x models x link types x
precisions. Published, reproducible, versioned. This is the contribution the
field literally does not have.

### R6 — videng v1
Thin pip-installable engine: `python -m videng "prompt" --gpus all`.
Model-agnostic config, farm/queue mode, ComfyUI-compatible API endpoint so
existing workstation UX drives it. GGUF-style quantization support.

### R7 — Flagship validation
HunyuanImage-3.0 INT8 natively across a rented multi-GPU box, then (post-gate-1)
on owned hardware. Headline: *80B-class generation on $8K of scrap silicon.*

## Risks / honest unknowns

- T4s are sm_75: no bf16, no flash-attention — some kernels may need xformers
  fallbacks; findings may shift on Ampere+. Mitigation: R0 atlas measures first.
- Wan/LTX APIs evolve fast; pin versions per rung, archive wheels.
- MoE routing might be chaotic (R3 falsifier exists for exactly this).
- Single-maintainer systems projects die from scope creep — the rung structure
  exists so any rung can be the last one shipped and still be worth shipping.

## Relation to the longer arc

Serving teaches the systems layer; the stated ambition — building a brand-new
model type — needs the training/architecture layer too. Deliberate sequence:
engine first (infrastructure credibility + audience), novel-model work after
gate 2, informed by everything the engine work exposed about how these models
actually behave under constraint.
