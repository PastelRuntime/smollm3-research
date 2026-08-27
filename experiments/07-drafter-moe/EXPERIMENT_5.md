# EXPERIMENT_5 — Does speculative drafting survive expert offload?

> Pre-registered 2026-08-26 BEFORE any kernel ran. Predictions below are
> committed; results get reported whether they confirm or falsify.
> Standing rules apply: no replications, report what the data says,
> one verified claim per post.

## The question

ENGINE.md gap R4: the interaction between speculative drafting and
expert-offload scheduling is unstudied. If a drafter's proposals keep getting
accepted while cold MoE experts stream from CPU→GPU per step, then
drafting+offload compose and videng gets both speedups stacked. If they
cancel (e.g., accept-rate collapses because offload changes numerics, or
drafting hides less latency because expert streaming dominates), that kills
a core engine assumption *before* it costs $40 on rented GPUs.

## Sizing recon (done locally, 2026-08-26, zero quota burned)

Via HF API `model_info(files_metadata=True)`:

| model | bf16 repo | verdict vs 2×T4 (~30 GB) |
|---|---|---|
| Qwen/Qwen3.5-35B-A3B | 71.9 GB / 14 shards | **primary target**: ~19 GB @4-bit fully resident; ~36 GB @8-bit → offload scenario reachable by packing 8-bit experts across GPU0+CPU |
| Qwen/Qwen3.5-122B-A10B | 250.2 GB / 39 shards | too large even at 4-bit (~63 GB); would be all-CPU-bottlenecked — excluded from Phase A/B |
| Qwen/Qwen3.5-397B-A17B | 806.8 GB | reference only |
| Qwen3.5-0.8B / 2B drafters | small dense | vocab_match=true vs 122B family confirmed in exp4 recon2.json |

Assumption to verify inside kernel (not assumed true here): Kaggle T4×2 host
RAM is sufficient to hold the ~17–20 GB of offloaded 8-bit experts; kernel
prints `psutil` totals before loading anything.

## Design

**Phase A — sizing + residency probe (this kernel):**
Load Qwen3.5-35B-A3B @ 4-bit across 2×T4. Record: actual VRAM per GPU, host
RAM used, time-to-first-token for a fixed prompt, tokenizer match check
against drafter candidates (Qwen3.5-0.8B, Qwen3.5-2B). No drafting yet.
Cheap, survives being run alone.

**Phase B — 2×2 measurement kernel (separate push, only if Phase A green):**
1. Target alone, all-experts-resident, greedy decode → tokens/sec (baseline B0)
2. Drafter+target speculative decoding, all-experts-resident → accepted-tokens/step + effective tok/s (A0)
3. Target under forced expert-offload regime (selected experts pinned to CPU
   memory, streamed per layer), NO drafting → O1
4. Same offload regime WITH drafting → A1

**Primary metric:** composition ratio C = (effective tok/s in cell 4) /
(tok/s cell 3 × speedup-ratio observed in cell 2). C ≈ 1 means drafting and
offload compose multiplicatively. Meaningful evaluation window ≥ 32 generated
tokens × ≥ 10 prompts (12–30 s decode per cell at minimum — enough signal).

**Drafter choice:** Qwen3.5-2B preferred over 0.8B (higher acceptance expected;
0.8B fallback if VRAM-tight after 4-bit packing). Decided BEFORE Phase B runs,
from Phase A's free-memory numbers — recorded in this doc, not chosen after
seeing drafting results.

## Pre-registered predictions

- P1: drafter accept rate in cell A0 is ≥ 40% of drafted tokens (else pair unusable).
- P2: offload itself costs ≥ 30% throughput vs B0 at these sizes.
- P3 (the composition claim): C ≥ 0.85 — i.e., stacking drafting on top of an
  already-offloaded target retains ≥85% of the drafting benefit measured
  without offload. **Falsified if C < 0.70.**
- Nothing about quality; quality regression gates come later (acceptance +
  output-logit agreement on a fixed prompt set first).

## Anti-theater controls

- Fixed seeds, greedy target decode everywhere (no sampling noise).
- Same prompt set pre-written in-kernel before launch.
- Phase A failures (OOM, cudaErrorNoKernelImageForDevice-class issues, mmap
  thrash) are reported as Phase A results, not silently retried into P-hacking.
- If Phase A shows < 2 GB free VRAM after load, Phase B cell set shrinks to
  0.8B drafter before anything is measured — also recorded here when done.

## Environment pins (learned from exps 1–2)

- `"machine_shape": "NvidiaTeslaT4"` pinned; uninstall torchao<0.16 at boot;
  datasets pin irrelevant here (no pg19); write results.json BEFORE printing DONE.
