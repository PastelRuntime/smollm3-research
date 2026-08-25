# EXPERIMENT 2 — 2×2 CLOSE-OUT (pre-registered addendum)

## RESULTS (2026-08-24, 40/40 runs)

| Weights \ Eval config | Stock | Windowed (8k) |
|---|---|---|
| Baseline (no adapter) | 5/5 all lengths | 5/5 @8k, **0/5** @16k+ |
| Treatment (LoRA w/ window) | **5/5 all lengths** ✓ | **5/5 all lengths** |
| Control (LoRA stock) | 5/5 all lengths | **5/5 @8k, 0/5 @16k+** ✓ |

Both pre-registered predictions confirmed:

- **Cell A: control weights collapse under the window** — 0/5 at 16k/32k/64k,
  identical to unadapted Phase A weights. Generic LoRA CPT does NOT confer
  robustness to windowed attention. This was the causal keystone.
- **Cell B: treatment weights retain full stock performance.** No overfitting
  to the windowed regime; no capability trade-off.

**Conclusion:** recovery past the window occurs ONLY in the cell where training
happened WITH the window active AND eval uses the windowed config. Adaptation
per se explains nothing (Cell A); adaptation without the window costs nothing
(Cell B). The causal claim is closed:
*RNoPE-SWA transfers to SmolLM3 specifically via training under the windowed
config — ~6.5M tokens of LoRA CPT, a few GPU-hours.*

---

Date: 2026-08-24, pushed before kernel launch.
Kernel: `haylee00/smollm3-rnope-swa-2x2` — eval-only, adapters loaded from the
completed treatment/control kernel outputs via kernel_sources.

## The matrix (predictions, written before launch)

| Weights \ Eval config | Stock | Windowed (8k) |
|---|---|---|
| Baseline (no adapter) | 5/5 all lengths (Exp 1 + control arm) | 5/5 @8k, **0/5** @16k+ (Exp 1) |
| Treatment (LoRA w/ window) | **Cell B — untested** | **5/5 all lengths** (Exp 2) |
| Control (LoRA stock) | 5/5 all lengths (Exp 2) | **Cell A — untested** |

## Pre-registered predictions

- **Cell A (control weights, windowed eval): retrieval collapses** — ≤1/5 at
  16k/32k/64k, replicating Phase A with fine-tuned weights. This is the causal
  keystone: generic LoRA CPT does not confer robustness to the window.
  *If Cell A instead passes (≥4/5 at 16k AND 32k): any-LoRA explains recovery,
  the window-during-training claim weakens materially, and we report that.*
- **Cell B (treatment weights, stock eval): full retrieval retained** — 5/5 at
  all lengths. Training under the window should not break stock-config behavior.
  *If Cell B fails (<4/5 anywhere beyond 8k): treatment overfit to the windowed
  regime — capability trade-off, reported as such.*

## Method

Identical NIAH harness to Experiments 1/2 (same needle, lengths 8k–64k, depths
0.1–0.9, greedy, chunked prefill 1024, fp16, 2×T4, device_map=auto). Adapters
merged (merge_and_unload) before eval. 40 runs total, ~3h actual.

## Decision

Matrix complete → fill interpretation table in EXPERIMENT_2_CONTROL.md, close
Experiment 2, update README + X post follow-up.
