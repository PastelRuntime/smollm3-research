# SmolLM3 Research

Independent, pre-registered experiments on [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) —
Hugging Face's 3B hybrid long-context model. Everything runs remotely (Kaggle/HF),
nothing local. Every hypothesis is written down **before** the run; negative
results get posted too.

## The headline result

**Sliding-window attention doesn't destroy long context — if the model adapts under it.**

SmolLM3 has 27 RoPE layers + 9 NoPE layers. Windowing the RoPE layers at 8k
inference-time-only: needle-in-haystack collapses to **0/5 past 8k**, but runs
11–21% faster. Fine-tuning *with* the window active (LoRA r=32, ~6.5M tokens,
a few free GPU-hours): retrieval returns to **5/5 at 8k, 16k, 32k, and 64k**
— full retrieval at 8× the window. Concurrent control arm proves it's the
window-adaptation, not generic LoRA.

| NIAH length | stock + window | LoRA trained under window |
|---|---|---|
| 8k | 5/5 | 5/5 |
| 16k | 0/5 | **5/5** |
| 32k | 0/5 | **5/5** |
| 64k | 0/5 | **5/5** |

📦 **Weights:** [PastelRuntime/SmolLM3-RNoPE-SWA-Adapters](https://huggingface.co/PastelRuntime/SmolLM3-RNoPE-SWA-Adapters)
(treatment + control) · 📜 Full pre-registrations + raw JSONs in `experiments/`

## Experiment index

| # | Question | Status | Result |
|---|---|---|---|
| 1 (`01-rnope-swa-inference`) | Does inference-time RNoPE-SWA keep retrieval? | ✅ done ×2 | H falsified — 0/5 past window, speed win real |
| 2 (`02-rnope-swa-lora` + `03-lora-control`) | Does training *under* the window restore it? | ✅ closed, 2×2 complete | Yes — 5/5 to 64k; no free lunch on speed |
| 3 (`04-spec-interrogator`) | Can a 3B be LoRA'd into a spec-writing interrogator? | 📋 pre-registered | — |
| 4 (`06-drafter-track`) | Qwen3.8-27B speculation-drafting recon for engine work | 📋 pre-registered | — |
| 5–6 | videng Phase 0: CFG split / expert-locality atlas (see `ENGINE.md`) | designed | — |

## The model

SmolLM3-3B is a 36-layer hybrid: **27 RoPE + 9 NoPE layers** (no positional
embedding every 4th layer), 64k native context (128k YaRN), dual `/think`
modes, GQA. The RNoPE recipe (arXiv:2501.18795) claims long-context retrieval
*and* memory savings — nobody had tested it on this actual model. That was the gap.

### Finding 1 — the `layer_types` inversion in transformers

transformers' SmolLM3 config auto-derives `layer_types` when unset and assigns
`sliding_attention` to the **NoPE** layers — the **opposite** of the RNoPE
recipe. Anyone who "just enables SWA" windows the wrong layers. All experiments
here pass `layer_types` explicitly. If you replicated RNoPE-SWA and got garbage:
check this first.

## Repo layout

```
experiments/   numbered = canonical; each holds pre-registration, kernel script, run data
artifacts/     treatment/control LoRA adapters (+ mirrored on HF)
archive/       superseded working copies, quarantined with explanation
ENGINE.md      parked blueprint: native multi-GPU MoE diffusion serving ("videng")
NORTH_STAR.md  where this is going; PRESENCE.md governs posting discipline
```

## Environment notes (so nobody repeats this)

- Kaggle GPU pushes default to **P100** unless `"machine_shape": "NvidiaTeslaT4"`
  is pinned — Pascal sm_60 kernels are missing from the default torch cu128 image.
- transformers ≥4.57 raises ImportError at load if any `torchao < 0.16` is
  installed (even unused). Uninstall it at startup.
- T4 (sm_75) has no flash SDPA kernel; `model.generate()` on long prompts falls
  back to the math backend and materializes the full attention matrix → OOM at
  8k. Fix: manual chunked prefill (1024-token slices) + greedy decode loop.
- Kaggle CLI 2.2.4 reads `enable_gpu` from kernel-metadata.json — not `is_gpu`,
  not `accelerator`.
- `datasets>=3.0` dropped loading-script support; pin `datasets==2.21.0` +
  `trust_remote_code=True` for `deepmind/pg19`.
- Any file >10 MB goes to HF, not git history.

## Rules

- Nothing runs locally. Remote only (Kaggle/HF).
- No replications of things done 50 times.
- Pre-register hypotheses before launching; report what the data says.
- One verified claim per post; public corrections when wrong.

## Update log

- **2026-08-24** — **Experiment 2 CLOSED. The 2×2 matrix is complete (40/40 runs)
  and the causal claim is airtight:** retrieval past the window recovers ONLY
  when training happened with the window active AND eval uses it. Control
  weights under the windowed config collapse to 0/5 at 16k+ — identical to
  unadapted weights — so generic LoRA explains nothing. Treatment weights keep
  full stock performance, so there's no trade-off. Full matrix in
  `05-swa-2x2-closeout/`. Both adapters archived in-repo (`artifacts/adapters/`).
- **2026-08-24** — Strategy sessions locked the north star (`NORTH_STAR.md`): the
  one-machine studio — small interrogator brain + native multi-GPU MoE engine +
  big open models. Experiment 3 pre-registered: the **Spec Interrogator**
  (clarify-aware LoRA on SmolLM3, evaluated on ClarifyCodeBench +
  downstream agent deltas, `04-spec-interrogator/EXPERIMENT_3.md`). Remaining
  close-out item: the 2×2 cross-config eval kernel for Experiment 2.
- **2026-08-24** — **Experiment 2 complete, both arms. Treatment: 5/5 NIAH at every
  length (8k→64k) with the windowed config.** Control (stock config): 5/5 everywhere
  as expected. Combined with Experiment 1 (unadapted weights under the same windowed
  config: 0/5 past 8k), this is the causal cell: retrieval under RNoPE-SWA is
  restored by LoRA CPT *with* the window active — ~6.5M tokens, a few GPU-hours.
  P1 confirmed (stronger than predicted), P2 confirmed, P3 falsified: no speed win
  post-adaptation (treatment 15–23% slower than control; peak memory identical).
  Side finding: control's initial CPT loss of 0.665 nats suggests SmolLM3 largely
  memorized pg19 in pretraining; both arms converged to ≈2.7 nats. Data archived in
  `02-rnope-swa-lora/run1_data/` and `03-lora-control/run1_data/`.
- **2026-08-21** — Experiment 1 confirmation run completed on both T4s (`device_map="auto"`):
  40/40 runs, mask parity verified, results identical to the archived data. Experiment 1 is
  officially reproduced.
- **2026-08-21** — Experiment 2 upgraded to a two-arm design: treatment (LoRA with window)
  and control (identical training, stock config) now run **concurrently**, same GPUs, same
  data, same day. Amendment pre-registered in `03-lora-control/EXPERIMENT_2_CONTROL.md`
  with the full interpretation matrix. Two environment bugs found and fixed en route
  (datasets≥3 vs pg19 script; torchao version gate). Also logged: runtime config reports
  `rope_theta=5e6` — a third value for the theta discrepancy file.
- **2026-08-20** — Experiment 1 results in (40/40 runs): inference-only RNoPE-SWA
  destroys retrieval past the window (0/5 at 16k–64k) but saves 11–21% time and
  real memory. Baseline perfect everywhere. Pre-registered H2 falsified cleanly;
  H1 partially confirmed. Data archived in `01-rnope-swa-inference/run1_data/`.
- **2026-08-20** — Experiment 2 (Phase B) launched: LoRA fine-tune with the window
  active during training, followed by the identical NIAH harness. Pre-registration
  in `02-rnope-swa-lora/EXPERIMENT_2.md`.
- **2026-08-20** — Repo created. README covers the project; update log started.