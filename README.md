# SmolLM3 Research

Independent experiments on [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) —
Hugging Face's 3B hybrid long-context model. Everything runs remotely (Kaggle/HF),
nothing local. Every experiment is pre-registered before launch.

## The model

SmolLM3-3B is a 36-layer hybrid: **27 RoPE layers + 9 NoPE layers** (no positional
embedding, every 4th layer), 64k native context (128k with YaRN), dual `/think` and
`/no_think` modes, tied embeddings, GQA. The NoPE-hybrid design comes from the RNoPE
paper (arXiv:2501.18795), which claims the combo gives long-context retrieval *and*
memory savings — but nobody had tested the recipe on this actual model. That's the gap
this project attacks.

## Findings so far

### Finding 1 — the `layer_types` inversion in transformers

transformers' SmolLM3 config auto-derives `layer_types` when unset: it assigns
`sliding_attention` to the **NoPE** layers (`configuration_smollm3.py`, the
`not has_rope` branch) — the **opposite** of the RNoPE recipe, which windows the
RoPE layers and keeps NoPE full-attention. Anyone who "just enables SWA" gets the
wrong layers windowed and a misleading result. All experiments here pass
`layer_types` explicitly.

### Finding 2 — RNoPE-SWA at inference: fast, but retrieval dies past the window

**Experiment 1** (`experiments/01-rnope-swa-inference/`): windowed the 27 RoPE layers
at 8k (NoPE kept full), then ran needle-in-haystack at 8k–64k × 5 depths vs baseline,
identical everything else. 40/40 runs completed.

| length | baseline acc | windowed acc | windowed speed | windowed mem |
|--------|-------------|--------------|----------------|--------------|
| 8k | 5/5 | 5/5 (parity ✓) | −4% | same |
| 16k | 5/5 | **0/5** | **−21%** | −0.15 GB |
| 32k | 5/5 | **0/5** | **−18%** | −0.37 GB |
| 64k | 5/5 | **0/5** | **−11%** | −0.70 GB |

Interpretation: you cannot bolt the paper's recipe onto an already-trained model.
Retrieval lives in the NoPE layers, but the RoPE layers' full-attention contribution
beyond 8k matters at inference — kill it and long-context recall collapses. The
11–21% speedup and memory savings are real, which is exactly the incentive to test
train-time windowing. Full pre-registration + data in the experiment folder.

## Running now

- **Phase A confirmation** — re-run of Experiment 1 with the model split across both
  T4s (`device_map="auto"`), for a clean green run of the archived results.
- **Experiment 2 / Phase B** (`experiments/02-rnope-swa-lora/`): LoRA fine-tune
  **with the window active during training** (r=32, ~6.5M tokens of pg19 at 16k
  seq len — 2× the window, forcing far-context through NoPE layers), then the exact
  Experiment 1 harness re-runs. Pre-registered: P1 = retrieval recovers to ≥4/5 at
  16k & 32k; P2 = 8k stays 5/5; P3 = speed win persists. If P1 holds, a control
  kernel (same training, no window) isolates whether the window itself caused
  recovery. Both outcomes are publishable.

## Repo layout

```
experiments/
  01-rnope-swa-inference/   Experiment 1: inference-only SWA (results in)
  02-rnope-swa-lora/        Experiment 2, treatment arm: LoRA fine-tune with window (running)
  03-lora-control/          Experiment 2, control arm: identical training, stock config (running)
```

Each experiment folder holds: the kernel script, `kernel-metadata.json` (Kaggle push
config), the pre-registration doc, and archived run data (`results.json`).

## Environment notes (so nobody repeats this)

- Kaggle GPU pushes default to **P100** unless `machine_shape: NvidiaTeslaT4` is set
  in `kernel-metadata.json`. The image's torch (cu128, sm_70+) cannot run Pascal at all.
- On T4 (sm_75), SDPA has no flash kernel; `model.generate()` on long prompts falls
  back to the math backend and materializes the full attention score matrix → OOM at
  8k tokens. Fix: manual chunked prefill (1024-token slices) + greedy decode loop.
- Kaggle CLI 2.2.4 reads `enable_gpu` from kernel-metadata.json — not `is_gpu`, not
  `accelerator`.
- The T4 image ships `torchao==0.10.0`; transformers 4.57.6 raises ImportError at
  model load if any torchao < 0.16 is installed (even unused). Uninstall it at startup.
- `datasets>=3.0` removed loading-script support; `deepmind/pg19` still uses a script.
  Pin `datasets==2.21.0` + `trust_remote_code=True`.

## Rules

- Nothing runs locally. Remote only (Kaggle/HF).
- No replications of things done 50 times.
- Pre-register hypotheses before launching; report what the data says.
- One verified claim per post; public corrections when wrong.

---

## Update log

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