# EXPERIMENT 3 — The Spec Interrogator: clarify-aware LoRA on SmolLM3-3B (pre-registered)

Date: 2026-08-24
Status: **pre-registered before any training runs**
Code: `experiments/04-spec-interrogator/` (to be created at push time)

## Motivation

Three years of vibe coding, distilled: the prompt is the spec, and the spec is the
product. Agents amplify discipline, not intent. The gap between "build me a login
page" and production is not intelligence — it's *interrogation*: the questions a
senior engineer asks before writing code.

The field agrees: ClarifyGPT (FSE'24), ClarifyCoder (2025, proved clarify-aware
fine-tuning works), HumanEvalComm V2, Code ClarQA, and ClarifyCodeBench (July
2026, 419 ambiguous tasks, multi-round interactive protocol). **All of it targets
frontier models. Nobody has tested whether a sub-5B can learn interrogation** —
the one job cheap enough to run all day on an 8th-gen iGPU while the expensive
agent sleeps.

## Termination & completeness criteria (operationalizing the subjective)

"Questions until the spec is full" is subjective as stated. All three layers below
replace it with mechanical criteria, locked before training:

1. **Ground-truth coverage (eval).** ClarifyCodeBench tasks carry planted,
   known ambiguity points. Coverage = LLM-judge-matched ambiguities resolved /
   total planted. Termination = all points covered OR round cap (N=8). Both
   failure modes are scored: under-asking (low coverage) and over-asking
   (turns per ambiguity). Judge subjectivity mitigated by: majority vote of 3
   judge passes, exact-match check first for canonical question forms, and a
   human-spot-checked calibration sample (~50 dialogues).
2. **Schema-validated spec (output).** The terminal spec must pass a fixed JSON
   schema — required sections: goal, scope in/out, data model, interfaces, error
   handling, auth/security, edge cases, tests-as-acceptance, open questions.
   Validation is deterministic code. Missing section → rejected → interrogation
   continues. The same checklist that guides questioning defines completeness.
3. **Downstream pass-rate (I3, the backstop).** The spec is sufficient iff the
   same coding model with the interrogated spec beats the same model with the
   raw prompt on tests-passed. No judgment involved.

Reported metrics: coverage %, turns-to-coverage, schema-validity rate on first
terminal attempt, and the I3 delta.

## Design

- Model: HuggingFaceTB/SmolLM3-3B, LoRA (r=32, α=64, q/k/v/o), fp16, 2×T4 Kaggle
- Role: **interrogator, not coder.** Input: vague task. Output: clarifying
  questions, one per turn, until it can emit a full engineering spec document
  (goals, constraints, data model, edge cases, tests-as-definition-of-done).
  The spec then feeds any downstream agent.
- Data (three sources):
  1. ClarifyCoder's synthetic ambiguity-injection recipe over a programming
     corpus, seeded with HumanEvalComm + Code ClarQA formats
  2. Llama-3.1-8B distillation — shares SmolLM3's tokenizer, so (rough prompt →
     interrogation dialogue → final spec) tuples generated on free Kaggle GPUs
     transfer losslessly
  3. This repo's own artifacts (EXPERIMENT docs, update logs) as format seeds
- Context injection: engineering checklists (security, testing, deployment,
  edge cases) live in the prompt at inference, not the weights — competence in
  weights, catalog in context.

## Pre-registered hypotheses

| # | Hypothesis | Falsified if |
|---|-----------|--------------|
| I1 | Fine-tuned SmolLM3 asks relevant clarification questions (LLM-judge match rate vs ambiguity points) at ≥60% of ClarifyCodeBench's frontier baseline | <40% of frontier match rate |
| I2 | Clarify-awareness does not destroy general ability: MMLU/GSM8K subset drop <5 points | ≥5 point drop |
| I3 | The amplified spec improves downstream codegen: same coding model, raw prompt vs interrogated spec, on HumanEval-style tasks — pass rate improves by ≥8 points | ≤3 points improvement |
| I4 | Runs conversational interrogation on CPU/integrated-class hardware at usable speed (<3s/turn at Q4) | unusable latency |

Secondary: does SmolLM3's dual-mode (/think for planning questions, /no_think
for rapid-fire) outperform single-mode interrogation? Recorded, not gating.

## Decision rules

- I1+I3 hold → the artifact ships (GGUF + Ollama), paper draft: "Small Models
  Can Interrogate: A 3B Spec Amplifier for Agentic Coding"
- I1 holds, I3 fails → interrogation transfers but specs don't help agents;
  publish the negative with dialogue analysis
- I1 fails → capability doesn't compress to 3B at this budget; clean negative,
  pivot to 4B-class candidates (Qwen3-4B, Phi-4-mini) for the same test

## Post-run checklist

1. `kaggle kernels status haylee00/smollm3-spec-interrogator`
2. Archive results.json → `run1_data/`
3. Update README update log + X post with actual numbers
4. Close out Experiment 2's 2×2 kernel before launching this (founding receipt first)
