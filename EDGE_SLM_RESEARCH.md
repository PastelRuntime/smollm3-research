# Edge SLM Research — SmolLM3-3B as a Companion Agent on Old Machines

**Date:** Aug 16, 2026
**Status:** Planning. Nothing has been run yet. This is the reference doc for the baseline-first research thread.
**Goal:** Make SmolLM3-3B (or its successor) a *genuinely usable* companion-agent brain on an old 8GB computer/GPU — measured, not assumed.

---

## 0. Name disambiguation (read first — we got burned by this)

Three different "Hermes" things. Do not conflate:

| Name | What it actually is | Relation to SmolLM3 |
|---|---|---|
| **Hermes (the harness)** | A personal-companion agent harness. **Competitor to OpenClaw.** A product/platform, separate company, separate everything | None. We want to study what tools it demands of a model |
| **`hermes` (vLLM parser)** | A tool-call *format parser* in vLLM, named after the `<tool_call>` JSON-in-tags format popularized by Nous Research's Hermes model line | **Everything.** SmolLM3's chat template natively emits this format (xml_tools mode). The repo README itself recommends `--tool-call-parser hermes` (README.md:216) |
| **Hermes models (Nous Research)** | A family of fine-tuned LLMs that popularized the format above | Only the format matters to us |

So: the *harness* Hermes is a competitor to study. The *parser* hermes is the wire format SmolLM3 already speaks. Unfortunate name collision — now documented.

---

## 1. The target: what "old 8GB machine" actually means

The busker's corner, not a theater. Four budgets replace "quality at any cost": **RAM, latency, battery, capability.**

Paper math (the D.5 "will it fit" muscle):

| Setup | Weights | KV @ 8k ctx | Overhead | Total | Fits 8GB? |
|---|---|---|---|---|---|
| bf16 3B | 6.2GB | 0.6GB | ~1GB | ~7.8GB | barely, tiny ctx |
| **INT4 3B** | **~1.8GB** | 0.6GB | ~1GB | **~3.4GB** | **yes, room to spare** |
| INT4 8B | ~4.8GB | 0.6GB | ~1GB | ~7GB+ | painfully tight |

KV math: 72KB/token (36 layers × 4 KV heads × 128 dims × 2 (K+V) × 2 bytes). GQA already cut this 4× — MHA would be 288KB/token.

**Key reframe:** INT4-3B is not anemic — it's the *right size* for the target. SmolLM3-3B's BFCL 92.3 nearly ties Qwen3-4B. The open question is whether it *survives a real harness* — multi-turn, 20–50 tools, error recovery. Nobody has published that for this model. That's the research niche.

---

## 2. What the harnesses demand (the requirements side)

### OpenClaw (researched Aug 16, docs.openclaw.ai)
Tool surface (what an agent can be given):
- **Runtime:** `exec`, `process`, `terminal`, `code_execution`
- **Files:** `read`, `write`, `edit`, `apply_patch`
- **Web:** `web_search`, `web_fetch`, `x_search`, `browser`
- **Messaging:** `message`, agent-send
- **Sessions/agents:** `sessions_*`, `subagents`, `agents_list`, goals
- **Automation:** `cron`, `heartbeat_respond`
- **Media:** `image`, `image_generate`, `tts`, video/music generation
- **Human input:** `ask_user`

Three pressure points BFCL never tests:
1. **Tool count.** Companion agents carry 20–60+ tools. Every schema rides in the system prompt. Selection among 50 ≠ selection among 3.
2. **Tool Search / Code Mode.** OpenClaw's answer to schema bloat: give the model `tool_search` / `tool_describe` / `tool_describe` instead of every schema, or have it write compact code against a hidden catalog. A 3B that *searches* well beats one that *memorizes* poorly.
3. **Multi-turn reality.** Tool result returns → model must read it, recover from failures, decide done-vs-call-again. Plus persona/system prompts stacked on top of tools.

### Hermes (the harness) — OPEN TODO
Competitor to OpenClaw. Tool surface NOT yet researched. Before the bench is finalized: fetch its docs, extract its tool list + schema format + multi-turn conventions. (Blocker for T5 fidelity; T1–T4 can proceed without it.)

### vLLM serving facts (verified from docs, Aug 16)
- Serve command for this model (per repo README): `vllm serve HuggingFaceTB/SmolLM3-3B --enable-auto-tool-choice --tool-call-parser hermes`
- **Constrained decoding semantics:**

| `tool_choice` | Schema enforced? | Notes |
|---|---|---|
| named function | always | args guaranteed valid JSON per schema |
| `required` | always | ≥1 call guaranteed |
| `auto` | only if `strict: true` on ≥1 tool | otherwise free text, calls extracted by parser |
| `none` | n/a | no calls |

- Env toggle `VLLM_ENFORCE_STRICT_TOOL_CALLING` (default true).
- vLLM ships a built-in BFCL benchmark (`vllm bench serve`) — use it to validate against the published 92.3 before trusting our own harness.
- Known parser-relevant quirk: SmolLM3's `<think>` blocks — serve with /no_think or a reasoning parser so think-text doesn't pollute tool-call parsing.

---

## 3. CompanionBench v0 — the baseline suite (measure BEFORE optimizing)

Five tiers, ~20 scenarios each, on Kaggle T4, vLLM + hermes parser, temperature 0 throughout (tool calls don't improvise).

| Tier | Tests | Metric |
|---|---|---|
| **T1 Format** | plain calls, empty args, unicode args | parse success % |
| **T2 Selection** | right tool among 3 / 10 / 25 / 50 installed; "no tool needed" refusals | selection accuracy vs tool count — *find the cliff* |
| **T3 Arguments** | enums, optional params, nested objects, plausible distractor tools | arg validity %, hallucinated-param rate |
| **T4 Flow** | 2–4 turn episodes: call → result → react; tool error → recovery; parallel calls | end-to-end success % |
| **T5 Companion reality** | persona + memory + 20 tools + /no_think + 4k+ accumulated context | degradation curve vs context length |

**Two comparison axes (this is what makes it research):**
- **bf16 vs INT4** — does quantization break tool calling before it breaks prose?
- **unconstrained vs `strict: true`** — how much failure is the model vs the plumbing?

Every result lands in a 2×2 table. Deliverables: `results.json` (written BEFORE the DONE print — fix the Session-2 kernel bug) + a "where the 3B breaks" table. That's the before-picture; every later optimization shows up as a delta against it.

**Harness realism notes:** schemas formatted exactly as OpenClaw/Hermes would send them; tool results injected as `tool` role messages; include the "tool returns error" path.

---

## 4. Optimization lever menu (for AFTER the baseline)

### Tier 1 — free (inference-side)
| Lever | Payoff | T4/8GB reality |
|---|---|---|
| Continuous batching + PagedAttention | throughput | automatic in vLLM |
| **Prefix caching** (`--enable-prefix-caching`) | huge for agents — tool schemas stay cached across turns | flag |
| INT4 weights (AWQ/GPTQ) | 6.2GB → ~1.8GB | works on Turing |
| KV cache fp8 (`--kv-cache-dtype fp8`) | halves KV | needs validation on T4 — unknown |
| **N-gram speculative decoding** | 2–3× decode, no draft model needed — great for repetitive prose | free option |
| Chunked prefill | long-ctx latency | flag |
| MInference sparse prefill | up to 10× prefill at 128k | research add-on |

### Tier 2 — post-training (the fine-tune territory)
1. Multi-turn agentic tool data with *executable validation* (smoltalk pipelines in-repo are the factory)
2. RL with executable rewards, GRPO→DAPO (incl. overlong penalty — documented /no_think reward-hack failure mode)
3. On-policy distillation (beats RL at ~1/10 compute under 30B)
4. Watch the silent-bug class (`custom_instructions=None` dropped system messages — tools live in system messages)
5. Merge-to-repair (0.9/0.1 with mid-train ckpt recovered RULER)

### Tier 3 — retrain/architecture (frontier)
RNoPE-SWA (sliding window on RoPE layers only — untested on SmolLM3, genuine novel experiment), per-head NoPE temperature, smaller vocab (~64k would cut the 262M-param embedding tax + shrink every softmax). Already rejected, don't revisit: QK-norm, untied embeddings, MoE, Muon.

---

## 5. Context extension (YaRN 64k → 128k)

Pure config, zero training:

```json
"rope_scaling": { "factor": 2.0, "original_max_position_embeddings": 65536, "type": "yarn" }
```

The catch — seats got cheaper to *address*, not cheaper to *remember at*:

| Setup | Weights | KV @ 128k | One 16GB T4? |
|---|---|---|---|
| bf16 + bf16 KV | 6.2GB | 9.2GB | **no** (15.4GB) |
| INT4 + bf16 KV | 1.7GB | 9.2GB | barely, tiny batch |
| INT4 + fp8 KV | 1.7GB | 4.6GB | **comfortable** |

"128k on the edge" is *only* a quantization story — test, don't assert. Quality caveats: base RULER 128k = 61.03; post-training degraded long context until the 0.9/0.1 merge fixed it. Never extend context without a needle-in-haystack test.

---

## 6. Experiment queue

| # | Experiment | Kernel | Deliverable | Status |
|---|---|---|---|---|
| E0 | Validate harness against published BFCL 92.3 | one | sanity number | pending |
| **E1** | **CompanionBench baseline** (bf16 vs INT4 × strict vs free) | one | the before-picture 2×2 table | **partial: bf16 smoke suite DONE Aug 17** — see below |
| E2 | vLLM lever bake-off (prefix caching, quant, spec decode, KV fp8) | one | TTFT/TPOT/throughput/mem table | **next up** |
| E3 | YaRN 128k + needle-in-haystack at depths 10k/40k/80k/120k | one | retrieval-vs-depth curve | pending |
| E4 | Hermes harness research → fold requirements into CompanionBench v1 | docs only | updated T5 | pending |

### E1 baseline results (Aug 17, 2026 — `kernels/smollm3-baseline-bf16/out/results.json`)
Plain transformers, bf16, Tesla T4, /no_think, xml_tools template path, temp 0:
- **Tool calling clean sweep:** T1 3/3 parsed, 3/3 correct names, args valid (incl. optional param filled from plain English: "in 20 minutes" → `minutes_from_now: 20`); T2 no spurious calls; T4 multi-turn flow end-to-end (read tool result, answered in natural language)
- 16.6 tok/s sanity generation (slow reference — plain transformers, no vLLM)
- 6.26 GB weights / 6.35 GB peak; params 3,075,098,624 confirmed again
- Caveat: this was the 5-prompt smoke suite, not the full ~100-scenario CompanionBench. Full bench still pending.

### Kaggle operational notes (hard-won)
- Push code via JSON `text` field. `kernelSource` = silent empty kernel.
- **Always pin `"machineShape": "NvidiaTeslaT4"`** — default P100 assignment dies (torch cu128 has no sm_60 kernels).
- Quota: 30h GPU/wk (`/api/v1/kernels/quota`), refreshes Sundays. T4 x2 shape exists (dual GPU) — irrelevant for 3B inference, relevant for QLoRA training later.

Rule carried from Session 2's kernel: write `results.json` before printing DONE.

---

## 7. Open questions / decisions pending

- Hermes harness tool surface (blocks T5 fidelity only)
- Does vLLM fp8 KV actually work on Turing T4s? (flag as unknown until tested)
- Which INT4 flavor — AWQ or GPTQ — for the bake-off? (AWQ first guess)
- Does /no_think + hermes parser interact cleanly, or does think-text leak into parsing?
- Base or Instruct checkpoint for the bench? (Instruct — it's the tool-trained one)

---

## 8. Sources

- vLLM tool-calling docs (fetched Aug 16, 2026): https://docs.vllm.ai/en/latest/features/tool_calling.html
- OpenClaw tools overview (fetched Aug 16, 2026): https://docs.openclaw.ai/tools
- Repo: `README.md` (BFCL 92.3, hermes parser recommendation, YaRN block), `chat_template.jinja` (xml_tools format), `RESEARCH_REPORT.md` (all training/architecture claims), `generation_config.json` (0.6/0.95)
- Kaggle: `haylee00/smollm3-session2` (prior forward-pass run, session2/kernel_log.txt)
