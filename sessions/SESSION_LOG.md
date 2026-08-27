# Session Log

A running pointer to where we are. At the end of every chat, the last 5–10 lines of that session get appended here so the next session can warm up in seconds.

---

## Session 1 — The transformer block, end-to-end

**Date:** (first session)
**File:** `SESSION_1_RECAP.md` for full content
**What we did:** Built the actor / Scrabble / mood-board / notepad / speed-dating / 11,008-desks metaphor stack. Traced one token from embedding through 36 transformer blocks to logits. Located where SFT, LoRA, QLoRA, and DPO live in the picture. Established the four-sentence summary as the spine.

**Where we ended:** Confirmed the two-documentation pattern (profile + recap) and added this session log. About to start Session 2 (loading a model and watching a single forward pass) but battery died first.

**What session 2 will do (do not redo these):**
- Load SmolLM3-3B's tokenizer in Python and run a single token through one block
- See the embedding table, the attention weights matrix, the MLP expansion in actual numbers
- Use the `huggingface_hub` library (works even with stub safetensors for tokenizer-only ops)
- Continue using the actor / notepad / 11,008-desks analogies

**Open threads for session 2:**
- Confirm whether to install `huggingface_hub` (lightweight, only needs `pip install`) or go fully pure-stdlib
- Decide if we want to inspect the real model's weights once the safetensors download finishes, or just use the config-derived architecture

---

## Session 2 — TBD (not yet started)

**Status:** Pre-session. About to begin. Haylee has confirmed she wants the tutor role locked in from this point forward, no coding yet, deep understanding first, eventual goal of AI engineer / researcher. Profile and recap patterns are the standing ruleset.

**Where we ended last chat:** A review pass on the repo flagged (a) `tokenizer.json` is locally broken (1.25M-line diff, almost certainly an encoding artifact), (b) the two safetensors shards are still 135-byte LFS stubs so the model can't actually be loaded yet, (c) the three note files are untracked in git. None of those block the conceptual work of Session 2, so we're not fixing them this turn.

**What session 2 will do (do not redo these):**
- Pick up from the Session 1 closing summary: actor + 5,000 great scenes + 2,000 preference pairs + rehearsal
- Go one level deeper on **a single transformer block** — the mechanics of attention and the MLP, using the actor / notepad / 11,008-desks / speed-dating stack as the spine
- Build intuition for *what numbers actually move* during a forward pass, using SmolLM3-3B's real config (2048 hidden, 16 heads, 4 KV heads, 11008 MLP, 36 layers, NoPE every 4th)
- Goal-anchor every concept back to the 120B fiction fine-tune

**Open threads for session 2:**
- Locked-in: A2 (split into sub-sessions) + B1 (attention first)
- Three sub-sessions planned in order: (1) attention deep, (2) MLP deep, (3) layer norm + residual + end-to-end block
- Where the actual "thinking" happens vs. the actual "knowledge storage" happens — and why that matters for fine-tuning (this is the bridge into Sessions 3–5)

**Sub-session 2a — Attention, deep (CURRENT)**
- Why attention first: Session 1 built the MLP metaphor strong but left the biggest attention gaps undelivered (Q/K/V, dot product, softmax, causal mask, 16 heads mechanically, GQA's actual savings)
- Goal-anchor: this is the part that controls long-range character consistency in the 120B fiction model
- Sequence: (1) what Q/K/V are in the speed-dating metaphor, (2) the dot product as "how relevant is this token to me," (3) the softmax as "turn scores into a recipe," (4) the causal mask, (5) why 16 heads, (6) why 4 KV heads
- **Format locked in by Haylee: SHOW AND TELL, not quiz. I read and explain, anchored in the config, with fresh metaphors. Haylee interrupts with questions whenever something doesn't land. One check-in per number, not five. No "guess first" framing.**
- Original "lab time" framing was overcorrected; Haylee's feedback: "Just because we covered those topics does NOT mean I have an understanding of said topics." Single-pass exposure ≠ ownership. Format revised accordingly.
- **Key correction logged (vocab vs model size):** Haylee's instinct that "bigger model = bigger vocab" is right in *direction* but off in *causation*. Vocab size (pencils in the box) is decoupled from model size (size of hand/desk). A 3B and 70B model can share a vocab. The coupling is asymmetric: a huge vocab *forces* the model to be bigger, but a big model doesn't *require* a huge vocab. Will reference this distinction in future sessions to prevent vocabulary/parameter-count conflation.
- Lab sequence: (1) `config.json` chunk by chunk, (2) relevant README sections, (3) `chat_template.jinja`
- This lab precedes the prose deep-dive on attention; the deep-dive will anchor in the config numbers we've already decoded

---

## RESUME CARD — read this first tomorrow

**Session 2, Sub-session 2a — Config Show and Tell.** Format: I explain, anchored in `config.json`, with fresh metaphors. No code, no installs. One check-in per number.

**Done so far (Numbers 1 and 2 of 8):**
1. `vocab_size: 128256` — the colored pencil set. Number of distinct tokens the model can use. 128k is the Llama-3 multilingual tokenizer. The vocab table is 128k × 2048 = 262M parameters, bigger than the MLP in 4 layers combined. Vocab size and model size are *correlated but decoupled* — bigger vocab forces bigger model, but bigger model doesn't require bigger vocab. Pencils vs hand-and-desk metaphor locked in.
2. `hidden_size: 2048` — the width of the notepad, OR equivalently, "the number of simultaneous questions the model can ask about a token." 2,048 is a hard contract: every component produces/consumes exactly 2,048 numbers. Change this number and *every* parameter count changes. 120B models have wider notepads (typically 4,096–8,192), which is part of why they cost way more than 40× a 3B model.

**Next up — Numbers 3–8, in order:**
3. `num_hidden_layers: 36` — the number of rooms. Why 36, why this is the *most* important number in the config.
4. `num_attention_heads: 16` — the division of labor inside one room. Why 16, what each head does differently.
5. `num_key_value_heads: 4` — GQA, the memory trick. Why 4 instead of 16.
6. `intermediate_size: 11008` — the 11,008 desks, in fresh detail. The 5.4× ratio.
7. `no_rope_layer_interval: 4` and the `no_rope_layers` array — the long-context trick. Where Session 8 will anchor.
8. `tie_word_embeddings: true` — why the embedding table and the final lookup share the same memory.

**Then after Numbers 1–8, the prose deep-dive on attention** (Q/K/V, dot product, softmax, causal mask) which will *use* the numbers we've already walked through.

**Where Haylee is at, end of session 2a-partial:** energized, learning, calls the model "someone with severe anxiety" which is a brilliant unscripted metaphor. Wants to fine-tune, has 6 hours in today, calling it done. Genuine enthusiasm. Not fatigued-confused, just fatigued-fulfilled. Tomorrow should pick up at Number 3.

**Open emotional / motivational note (carry forward):** Haylee said "I didnt realize it was just pure chaos during inference lol. Someone with severe anxiety." This is the kind of self-generated metaphor that indicates the concepts are landing and being *integrated*, not just memorized. Worth protecting that integration pace over speed.

**Next check-in options (pick when you resume):**
1. **A single block, end-to-end** — trace one token through attention + residual + MLP + residual in one sitting, see how a 2,048-dim notepad actually grows
2. **Just attention, deep** — speed-dating, softmax, why 16 heads split the work, why 4 KV heads are shared, what Q/K/V actually *are* in the metaphor
3. **Just the MLP, deep** — the 11,008 desks, what SwiGLU actually does, why this is where *your* 120B story model will store prose style

---

## Session 2 resumed — 24-hour refresher

**Date:** Aug 12, 2026 (approx. 24h after the 6-hour Session 1)

**What we did:** Haylee asked for a brief refresher over all Session 1 content plus config Numbers 1–2 before resuming sub-session 2a. Delivered as the full metaphor-stack recap (actor, Scrabble, mood board, notepad, speed-dating, 11,008 desks, rehearsal/post-its, 120B pipeline) in compact form.

**Where we ended:** Refresher delivered. Covered config Number 3 (`num_hidden_layers: 36`) and the depth-vs-width tradeoff (why fewer-room models exist in the same size class: latency, parallel reads; depth wins for reasoning/coherence; her 120B choice validated — depth bet for nuanced emotional/sarcasm tracking; serving already planned, latency not a concern).

**Then Haylee asked for hands-on practice** — chose the pure-stdlib simulation (option that had been pending). Built `exercise1_attention.py` (no installs, Python 3.12 stdlib only): a 6-token "Sarah told her that she left" sim with 4-dim mood boards, running the full attention step — dot-product scores, softmax recipe, weighted blend, and the causal mask.

**Three teaching moments that landed:**
1. Self-attention dominance (her vs her = +2.0) — why real models scale scores by 1/√d. Flagged as "the one cheat" in our sim (no learned Q/K/V projections, no √d scaling, no heads).
2. Resolution is a nudge not a copy — sarah-ness 0.00 → 0.13; notepad shifts in several directions at once; that's the residual stream's input.
3. Causal mask on screen — "she"/"that"/"left" → 0.000 when processing "her" (the anxiety is real); bonus: masked-out rival causes sarah-ness to rise to 0.245 (redistribution, not just deletion).

**Next options given:** (a) tweak the sim (add √d scaling, projection matrices, real numbers from config), (b) Number 4 (`num_attention_heads: 16`), (c) GQA (Number 5). Awaiting Haylee's pick.

**Standing rules (carried from Session 1, do not violate):**
- No code yet. Prose + analogies + real numbers from `config.json` only.
- One deep-dive per turn, not five.
- Every new term gets an analogy before a definition.
- Always end with 2–3 "which next?" options.
- Always anchor back to the 120B fiction fine-tune.

---

## Session 2 continued — hands-on: the MLP desks (BIG progress)

**Date:** Aug 12, 2026 (same day, later)

**Path this session:**
1. Attention sim built (`exercise1_attention.py`) — see above — then **stripped to its skeleton**: `attention_skeleton.py` (31 lines: data + 3 tiny functions + 3-line recipe). Rule learned: every component = same skeleton, different recipe.
2. **Toured `smollm/text/finetuning/train.py`** — the real fine-tune script, mapped to metaphors: `LoraConfig(r=16, target_modules=["q_proj","v_proj"])` = post-it notes on the attention desks; `BitsAndBytesConfig(load_in_4bit=True)` = QLoRA shorthand; `SFTTrainer` = rehearsal director; knobs = `learning_rate`, `max_steps`, `max_seq_length`. Her future fine-tune = same script, different `--model_id` and `--dataset_name` (+ a DPO stage).
3. **MLP lesson** (`mlp_example.py`): expand → think → collapse. 8 desks (toy `intermediate_size`), each a weight vector; `relu` = desks only write what they saw; `GAIN=0.25` = rooms add small touches.
4. **Her first attempt** (`mlp_haylee.py`): renamed desks with fiction flavors (sarcasm, past-tense...) but kept the example's weights → identical output. Teaching moment: **the name is decoration; the weight vector IS the desk.** In a real model no desk is labeled — names are our stories about learned numbers. Tied to her 120B thesis: sarcasm is never one checkbox, it's a pattern across many dimensions — why she needs depth.

**THE WALL (important — see learning profile for the full lesson):** Haylee hit "assembly paralysis" — understood +/−/0 but not which numbers go in which slot. Two unblocking reframes:
- **Position IS meaning.** The 4 numbers aren't a pile; each slot maps to a box via an ordering contract (slot 0 = female, 1 = verb, 2 = pronoun, 3 = sarah-ness), identical in notepad, desk, and output. Dot product pairs slot-by-slot. This is the "invisible chart" she was missing — it doesn't exist externally; **she is the chart.**
- **Numbers are translations of sentences.** "I care a lot about X" → +2; "I don't care" → 0. Built `DESK_BUILDER_CHEATSHEET.md` as the physical reference.
- **Labels are our invention, not the model's.** Real model = 2,048 unnamed slots; meaning is learned, then interpreted by researchers afterward (dimension-487 idea from Session 1).

**Config tie-ins landed this session:**
- `hidden_size: 2048` = the slot count (notepad width / desk width). Three counts now distinct: slots (2,048) vs desks per chamber (11,008) vs rooms (36).
- The 4 slots in the toy = my readability choice, not a law; once set, a hard contract for the whole setup.
- MLP has **two** sets of numbers: up-matrix (what each desk watches) + down-matrix (how much its note matters to the output) — the source of the 3× in 3×2048×11008 (SwiGLU gate adds the third). Toy simplified with a single GAIN.

**Format evolution:** Haylee has graduated from "no code" — she now runs pure-stdlib exercises, edits scripts herself, and posts output. Still no installs, and no torch/numpy.

**Where we are leaving off:** She has a pending assignment — build 2 desks in `mlp_haylee.py` sentence-first (use the cheatsheet), run, report verdicts. Excitement is high: "we are making REAL headway and I'm really excited."

**Next session options:**
1. Grade her 2 desks (the pending assignment)
2. Finish config Numbers 4–5 (heads + GQA) or 6 (intermediate_size, the desks in real scale)
3. Build the MLP with a real down-matrix (per-desk importance) — her two-question model in action

---

## Session 3 — Research track begins (Aug 15–17, 2026)

**Role shift:** Haylee is now an aspiring AI *researcher* leveraging AI tools + Kaggle for compute. The tutor role continues, but a parallel research track opened. Profile updated accordingly.

**Infrastructure learned (Kaggle REST API, no CLI installed):**
- Token: `~/.kaggle/access_token` (KGAT), account `haylee00`
- Kernel push: code goes in the JSON `text` field of `/api/v1/kernels/push`. An invented `kernelSource` field silently creates an EMPTY kernel (v1 lesson)
- **P100 trap:** default Kaggle image (torch cu128) omits Pascal sm_60 kernels — always pin `"machineShape": "NvidiaTeslaT4"` or first CUDA op dies with `cudaErrorNoKernelImageForDevice`
- Quota endpoint: `/api/v1/kernels/quota` — 30h GPU/week. T4 x2 (dual-GPU machine) available; useless for 3B inference, useful for future QLoRA training
- Kaggle kernels are BATCH jobs — nothing runs between sessions; `complete` = machine gone

**Kernels run:**
1. `smollm3-session2` (pre-existing, retrieved): forward-pass experiment — "it"→"cat" coreference, peak at layer 10 head 13 (0.508). Real attention.png saved to session2/. Its results.json was empty on Kaggle's side (kernel bug).
2. `smollm3-baseline-bf16` v3 (built + run): **E1 baseline — SmolLM3-3B bf16 on T4 via plain transformers.** Clean sweep: T1 tool format 3/3 parsed + 3/3 correct names + sensible args (incl. optional `minutes_from_now: 20` from plain English), T2 no spurious calls, T3 args valid, T4 multi-turn flow end-to-end ("light rain, 12°C"). 16.6 tok/s sanity gen. 6.26GB weights. ~60s wall. results.json saved to kernels/smollm3-baseline-bf16/out/.

**Research direction locked in (see EDGE_SLM_RESEARCH.md):**
- Goal: SmolLM3-3B as companion-agent brain on old 8GB machines — baseline first, then optimize (vLLM levers, INT4, YaRN 128k), then post-train
- CompanionBench v0 spec written (T1 format / T2 selection / T3 args / T4 flow / T5 companion reality; bf16×INT4 × strict×free 2×2)
- Hermes disambiguation logged: Hermes the *harness* = OpenClaw competitor (research TBD); `hermes` the *vLLM parser* = the wire format SmolLM3 natively speaks
- Next queued: **E2 — vLLM inference bake-off kernel** (prefix caching, INT4, KV fp8, n-gram spec decode; TTFT/TPOT/throughput/mem table)

**Standing rules added this session:** NEVER save anything to /tmp (all files in project folder). Write results.json BEFORE printing DONE. Pin NvidiaTeslaT4 on every push.
