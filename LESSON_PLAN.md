# LESSON PLAN — Everything LLM & Fine-Tuning

Built for Haylee. ADHD-format: small chunks, frequent wins, visible progress, one lesson per chat.
This file is the syllabus AND the scoreboard. We update the tracker as we go.

---

## How this works (the standing rules)

1. **One lesson per chat.** No marathons. Each lesson = one mental model + one tiny experiment.
2. **Mastery gate.** We do not move on until you can pass the "You own this when..." check — usually explaining it back to me in the metaphor stack, or your lab passing the rubric.
3. **Every lesson has the same shape:**
   - **READ** (book learning, ~10–20 min): prose + metaphors, anchored in a real file in this repo.
   - **DO** (hands-on, ~20–30 min): a pure-stdlib Python lab or a paper exercise. No installs unless you decide.
   - **WIN**: the "I just did a thing" moment, marked explicitly.
   - **OWN IT**: the mastery check + grading rubric (runs clean · recipe intact · metaphors used correctly).
4. **Breaks are part of the plan.** ~25-minute blocks. When you feel fog rolling in, we stop mid-lesson and mark the exact resume line here.
5. **Parking lot.** Tangents get written in the Parking Lot section at the bottom, not chased.
6. **Every lesson ends with 2–3 "which next?" options** where branching exists.
7. **Everything anchors back to the 120B fiction fine-tune.** No concept floats free.

## The metaphor stack (our shared language — from Session 1)

- **Actor** = the model. Pre-training = reading everything. Fine-tuning = rehearsal.
- **Scrabble bag** = the tokenizer's 128,256 pieces.
- **Mood board** = a token's embedding (2,048 numbers).
- **Notepad** = the residual stream; carried through 36 rooms; only ever added to.
- **Speed-dating** = attention; Q asks, K advertises, V is what you take home.
- **11,008 desks** = the MLP thinking chamber; each desk is a weight vector.
- **Post-it notes** = LoRA adapters stuck on frozen desks.
- **Recipe** = softmax output; how much of each token's notepad to blend.

## Progress Tracker

| Phase | Lesson | Title | Status |
|---|---|---|---|
| 0 | 0.1 | Repo map + how this plan works | ⬜ |
| A | A.0 | Warm-up: grade the 2-desk assignment | ⬜ pending submission |
| A | A.1 | Config #4 — `num_attention_heads: 16` | ⬜ |
| A | A.2 | Config #5 — `num_key_value_heads: 4` (GQA) | ⬜ |
| A | A.3 | Config #6 — `intermediate_size: 11008` + the down-matrix | ⬜ |
| A | A.4 | Config #7 — NoPE / RoPE (`no_rope_layers`) | ⬜ |
| A | A.5 | Config #8 — `tie_word_embeddings` + leftovers | ⬜ |
| A | A.6 | CAPSTONE: hand-count 3,075,098,624 parameters | ⬜ |
| B | B.1 | Q/K/V projections (removing "the one cheat") | ⬜ |
| B | B.2 | √d scaling (why her-vs-her hogs attention) | ⬜ |
| B | B.3 | RMSNorm + one full block, end to end | ⬜ |
| B | B.4 | Stacking rooms (early/middle/late layers) | ⬜ |
| C | C.1 | Tokenizer internals (`tokenizer.json`) | ⬜ |
| C | C.2 | Chat template (`chat_template.jinja`) | ⬜ |
| C | C.3 | Sampling (`generation_config.json`) | ⬜ |
| D | D.1 | Loss = surprise (cross-entropy) | ⬜ |
| D | D.2 | Gradients + one optimizer step (why 3× memory) | ⬜ |
| D | D.3 | Read `train.py` line by line | ⬜ |
| D | D.4 | LoRA math (rank, why low-rank works, merging) | ⬜ |
| D | D.5 | QLoRA + the memory budget | ⬜ |
| D | D.6 | Data for fiction (building scenes that don't suck) | ⬜ |
| D | D.7 | DPO — preference pairs | ⬜ |
| E | E.1 | Long context (YaRN, 64k → 128k) | ⬜ |
| E | E.2 | Evaluation (loss curves vs benchmarks vs eyeballing) | ⬜ |
| E | E.3 | Serving (vLLM, llama.cpp, GGUF) | ⬜ |
| E | E.4 | FINAL CAPSTONE: `HAYLEE_120B_PLAN.md` | ⬜ |

Already owned (from Sessions 1–2, do not redo): the actor metaphor stack · tokens/Scrabble · embeddings/mood boards · residual stream/notepad · transformer block overview · `vocab_size` (#1) · `hidden_size` (#2) · `num_hidden_layers` (#3) + depth-vs-width · dot product scoring · softmax recipe · causal mask · desk = weight vector · slot-ordering contract · sentence→numbers translation · MLP expand→think→collapse (toy, single GAIN).

---

# PHASE 0 — Orientation

## Lesson 0.1 — Repo map + how this plan works
- **Anchor:** the repo is the textbook; every file is a chapter.
- **Goal anchor:** by the end, every file in this folder will be something you could explain to another person — that's what "AI engineer" actually means day-to-day.
- **READ:** this plan's intro + tracker. Skim the repo file listing.
- **DO:** the File Labeling exercise — for each file below, write a one-line "what it's for" in your own words (metaphors encouraged). I grade against the answer key.
  - `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`, `model.safetensors.index.json`, `README.md`, `notebook.ipynb`, `smollm/text/finetuning/train.py`, your 4 exercise files, your 4 notes files.
- **WIN:** the whole folder goes from "a download" to "my textbook."
- **OWN IT:** point at any file and say which phase of the plan uses it. Rubric: every label one line · uses a metaphor or a job description · no "I don't know" without a guess.

---

# PHASE A — Inside the block (finish the config, then count everything)

## Lesson A.0 — Warm-up: grade the 2-desk assignment
- **Status note:** pending from Session 2 — you were building 2 desks in `mlp_haylee.py` sentence-first using `DESK_BUILDER_CHEATSHEET.md`.
- **READ:** your cheatsheet (2 min refresher).
- **DO:** finish/fix the 2 desks, delete the duplicated `NOTEPAD = ...` line at the top of `mlp_haylee.py`, run, paste output.
- **WIN:** your first self-authored weight vectors producing real output.
- **OWN IT (rubric):** runs clean · 3-step process followed (sentence → translate → place) · you can say out loud what each desk's numbers *mean* · bonus: explain why renaming a desk without changing weights changes nothing.

## Lesson A.1 — Config #4: `num_attention_heads: 16`
- **Anchor:** speed-dating with 16 simultaneous rounds — each round asks a different *kind* of question (syntax, coreference, recency, similarity).
- **Goal anchor:** head specialization is what tracks "which character is speaking" across 10k tokens of your fiction. Fine-tuning nudges these specialists.
- **READ:** SESSION_1_RECAP §6 "The 16 heads" + the config line.
- **DO:** extend `attention_skeleton.py` into `heads_skeleton.py` — split the 4-dim mood boards into two 2-dim halves; run the full attention recipe on each half separately; concatenate the two results. Observe: each head can disagree about who's relevant.
- **WIN:** you watch two heads give *different* recipes for the same token.
- **OWN IT:** explain why 16 small rounds beat 1 giant round (different questions in parallel, then concat). Rubric: runs clean · both halves use the same 3-step recipe · your explanation names at least two head "specialties."

## Lesson A.2 — Config #5: `num_key_value_heads: 4` (GQA)
- **Anchor:** 16 speed-daters but only 4 booths — groups of 4 daters share one booth's labels.
- **Goal anchor:** GQA is why your future 120B model's KV cache fits in memory at long context. This number directly controls your serving costs.
- **READ:** SESSION_1_RECAP §6 "GQA — the memory trick" + config.
- **DO:** paper math first — per token per layer, full MHA stores 16 heads × 128 dims × 2 (K and V) = 4,096 numbers; GQA stores 4 × 128 × 2 = 1,024. Then in `heads_skeleton.py`, make both heads share ONE half's board as their K/V. See that output barely changes.
- **WIN:** you computed the 4× memory saving yourself.
- **OWN IT:** explain what GQA shares, what it saves, and why quality barely drops (the model learned redundancy). 

## Lesson A.3 — Config #6: `intermediate_size: 11008` + the down-matrix
- **Anchor:** the 11,008 desks at real scale — and the missing second half of the lesson: each desk's note has an *importance* (the down-matrix), not just a flat GAIN.
- **Goal anchor:** the MLP is where prose style lives. Your SFT will mostly rewrite desks. The down-matrix is why a desk can scream or whisper.
- **READ:** SESSION_1_RECAP §6 MLP + the Session-2 MLP notes in SESSION_LOG.
- **DO:** rebuild `mlp_example.py` as `mlp_real.py` — each desk gets TWO vectors: `watch` (what it looks for) and `shout` (which output dims its note lands in, and how loud). Collapse = sum of `relu(dot(watch, notepad)) × shout`. Then SwiGLU in one sentence: a third "gate" vector multiplies the note by 0..1 before it counts — that's the third matrix in 3 × 2048 × 11008.
- **WIN:** the toy now has the same *shape* as the real 67.6M-parameter chamber.
- **OWN IT:** explain the three matrices (gate/up/down) as watch/write/shout; explain why 11,008 ≈ 5.4× of 2,048. Rubric: runs clean · two vectors per desk · collapse sums per-dim correctly.

## Lesson A.4 — Config #7: NoPE / RoPE (`no_rope_layer_interval: 4`, `no_rope_layers: [1,1,1,0,...]`)
- **Anchor:** RoPE = every Q/K vector gets rotated by an angle proportional to its seat number at the speed-dating event — so dot products encode *distance apart*, not absolute seat numbers. NoPE layers = no seat numbers at all; pure content matching.
- **Goal anchor:** this 3:1 pattern is why your story model stays coherent at 30k tokens of accumulated plot. Session E.1 builds directly on this.
- **READ:** SESSION_1_RECAP §6 RoPE paragraph + the two config fields. Confirm the pattern: 0s at layers 3,7,11,...,35 → 9 RoPE layers, 27 NoPE.
- **DO:** `rope_demo.py` (stdlib) — take 2D vectors, rotate K by angle θ×position, show dot(q,k) depends on the *difference* of positions. Then mark up the `no_rope_layers` array by hand.
- **WIN:** you see "distance = angle" happen in numbers.
- **OWN IT:** explain why rotation encodes relative distance, and what the NoPE layers contribute (global content matching that doesn't decay with distance).

## Lesson A.5 — Config #8: `tie_word_embeddings: true` + leftovers
- **Anchor:** the same pencil set used to read AND to write — the embedding table and the final token-chooser are one shared spreadsheet.
- **Goal anchor:** tying saves 262M params — at 120B scale this trick is the difference between "fits" and "doesn't."
- **READ:** `config.json` remaining fields + `special_tokens_map.json` (pad = eos = `<|im_end|>`: one token, two jobs).
- **DO:** verify in `model.safetensors.index.json` that there is NO `lm_head` tensor (proof of tying). Compute the embedding table size by hand (128,256 × 2,048). Quick-fire round on leftovers: `rope_theta: 5000000.0` (rotation speed — big = long-context friendly), `rms_norm_eps: 1e-06` (anti-divide-by-zero shim), `attention_bias: false` / `mlp_bias: false` (no "+c" terms — Llama-style cleanliness).
- **WIN:** the entire config.json has zero unexplained lines.
- **OWN IT:** explain tying, and why pad=eos is safe for generation but matters for training batches.

## Lesson A.6 — CAPSTONE: hand-count 3,075,098,624 parameters
- **Anchor:** the weight map is a receipt; you're auditing it.
- **Goal anchor:** when you pick your 120B base, you'll do this exact audit to predict memory before spending a dollar.
- **READ:** `model.safetensors.index.json` — 326 tensors, per-layer names: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, two norms per layer, plus `embed_tokens` and final `norm`.
- **DO:** paper/spreadsheet math. Per layer: attention 2048×2048 ×2 + 2048×512 ×2 = 10.49M; MLP 3 × 2048×11008 = 67.63M; norms 4K → 78.12M × 36 layers. Plus embedding 262.7M + final norm 2K. Target: **3,075,098,624 — it matches exactly.**
- **WIN:** your hand count equals the metadata to the digit. This is a top-3 dopamine moment of the whole course.
- **OWN IT:** given any config change ("what if hidden_size were 4096?"), say which tensors grow and roughly by how much.

---

# PHASE B — Attention & MLP mechanics, for real (toy sizes, real math)

## Lesson B.1 — Q/K/V projections (removing "the one cheat")
- **Anchor:** in the sim so far, Q = K = V = the mood board — tokens ask questions with the same face they advertise with. Real models learn three *costumes*: W_q (how I ask), W_k (how I advertise), W_v (what I actually hand over).
- **Goal anchor:** `q_proj`/`v_proj` are the exact targets in `train.py`'s LoRA config — you're about to understand what the post-its stick onto.
- **READ:** SESSION_LOG "the one cheat" note + the tensor names in index.json (`self_attn.q_proj.weight` etc.).
- **DO:** `projections_skeleton.py` — give each token three 4×4 hand-set matrices; compute q = W_q·board, k = W_k·board, v = V·board; rerun the whole speed-dating recipe with q·k scores and a blend of **v**s. Observe: who-you-ask-about and what-you-take are now different things.
- **WIN:** the sim is now mechanically identical to real attention (just tiny).
- **OWN IT:** explain what each projection learns and why separating ask/advertise/hand-over makes the model more expressive. Rubric: runs clean · scores use q·k · blend uses v.

## Lesson B.2 — √d scaling
- **Anchor:** dot products grow with vector width like gossip grows with crowd size; dividing by √d keeps the volume constant.
- **READ:** your `exercise1_attention.py` output — remember her-vs-her = +2.0 dominating.
- **DO:** add `score / sqrt(d_k)` to the sim (d_k = head width); watch the recipe flatten from "self-hogging" toward balanced. Try d=4 vs d=2048 to feel why it matters at real scale.
- **WIN:** you fixed the exact pathology you spotted in Session 2.
- **OWN IT:** explain what softmax does to big scores (winner-take-all) and why that's bad for gradients.

## Lesson B.3 — RMSNorm + one full block, end to end
- **Anchor:** the notepad's handwriting gets messier every room; RMSNorm is the librarian who re-flattens the pages to standard thickness *before* each sub-room reads it — without erasing any notes.
- **Goal anchor:** this is THE block recipe. When you read modeling code (or debug a fine-tune that produces garbage), this order is what you check.
- **READ:** config `rms_norm_eps` + recap §5 (residuals).
- **DO:** `block.py` — the full recipe on toy data: norm → attention → **add residual** → norm → MLP → **add residual**. Print the notepad after every stage.
- **WIN:** one token, one complete room, every stage visible. This is the picture from Session 1 made of *your* code.
- **OWN IT:** recite the recipe order from memory and explain why the adds (not replaces) make 36 rooms stackable. Rubric: runs clean · residuals add pre-sub-room values · norm applied before, not after.

## Lesson B.4 — Stacking rooms (early/middle/late layers)
- **Anchor:** 36 rooms as an assembly line: early rooms do spelling and grammar, middle rooms track who's who, late rooms plan what to say.
- **Goal anchor:** layer specialization is why some fine-tuners only LoRA the late layers for style — a real option for your 120B run.
- **READ:** SESSION_1_RECAP §7.
- **DO:** loop `block.py` 3 times; watch the notepad accumulate. Paper discussion: what changes if room 2 is deleted vs room 35?
- **WIN:** the notepad after 3 rooms visibly carries all three rooms' notes.
- **OWN IT:** explain the rough early/middle/late division and why it's emergent, not assigned.

---

# PHASE C — The text pipeline (what the model actually sees and says)

## Lesson C.1 — Tokenizer internals
- **Anchor:** the Scrabble bag, opened up: `vocab` (piece → ID), `merges` (the recipe for gluing small pieces into big ones), `added_tokens` (the 256 special pieces like `<|im_start|>`).
- **Goal anchor:** your fiction dataset lives or dies by tokenization — dialogue punctuation, em-dashes, ellipses all chop differently. Dataset lesson D.6 depends on this.
- **READ:** `tokenizer_config.json` (structure only) + SESSION_1_RECAP §3.
- **DO:** `tokenizer_explore.py` (stdlib json) — load `tokenizer.json`; find your own name's pieces; chop 3 sentences by hand using greedy longest-match; confirm 128,000 vocab + 256 added = 128,256 = `vocab_size`.
- **WIN:** you hand-tokenized a sentence the same way the model does.
- **OWN IT:** explain BPE greedy longest-match, why " Paris" ≠ "Paris", and where special tokens live.

## Lesson C.2 — Chat template
- **Anchor:** the template is the stage directions printed around the script — the actor never sees your message list, only one long string with `<|im_start|>`/`<|im_end|>` scaffolding.
- **Goal anchor:** reasoning-mode fiction tricks, system-prompt personas, and training-data formatting all come through this file. Getting it wrong = training on garbage formatting.
- **READ:** `chat_template.jinja`, chunk by chunk: think/no_think switch, the auto metadata header (knowledge cutoff, today's date), `/system_override`, the empty `<think></think>` injection in no-think mode (lines 80, 92), tool blocks.
- **DO:** paper first — hand-render the exact string for `[system: /no_think, user: "Hi"]`. Then `template_render.py` (stdlib): rebuild that string with f-strings, matching the jinja logic by eye.
- **WIN:** you can predict, character for character, what the model sees.
- **OWN IT:** explain why no-think mode injects an *empty* think block (the model starts writing after it — reasoning is skipped by construction, not by willpower).

## Lesson C.3 — Sampling
- **Anchor:** the final lookup gives 128,256 scores; sampling is the actor's improv knob — temperature = how wild the improv gets, top-p = how big the allowed suggestion pile is.
- **READ:** `generation_config.json` (temperature 0.6, top_p 0.95, do_sample true) + the README tip.
- **DO:** `sampler.py` (stdlib) — 10 fake token scores; apply temperature; softmax; top-p filter; sample 20 times with `random`. Compare t=0.2 vs t=1.5 tallies. Then greedy (argmax) for contrast.
- **WIN:** you watch the same scores produce boring vs chaotic text, controlled by two numbers.
- **OWN IT:** explain why 0.6/0.95 is a sane default for fiction, and what t→0 approximates (greedy).

---

# PHASE D — Training (where the post-its come from)

## Lesson D.1 — Loss = surprise
- **Anchor:** after every guess the actor makes, we check the script's real next token; loss = how surprised they were, averaged over the whole rehearsal. Low surprise = good performance.
- **READ:** nothing new — this lesson converts Phase B/C into the training signal.
- **DO:** `loss.py` (stdlib) — 5 fake logits, one "true" token; softmax; loss = −ln(probability of truth). Run it 4 times, raising the true logit each time; watch loss fall from ~2.3 toward ~0.
- **WIN:** the abstract word "loss" becomes one line of arithmetic you wrote.
- **OWN IT:** explain what loss 2.1 vs 0.7 means and why we average over tokens.

## Lesson D.2 — Gradients + one optimizer step (why 3× memory)
- **Anchor:** the optimizer is the rehearsal director with TWO notepads per desk: "which way have I been nudging this weight" (momentum) and "how jumpy have the nudges been" (variance). Weights + 2 notepads = 3× memory.
- **READ:** SESSION_1_RECAP §8 full fine-tuning paragraph.
- **DO:** `one_step.py` (stdlib) — one weight `w`, loss = (w×2 − 6)²; wiggle w by ±0.01 to feel the slope (finite difference); take one step downhill; repeat 10 times; watch w → 3. Then the memory math: 3.075B params × 2 bytes (bf16) ≈ 6.2GB weights → ~18.6GB with Adam's two notepads, before activations.
- **WIN:** you performed gradient descent by hand. No black box left.
- **OWN IT:** explain momentum and variance in director language, and recite the 3× memory rule with the activation caveat.

## Lesson D.3 — Read `train.py` line by line
- **Anchor:** this script is the rehearsal hall's control panel — every knob you've learned has a physical switch here.
- **READ:** `smollm/text/finetuning/train.py` — you toured it in Session 2; now we annotate.
- **DO:** copy to `train_annotated.py`; write a comment on every block: args (the knobs), `LoraConfig` (post-it placement: r=16, q_proj/v_proj), `BitsAndBytesConfig` (QLoRA shorthand), model load, dataset load, `SFTConfig` (learning_rate, max_steps, cosine schedule, warmup, paged_adamw_8bit), `trainer.train()`. Flag the gotchas: `push_to_hub=True` and `report_to="wandb"` will nag on Colab; defaults point at SmolLM2-1.7B + a Python-code dataset.
- **WIN:** a real production training script with zero unexplained lines.
- **OWN IT (rubric):** every block commented in your words · you can change 5 knobs deliberately (model_id, dataset_name, learning_rate, max_steps, target_modules) and justify each change.

## Lesson D.4 — LoRA math
- **Anchor:** the post-it, mathematically: instead of rewriting a 2048×2048 desk rule (4.2M numbers), write two skinny strips 2048×16 and 16×2048 (65K numbers) whose product *adds to* the rule. Rank = strip width = post-it size.
- **READ:** SESSION_1_RECAP §8 LoRA.
- **DO:** `lora_toy.py` (stdlib) — 4×4 frozen base matrix; rank-1 A (4×1) and B (1×4); output = base·x + (A·B)·x; count params (16 base vs 8 adapter). Then paper math: rank-16 on q_proj+v_proj for all 36 layers using real shapes from index.json.
- **WIN:** you see exactly why "rank 16" trains ~8M params instead of 3B.
- **OWN IT:** explain why low-rank updates suffice (fine-tuning is a small nudge in a low-dimensional direction), and what "merging" does (one matmul, zero inference cost).

## Lesson D.5 — QLoRA + the memory budget
- **Anchor:** the desks' permanent rules stored in shorthand (4-bit, ¼ the size) while post-its stay full-precision — you only un-compress a desk at the moment you read it.
- **READ:** SESSION_1_RECAP §8 QLoRA + `BitsAndBytesConfig` in train.py (`nf4`, compute dtype bf16).
- **DO:** paper budget — 3B model: 4-bit weights ≈ 1.5GB + 16-bit adapters + optimizer notepads on adapters only + activations → fits a free Colab T4 (16GB). Then the same table for 70B and 120B: watch QLoRA become the *only* option.
- **WIN:** you can predict "will it fit" before touching a GPU.
- **OWN IT:** fill the memory table unaided for 3B/70B/120B and explain what nf4 is in one sentence (a 4-bit codebook shaped like a bell curve).

## Lesson D.6 — Data for fiction
- **Anchor:** 5,000 great scenes beat 500,000 mediocre ones — the actor mirrors what they rehearse, including the flaws.
- **Goal anchor:** THE core skill for your 120B model. This lesson is your whole project's foundation.
- **READ:** dataset formats (messages vs prompt/completion) + chat template's role in training strings + README's SFT mention.
- **DO:** write 3 mini-scenes (10–15 lines each) from your own genre in messages format; hand-apply the template to each; mark which tokens the loss should care about (assistant turns only — the concept of loss masking).
- **WIN:** three real training records, formatted exactly like a production dataset.
- **OWN IT:** explain quality>quantity in actor language, what a record contains, and why we mask the user's tokens.

## Lesson D.7 — DPO
- **Anchor:** not "mimic this perfect take" but "here are two takes of the same scene — prefer A." Easier to make data (judging beats authoring), and it teaches *taste*, not just imitation.
- **READ:** SESSION_1_RECAP §8 DPO.
- **DO:** write 1 real preference pair for a fiction prompt (A = evocative, B = flat). Toy math: if the actor currently scores A and B equally, one DPO step nudges A's tokens up and B's down — trace which direction the loss pushes.
- **WIN:** you've built the atomic unit of preference data.
- **OWN IT:** explain SFT vs DPO losses, when each is needed, and why DPO generalizes better to unseen scenes.

---

# PHASE E — The long game

## Lesson E.1 — Long context (YaRN, 64k → 128k)
- **Anchor:** RoPE's rotation speed was tuned for 64k seats; YaRN is the projector lens that stretches the seat map so position 100,000 doesn't rotate into nonsense.
- **READ:** README "Long context processing" + `rope_theta`, `max_position_embeddings`, `rope_scaling` in config.
- **DO:** copy config.json → config_128k.json and add the exact YaRN block from the README (factor 2.0, original 65536). Explain each field.
- **WIN:** you performed the real 128k upgrade the README describes.
- **OWN IT:** explain why you can't just raise `max_position_embeddings` without scaling rotations.

## Lesson E.2 — Evaluation
- **Anchor:** three judges: loss curves (is rehearsal progressing), benchmarks (standardized auditions), eyeballing (does the prose *feel* right — the only judge that matters for your goal).
- **READ:** README eval tables; note which benchmarks exist and which are absent (nothing measures "emotionally nuanced fiction").
- **DO:** design your personal 10-prompt hand test — scenes that probe character voice, sensory detail, long-context recall. Write them now; they become your regression suite forever.
- **WIN:** your eval harness exists before your first training run — the pro move.
- **OWN IT:** explain what loss can and can't tell you, and why your 10 prompts catch things benchmarks can't.

## Lesson E.3 — Serving
- **Anchor:** the post-its merge back into the desks (one matmul), then the whole actor gets photocopied into cheaper formats (GGUF quantization) for the home theater (llama.cpp) or a real stage (vLLM).
- **READ:** README serving sections + walk through `notebook.ipynb` (the official inference notebook, 18 cells).
- **DO:** paper-trace the pipeline: merge LoRA → export → quantize (what Q4/Q8 mean for a 120B: ~60GB vs ~120GB) → serve. Pick your home stack and justify it.
- **WIN:** a concrete, chosen serving plan.
- **OWN IT:** explain what quantization loses and why Q4 is usually fine for prose.

## Lesson E.4 — FINAL CAPSTONE: `HAYLEE_120B_PLAN.md`
- **DO:** write the plan, top to bottom, in your own words: base model shortlist (with a parameter audit like A.6), data plan (D.6), SFT config (D.3 knobs chosen deliberately), LoRA rank + targets (D.4), memory budget (D.5), DPO stage (D.7), eval harness (E.2), serving (E.3), context strategy (E.1).
- **WIN:** a document you could hand to a practitioner and defend every line of.
- **OWN IT:** the rubric is the whole course: every choice justified with a metaphor AND a number.

---

## Parking Lot (tangents worth revisiting, not chasing mid-lesson)

- `smollm/text/pretraining/` — the 11T-token curriculum (staged web/code/math). Post-course reading.
- `smollm/text/evaluation/` — lighteval harness details.
- `smollm/vision/` — SmolVLM (multimodal sibling). Out of scope, interesting later.
- `notebook.ipynb` GPU walkthrough — do it live once installs happen.
- Tool-calling sections of the chat template (`xml_tools`/`python_tools`) — agentic usage, separate mini-course.

## Corrections log (small fixes to earlier sessions, verified Aug 13)

- True parameter count is **3,075,098,624** ("3.3B" in the recap was rounded high).
- Rank-8 LoRA on q/k/v/o ≈ **3.8M** params (recap said ~6M).
- Embedding table 262.7M ≈ *slightly less than* 4 layers of MLP (270.5M), not bigger.
- `tokenizer.json` is healthy now (128,000 + 256 added = 128,256).
