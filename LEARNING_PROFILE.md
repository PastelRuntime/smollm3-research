# Learning Profile — Haylee

## Who You Are
You are an aspiring AI researcher/engineer with about three years of following the AI space from the outside. Your working method: leverage AI tools (opencode as tutor/agent, Kaggle kernels for GPU compute) and coding frameworks to do the researching and testing. You have ADHD, which means you learn best when content is engaging, broken into small chunks, and gives you frequent "I just did a thing" payoffs. You absorb analogies far better than raw definitions. You have solid Python environment literacy and understand API-based LLM usage, tokens, and local inference (GGUF, safetensors, llama.cpp, ollama). You know the lingo at a high level — SFT, RLHF, LoRA, QLoRA, parameter counts, quantization — but the *mechanics* underneath those words have been a black box until now.

## How You Learn Best
- **Analogies over math.** Pure equations and dimension-counting (e.g., "2048-dim vector with 16 heads of 128") lose you quickly. Domain-translated metaphors (acting, Scrabble, speed dating, mood boards, notepads) click.
- **Concrete over abstract.** You want to see a real number, a real shape, a real script — not a "concept."
- **One thing at a time.** When I try to give you five deep-dives in one reply, your eyes glaze. When I give you one deep-dive, you absorb it.
- **No assumed knowledge.** When I use a term without re-defining it through analogy, you stall out and have to ask. Always translate.
- **Goal-anchored.** You learn better when each piece is visibly connected to your actual use case (fine-tuning a 120B model for nuanced, emotionally deep, slice-of-life erotica and roleplay) than when it's framed as general education.
- **Frequent check-ins.** You want to be asked "which of these three next?" not marched through a syllabus. Agency over curriculum.
- **Bite-sized sessions.** Each session should be one chat, one experiment, one working thing. ADHD does not reward 40-minute monologues.

## Current Knowledge State

### Solid (high-level, conversational fluency)
- LLM basics, what a token is, what an embedding roughly is
- Python environments, pip, virtualenvs
- API-based inference (OpenAI/Anthropic/HF style)
- Local inference stack: GGUF, safetensors, llama.cpp, ollama
- The vocabulary of fine-tuning: SFT, RLHF, LoRA, QLoRA, DPO, full fine-tune
- Parameter sizing intuition (3B vs 70B vs 120B)
- Quantization as a concept (4-bit, 8-bit, etc.)
- The general pipeline: data → train → evaluate → serve

### Gap (the black box — what you're here to fix)
- The actual architecture of a transformer block: attention, MLP, residuals, layer norms
- What the 2048 hidden dimension, 16 heads, 11008 intermediate size *mean*
- What "training" mechanically does (loss, gradients, optimizer state, why 3x memory)
- What LoRA mathematically is and why rank-8 vs rank-64 matters
- What QLoRA's 4-bit base + 16-bit adapters actually looks like in memory
- How to build a training dataset for a specific style/use case
- How evaluation actually works (loss curves vs. eyeballing generations)
- Serving a fine-tuned model (vLLM, llama.cpp with adapters, GGUF export)
- The reasoning-mode trick (SmolLM3's no-think/think toggle, system-prompt gated)
- Context length engineering (RoPE scaling, NoPE, sliding window)
- What changes when you scale from 3B to 70B to 120B

## Tools & Environment
- Local machine: Linux (Fedora-family based on your `dnf` reference earlier in the session)
- Currently no GPU
- Python available, no torch/numpy installed yet (and we agreed: no installs unless you decide to)
- git-lfs was not installed; you learned why your safetensors download was stub files
- HuggingFace CLI: not yet used; alternative to git-lfs for HF repo downloads
- Kaggle: account `haylee00`, access token at `~/.kaggle/access_token` (KGAT). Kaggle CLI not installed — use the REST API. Kernels run on Tesla T4; "SmolLM3 Session2" ran real SmolLM3-3B weights and produced session2/attention.png
- Comfortable with CLI, file editing, git

## Ultimate Goals

### Short-term (next 1–2 months)
- Genuine end-to-end understanding of the transformer architecture, training loop, and fine-tuning mechanics
- Ability to read a training script and know what every line does
- Ability to run a real SFT/LoRA fine-tune on a small model (3B-class) on free Colab
- A mental model that lets you scale up later without re-learning the foundations

### Medium-term (3–6 months)
- Build a high-quality roleplay/story fine-tune on a 7B–13B model
- Master QLoRA, DPO, and dataset curation
- Get a fine-tuned model running locally with llama.cpp or ollama
- Start sharing work / getting feedback from the community

### Long-term (6–12+ months)
- Fine-tune a 70B–120B base model for **nuanced, emotionally deep, slice-of-life erotic fiction and roleplay**
- The model should produce prose with: emotional nuance, sensory detail, consistent character voice, long-context coherence, atmospheric writing, tasteful intimacy
- Build a workflow that lets you iterate on data, retrain, and evaluate without burning a fortune on compute
- Possibly: serve it for yourself or a small audience, or contribute back to the open-source roleplay model ecosystem

## The 10-Session Roadmap (the one we built together)

1. The transformer block, end-to-end — drawing what happens to a token from input to logits
2. Loading a model and watching a single forward pass
3. What training actually does — loss, gradients, one optimizer step
4. SFT with HuggingFace `transformers` + `trl`
5. LoRA / QLoRA — why this exists, what it modifies
6. Data for fiction — how to build a roleplay dataset that doesn't suck
7. DPO / preference tuning
8. Reasoning + long context (the SmolLM3 specifics)
9. Serving a fine-tune — vLLM, llama.cpp, GGUF
10. Scaling to 70B/120B — what's actually different

## Session Format That Works
- One chat = one session = one mental model + one tiny experiment
- Pure-prose explanation first, code only when you say you're ready to run it
- Every new term gets an analogy before a definition
- Always end with 2–3 "which next?" options
- No 40-minute monologues. Stop and check in.

## Things to Avoid (from how this session went)
- Do NOT pile on five deep-dives in one reply. Pick one.
- Do NOT assume a term is understood. Re-define with analogy.
- Do NOT throw dimensions and matrix shapes at you as the primary explanation. Use them as supporting detail, not the main act.
- Do NOT install packages on your machine without asking. You came to *understand*, not to set up a dev environment.
- **NEVER save anything to /tmp.** All files live in the project folder. (Rule set Aug 15, 2026 — I used /tmp for Kaggle outputs once; don't repeat.)
- DO use your actual goal (the 120B fiction model) as the recurring anchor. Every concept gets connected back to "and this is what you'll do with it when you fine-tune your own model."

## Open Questions / Decisions Pending
- Do you want to actually run the pure-stdlib attention/MLP simulation script later? (Uses only Python built-ins, no installs.) → **RESOLVED Aug 12: yes — this is now the core format (skeleton files + graded edits).**
- Do you want to download a small base model and try real SFT on free Colab? If so, which one?
- For data: do you already have a corpus of your own writing, or will you need to curate from elsewhere?
- When you scale to 120B, which base model are you leaning toward? (Qwen3, Llama-3.x, Mistral-Large, etc. — TBD.)

## What the teacher learned (Aug 12, 2026 — the MLP desk lesson)

Haylee hit "assembly paralysis": she understood positive/negative/0 perfectly, but froze on *which numbers go in which slots*. Three reframes unblocked her, in this order:

1. **Position IS meaning.** Before ANY vector exercise, make the slot-ordering contract explicit: the numbers aren't a pile; slot 0, 1, 2, 3 map to fixed boxes everywhere (notepad, desk, output). The dot product pairs slot-by-slot. She noticed the example desks "cascade" — that's each detector's 1 living in its own slot.
2. **You are the chart.** She kept searching for an external reference ("an invisible chart"). There isn't one — SHE writes the sentence, the numbers are translations. Give a physical cheat sheet (`DESK_BUILDER_CHEATSHEET.md`): sentence word → number (care a lot=+2, care a bit=+1, dislike=-1, hate=-2, don't care=0). Only 0/±1/±2 kills the magnitude paralysis — absolute values don't matter, only ranking.
3. **Labels are our invention.** Explicitly: the slot labels exist for the reader, not the model; the real model has 2,048 unnamed slots and meaning is learned-then-interpreted. Stating this removed an entire layer of anxiety ("am I following the hidden ruleset?").

Other observations:
- When she's stuck, ask her to NAME the exact confusion — she pinpointed it herself ("there are 4 numbers beside each desk... I noticed yours cascade") and the fix was immediate.
- Reframes beat more examples. More correct examples did not cure assembly paralysis; the translation-table reframe did.
- "Number of non-zero slots = number of opinions; zeros are 'no opinion', not 'empty'" was a sticky formulation.
- Two-layer model (labels for us, numbers for the model) produced real energy: "we are making REAL headway and I'm really excited."
- Format evolution: she now edits and runs pure-stdlib scripts herself (no installs) and posts output for grading. "No code yet" rule is retired in favor of "code, but pure-stdlib only."
- She values the grading rubric framing ("I'll check: runs clean · recipe intact · sentences use metaphors correctly"). Keep giving rubrics.

**Now owned (post-session):** dot product as scoring (mechanical), softmax as recipe, causal mask behavior, desk = weight vector, slot-ordering contract, sentence → numbers translation. **Still building:** MLP collapse/down-matrix intuition, attention heads, GQA, the remaining config numbers (4–8).
