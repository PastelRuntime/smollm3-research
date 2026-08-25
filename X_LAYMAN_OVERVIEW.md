# X Post — Experiment 2 Results + The Road Ahead

> Draft for posting on X. First-person, single post. Layman + exact architecture.
> Keep the energy, keep the receipts. ALWAYS REPLACE BELOW for future posts.

---

I gave a 3B model a lobotomy and then taught it to work around the damage. Results are in. 🧵→ (no, it's all one post, stay with me)

The model: SmolLM3-3B. Weird little thing — 36 layers, and every 4th layer has NO positional encoding at all (27 RoPE + 9 NoPE layers). The NoPE layers are blind to where words sit; the RoPE layers track position. Nobody fully knew what that buys you.

A 2025 paper claimed you could cap how far back the RoPE layers "look" (sliding attention, 8k window) and get long-context retrieval AND big memory savings. Nobody had tested it on a model that already exists. So I did — pre-registered, all 40 runs public on my GitHub.

Experiment 1: flip the window on AFTER training. It ran up to 21% faster… and went completely blind past 8k. Asked it to find one fact buried in a 32k-token document: 0 out of 5. Every time. You can't bolt the trick onto a finished brain.

Experiment 2 (just landed): fine-tune WITH the window on — LoRA, rank 32, ~6.5M tokens of books, a few GPU-hours on free Kaggle T4s. Training sequences were 2x the window, so every example forced far-context info to route through those position-blind NoPE layers.

Result: needle-in-haystack 5/5 at 8k, 16k, 32k, and 64k. Perfect retrieval at FOUR TIMES the window. The model rebuilt its own retrieval pathway around the constraint. The speed gain didn't survive (honest data: it's actually slower than stock now, and memory parity — the efficiency prize stays unclaimed), but the capability claim is airtight: windowed attention doesn't destroy long context IF the model adapts under it.

Two things I learned building this:
1. Small models aren't limited by what people think they're limited by. They're limited by what nobody's bothered to test.
2. Pre-registering hypotheses before every run is the entire ballgame. I can't retrofit a story onto the data because the story was written first.

What's next — and this is where it gets fun. Three years of vibe coding taught me one law: the prompt is the spec, and the spec is the product. Agents amplify your discipline, not your intent. Shit in, shit out.

So the next experiment: teach the 3B to INTERROGATE. You give it "build me a login page," it asks what a senior engineer would ask — token expiry? rate limits? what test proves done? — and emits the full engineering spec. The cheap model writes the questions; the expensive model answers them. It's the kind of job a laptop iGPU can do all day while the big GPU sleeps.

After that: bigger swing. The new open image/video models are 80B+ (HunyuanImage-3.0, MAGI-2), all memory-heavy MoE. They fit in landfill-priced used GPU fleets — if someone builds the native multi-GPU engine. Nobody has. Receipts-first, same as always.

Everything's public: pre-registrations, kernels, raw results JSONs, warts and all. github.com/PastelRuntime/smollm3-research

DMs open if you're into NoPE, small models, or scrapheap GPU farms. 🤝
