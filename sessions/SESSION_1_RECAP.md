# Everything We Covered — Session 1

Read this top to bottom and the whole framework should be in your head. Every concept is anchored in the actor / Scrabble / speed-dating / notepad metaphors we built together.

---

## 1. Why your safetensors download was 135 bytes - NOT DOWNLOADING NOW!

You ran `git clone` on a HuggingFace repo. The big files came down as 135-byte stubs, not gigabytes. That wasn't a bug — it was Git LFS (Large File Storage) doing exactly what it's designed to do on a machine that doesn't have `git-lfs` installed. The `.gitattributes` file in the repo marks `*.safetensors` and `tokenizer.json` as LFS-tracked, and the stub files contain a pointer (a hash and a size of ~4.96 GB each) telling LFS where to fetch the real file from. Without the `git-lfs` binary, those pointers are never resolved, and you just have empty stubs.

`tokenizer.json` happened to come down real on your `git lfs pull` because the LFS server sometimes delivers small files inline; the two 4.96 GB safetensors shards did not. The fix when you reinstall or on a new machine:

- Install `git-lfs` (`sudo dnf install git-lfs` on Fedora, `apt install git-lfs` on Debian/Ubuntu, `brew install git-lfs` on macOS)
- `cd` into the repo, then `git lfs install` (one-time) and `git lfs pull`
- For HuggingFace repos specifically, the `huggingface-cli download REPO --local-dir DIR` command is often easier than `git lfs` because it's resumable, parallel, and doesn't need any LFS plumbing

---

## 2. The actor metaphor — the single mental model

The model is a method actor. Over the course of their life, they have read every book, every script, every Reddit post, every Wikipedia article ever written. They didn't *understand* all of it. They just read it, over and over, until the patterns of how words flow together were burned into their bones. This is **pre-training**. They can finish your sentences, recite Hamlet, write a Wikipedia article about the French Revolution. But they have one big problem: they don't know when to *stop* being a parrot and start being a character. If you say "User: Hi, how are you?" they'll keep going with another fake "User:" turn, or a Wikipedia article about greetings, or a Reddit thread about AI assistants. They haven't been shaped to *respond* yet. They just *continue*.

**Fine-tuning is the rehearsal process.** You put them in a room and say, "Okay, here are 1,000 great scenes. Watch them. Notice how the actor in each scene pauses, chooses words carefully, stays in character, ends the scene when it's done. Then try it yourself on new scenes." After enough rehearsal, they get it.

That's the whole game. Everything else is unpacking what "watching 1,000 great scenes and learning from them" means mechanically.

---

## 3. Tokens — the Scrabble player

A token is a pre-chopped piece of a word. "I love Paris" reaches the model as 3 tokens: `["I", " love", " Paris"]`, not 11 characters. Common words stay whole; rare words split into common sub-pieces. The model has a **vocabulary** — a Scrabble bag of about 128,256 pre-chopped pieces (SmolLM3-3B uses the Llama-3 tokenizer). Every input gets chopped into the fewest pieces from that bag. The `vocab_size: 128256` in your `config.json` is the size of that bag.

---

## 4. Embeddings — the mood board

A token is just an integer, like `6204` for " Paris". A single number has no meaning. The model converts it into a **vector** — a list of 2,048 numbers — that encodes what " Paris" means in all the ways the model has learned to care about. Think of it as a mood board: 2,048 little checkboxes, some ticked, some half-ticked. " Paris" might score high on "European capital" and "romance" and "ends in -s." " potato" scores high on "food" and low on "European capital." The model has a giant **embedding table** — a 128,256 × 2,048 spreadsheet — where row 6204 is " Paris"'s mood board. Looking up a token means grabbing its row.

The 2,048 checkboxes are not labeled. The model discovered them by reading. During pre-training, it saw " Paris" appear in many contexts — "I love Paris," "Paris, France," "the Paris agreement" — and gradually figured out which checkboxes should be ticked to predict the next word in those sentences. We humans only see the patterns after the fact, when we look at the spreadsheet and notice "oh, dimension 487 always lights up for European cities."

---

## 5. The residual stream — the notepad

The embedding is the *starting* mood board. As the token flows through the model, that 2,048-dim vector gets *updated* again and again across 36 transformer blocks. Imagine the notepad the model carries through 36 rooms. When the token enters Room 1, the notepad has just the mood board. By Room 2, more notes. By Room 36, the notepad is covered in layered, accumulated notes from every room.

**Critically: nobody ever erases the notepad.** Each room only *adds* to it. This is the **residual connection** — the most important idea in deep learning. If a room's addition is bad or noisy, the notepad still has the original. The model can "ignore" a room's contribution just by not leaning on those dimensions later. If rooms *replaced* the notepad each time, mistakes would compound catastrophically. Adding-instead-of-replacing is what makes 36-deep stacks stable.

---

## 6. The transformer block — two sub-rooms

Each of the 36 rooms has the same layout: two sub-rooms, one after the other, each of which adds notes to the notepad.

### Sub-room 1: Attention — "look back at what was said"

When the model is processing the token "her" in *"Sarah told her that she left,"* the notepad's current state is just "a female pronoun, low information." It doesn't know *which* her this is. To figure that out, "her" has to look at the other tokens in the sequence.

**Analogy: attention is a speed-dating round.** Each token sits at a little table. The current token ("her") walks around to every other table and asks, "On a scale of 0 to 1, how relevant are you to me?" The other tokens show their "I am a female name" score, their "I am a pronoun too" score, their "I am far away in the sentence" score. The current token multiplies all those scores into one number per other token, then takes a weighted blend of all the other tokens' notepads.

The result: "her"'s notepad is now partly its own mood board, partly a copy of "Sarah"'s notepad (because Sarah scored very high on "I am a female name that this pronoun refers to"), and partly a copy of everyone else's notepads with smaller weights. **"her" has now resolved: "I refer to Sarah."** That weighted blend gets *added* to the notepad.

**The 16 heads.** Why 16 attention heads per layer? Because there are 16 different kinds of relationships worth tracking at once. One head might specialize in "what's the previous word" (syntax). Another in "what's my pronoun's referent" (coreference — exactly the "her" example). Another in "what was the most recent period" (sentence boundary). Another in "what's the most semantically similar word to me" (analogy/retrieval). They all run in parallel, each with their own opinion about who's relevant. The model concatenates all 16 opinions and adds the result to the notepad. The model learned this division of labor itself during pre-training. Nobody told it "head 7 should do coreference." It figured that out because doing coreference well helped it predict the next word on average.

**GQA — the memory trick.** SmolLM3-3B has 16 query heads but only 4 key/value heads. The 4 K/V heads are shared, each used by 4 query heads. Why? Storing the "notepad snapshot" for every token at every layer is expensive. The model found that 4 notepad snapshots, each shared by 4 query heads, was almost as good as 16 separate snapshots. This is **Grouped Query Attention**. It saves memory for the speed-dating part with barely any quality loss. Standard on basically every model post-2023.

**NoPE / RoPE — the long-context trick.** Every 4th layer (3, 7, 11, …) uses RoPE — rotary position embeddings. RoPE rotates each Q and K vector by an angle that depends on its position in the sequence, so the dot product Q·K naturally encodes "I'm 5 tokens away from you" rather than "I'm at absolute position 7." The other layers skip this. The result is a model that can handle very long sequences (65k tokens) without position information degrading the way pure-RoPE models do at long distances. SmolLM3 is one of the first widely-used models to ship this.

### Sub-room 2: MLP — "think about what I now represent"

After attention has updated "her"'s notepad with "I refer to Sarah," the notepad is a mixed bag: original mood board + "I'm a pronoun" + "Sarah is my referent" + everything else. Time to *think* about it.

**Analogy: the MLP is a thinking chamber with 11,008 desks.** The notepad enters the chamber. For one moment, the model duplicates the notepad 11,008 times, each copy getting its own desk. Each desk looks at the full notepad and computes "given everything I see, what should I write down?" The 11,008 desks don't all write the same thing. One desk might be specialized for "if I see a female pronoun + a name, write down 'resolve reference.'" Another desk might be "if I see a past-tense verb, write down 'narrative mode.'" Another might be "if I see a sentence end, write down 'potential new topic.'" After all 11,008 desks have written, the model collapses all their notes back into a single 2,048-dim vector and adds it to the notepad.

**Why 11,008?** The model's designers picked 5.4× the notepad width (2,048 × 5.4 ≈ 11,000). It's a sweet spot: wide enough to give the model lots of thinking space, narrow enough to not be a memory disaster. Llama-3-style models all use ratios in the 4–5.4× range.

**The "knowledge" lives here.** The 67.5M parameters of the MLP per layer are where the model stores most of its facts and *style*. If you fine-tune the model to write in your voice, **most of the change happens in the MLP weights.** The attention weights change too, but mostly to do routing — deciding which other tokens to copy from. The MLP weights change to do production — actually generating the right words in the right style. This intuition will matter when you build your 120B story model.

---

## 7. The 36 layers — passing the notepad through 36 rooms

After Room 1, the notepad has: embedding + attention-1 + MLP-1. After Room 2: + attention-2 + MLP-2. By Room 36, the notepad is the sum of 1 + 72 additions. Each addition was small on its own, but layered this deep, the notepad now contains an extraordinarily rich representation of "given everything so far, what should come next."

**Rough specialization by depth** (observed by researchers):
- **Early layers (1–6):** surface stuff. Word-level meaning, basic syntax.
- **Middle layers (7–24):** phrase-level meaning, entity tracking, coreference resolution, local coherence.
- **Late layers (25–36):** planning, "what should I say next," style, factual recall, the actual response generation.

The notepad after layer 36 is fed into one final operation: a giant lookup table that converts the 2,048-dim notepad back into "scores for each of the 128,256 possible next tokens." The highest-scoring token gets picked (or, with sampling, the probabilities guide a random pick). That token becomes the next input. **That loop is inference.** Doing it 500 times is generating 500 tokens. Doing it with the right plumbing is what `transformers`, `vLLM`, `llama.cpp`, and `ollama` are doing under the hood.

---

## 8. Where fine-tuning lives

### Full fine-tuning
You put the actor through 1,000 scenes. The 11,008 desks in every thinking chamber of every one of the 36 rooms — all of their specialty rules get rewritten, little by little, so the actor produces the right kind of output. Every desk in every room gets a new rule. That's 3.3 billion rule changes for SmolLM3-3B. The optimizer (the thing doing the rewriting, usually Adam) needs to keep extra notes about how each desk is changing (momentum, variance) so it can rewrite intelligently. **That's why training needs ~3× the memory of inference:** the model's weights + the optimizer's notes about the weights. 3.3B params × 3 ≈ 10 GB just for the basics, plus activations during training → 30+ GB. A100/H100 territory. Way out of reach for free Colab.

### LoRA
Instead of rewriting the 11,008 desks, you stick a **post-it note** on each desk. The desk's original rule is still there, but the post-it says "and also, factor in this little adjustment." You only train what's written on the post-its. After training, you can merge the post-its back into the desk's permanent rule with one matmul, so inference is the same speed as before.

The post-it trick works because, mathematically, "how to update a giant matrix during fine-tuning" can almost always be approximated as "the product of two skinny matrices." A 2048×2048 update can be replaced with a 2048×8 times an 8×2048 — that's 32K numbers instead of 4M. The "rank" (the 8) controls how big the post-it is. **Rank 8** = small post-it, less expressive, less memory. **Rank 64** = big post-it, more expressive, more memory. For SmolLM3-3B, rank 16 is the typical sweet spot. With LoRA on q_proj, k_proj, v_proj, o_proj at rank 8, you're training roughly 6 million parameters — trains on a single 24 GB GPU.

### QLoRA
Same as LoRA, but the desks' *original* rules are stored in shorthand — instead of the full rule, the desk's permanent rule is stored in a 4-bit compressed form that's 1/4 the size. The post-its (the LoRA adapters) stay in full 16-bit because you need precision to train them. The base is 4-bit, the LoRA is 16-bit. **This is what fits a 3B model on a free Colab T4 (16 GB).** For your 120B target, QLoRA is the only realistic option without spending serious money.

### SFT vs DPO
- **SFT** = show the actor 1,000 scenes, ask them to mimic the best take. Loss = "how close was your take to the reference?"
- **DPO** = show the actor 1,000 pairs of takes (Take A vs. Take B of the same scene, A is better). Don't give a reference — just say "pick A over B." Loss = "how often do you make the right pick?"

**DPO is insanely powerful for your use case.** Imagine for each of 2,000 roleplay exchanges, you write Take A (rich, in-character, atmospheric) and Take B (correct but flat). You don't have to write a perfect "ideal" response — you just have to know which of two is better. That's way easier, and the model learns *preferences* rather than *mimicry*, which generalizes better to new scenes.

---

## 9. The pipeline for your 120B story model

Translating everything into the actor metaphor:

1. **Pick a base actor** (a 120B base model — Qwen3, Llama-3.x, Mistral-Large, TBD). They have read everything but don't know how to perform your kind of story.
2. **Curate 5,000–50,000 great scenes** from your own writing. Each "scene" is a (system prompt setting, story context, your ideal continuation). Quality matters way more than quantity. 5,000 great examples beats 500,000 mediocre ones.
3. **SFT with QLoRA.** Show the actor all the scenes. Run a few passes. Save the post-it adjustments.
4. **Curate 1,000–5,000 preference pairs for DPO.** Each pair is (same prompt, Take A which is more evocative, Take B which is flatter). Train the post-its again to prefer A over B.
5. **Test ruthlessly.** Hand the actor a brand-new scene they haven't seen. Watch what they do. If they break character at 10,000 tokens, that's a context-length issue (the chat template or the model's architecture) — not a training-data issue. If they're in character but the prose feels generic, you need more SFT data. If the prose is in voice but they're choosing *less* evocative completions, you need more DPO.
6. **Save the post-its merged back into the actor.** Export. Serve locally with llama.cpp or vLLM.

That's the whole thing. There are no other moving parts. There are *details* about each step — lots of them — but the architecture is "actor + 5,000 great scenes + 2,000 preference pairs + rehearsal."

---

## 10. The one-sentence summary

**The model is a method actor who has read everything but doesn't know how to perform. Fine-tuning is rehearsal. LoRA is post-it notes. SFT is "mimic these great takes." DPO is "prefer this take over that one."**

If you can explain those four sentences to someone else, you have the whole mental model. Everything else is detail.

---

## What we have NOT yet covered (the next 9 sessions)

- Session 2: Loading a model and watching a single forward pass on real numbers
- Session 3: What training mechanically does (loss, gradients, one optimizer step)
- Session 4: SFT with HuggingFace `transformers` + `trl` — reading a real training script
- Session 5: LoRA / QLoRA — the math of why rank-8 actually works
- Session 6: Data for fiction — how to build a roleplay dataset that doesn't suck
- Session 7: DPO / preference tuning — how to build the preference pairs
- Session 8: Reasoning + long context (the SmolLM3 no-think/think toggle)
- Session 9: Serving a fine-tune — vLLM, llama.cpp, GGUF export
- Session 10: Scaling to 70B/120B — what's actually different
