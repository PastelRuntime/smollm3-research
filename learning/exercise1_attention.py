EMBEDDING_SIZE = 4

TOKENS = ["Sarah", "told", "her", "that", "she", "left"]

MOOD_BOARDS = {
    "Sarah": [1.0, 0.0, 0.0, 1.0],
    "told":  [0.0, 1.0, 0.0, 0.0],
    "her":   [1.0, 0.0, 1.0, 0.0],
    "that":  [0.0, 0.0, 0.0, 0.0],
    "she":   [1.0, 0.0, 1.0, 0.0],
    "left":  [0.0, 1.0, 0.0, 0.0],
}

DIM_NAMES = ["female", "verb", "pronoun", "sarah-ness"]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    e = [2.718281828 ** s for s in scores]
    total = sum(e)
    return [x / total for x in e]


def blend(weights, vectors):
    out = [0.0] * EMBEDDING_SIZE
    for w, v in zip(weights, vectors):
        for i in range(EMBEDDING_SIZE):
            out[i] += w * v[i]
    return out


FOCUS = "her"
pos = TOKENS.index(FOCUS)

print("=== ACT 1: the mood boards ===")
for t in TOKENS:
    print(f"  {t:5s} -> {MOOD_BOARDS[t]}")

print("\n=== ACT 2: Q/K/V (today: Q = K = V = the mood board) ===")
q = MOOD_BOARDS[FOCUS]
print(f"  {FOCUS}'s question (Q): {q}")
print("  everyone's label (K):  their mood board")

print("\n=== ACT 3: speed-dating — dot product scores ===")
raw_scores = []
for t in TOKENS:
    s = dot(q, MOOD_BOARDS[t])
    raw_scores.append(s)
    print(f"  score({FOCUS} vs {t:5s}) = {s:+.2f}")

print("\n=== ACT 4: softmax — scores become a recipe ===")
weights = softmax(raw_scores)
for t, w in zip(TOKENS, weights):
    print(f"  {t:5s} gets {w:.3f} of the blend")

print("\n=== ACT 5: the weighted blend (her's new notepad) ===")
old_notepad = MOOD_BOARDS[FOCUS]
new_notepad = blend(weights, [MOOD_BOARDS[t] for t in TOKENS])
print(f"  before: {old_notepad}")
print(f"  after:  {[round(x, 3) for x in new_notepad]}")
for i, name in enumerate(DIM_NAMES):
    print(f"    {name:12s} {old_notepad[i]:+.2f} -> {new_notepad[i]:+.2f}")

print("\n=== ACT 6: the causal mask (why the model is 'anxious') ===")
masked_scores = [s if i <= pos else float("-inf") for i, s in enumerate(raw_scores)]
masked_weights = softmax(masked_scores)
for t, w in zip(TOKENS, masked_weights):
    print(f"  {t:5s} gets {w:.3f} of the blend")
masked_notepad = blend(masked_weights, [MOOD_BOARDS[t] for t in TOKENS])
print(f"  her's notepad, unable to see the future: {[round(x, 3) for x in masked_notepad]}")