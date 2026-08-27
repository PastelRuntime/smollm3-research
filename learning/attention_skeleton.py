TOKENS = ["Sarah", "told", "her", "that", "she", "left"]

BOARDS = {
    "Sarah": [1.0, 0.0, 0.0, 1.0],
    "told":  [0.0, 1.0, 0.0, 0.0],
    "her":   [1.0, 0.0, 1.0, 0.0],
    "that":  [0.0, 0.0, 0.0, 0.0],
    "she":   [1.0, 0.0, 1.0, 0.0],
    "left":  [0.0, 1.0, 0.0, 0.0],
}

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(scores):
    e = [2.718281828 ** s for s in scores]
    total = sum(e)
    return [x / total for x in e]

def blend(weights, vectors):
    return [sum(w * v[i] for w, v in zip(weights, vectors))
            for i in range(len(vectors[0]))]

q = BOARDS["her"]
scores = [dot(q, BOARDS[t]) for t in TOKENS]
weights = softmax(scores)
new_notepad = blend(weights, [BOARDS[t] for t in TOKENS])

print("scores:  ", [round(s, 2) for s in scores])
print("weights: ", [round(w, 3) for w in weights])
print("her now: ", [round(x, 3) for x in new_notepad])