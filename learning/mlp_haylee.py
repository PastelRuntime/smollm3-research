NOTEPAD = [0.854, 0.098, 0.721, 0.133]
NOTEPAD = [0.854, 0.098, 0.721, 0.133]

DESKS = [
    ("sarcasm detector",       [1.0, 0.0, 0.0, 0.0]),
    ("past-tense detector",         [0.0, 1.0, 0.0, 0.0]),
    ("gender-identity detector",      [0.0, 0.0, 1.0, 0.0]),
    ("possessiveness detector",        [0.0, 0.0, 0.0, 1.0]),
    ("female + sarah combo",  [1.0, 0.0, 0.0, 1.0]),
    ("pronoun + sarah combo", [0.0, 0.0, 1.0, 1.0]),
    ("female + verb combo",   [1.0, 1.0, 0.0, 0.0]),
    ("everything detector",   [0.2, 0.2, 0.2, 0.2]),
]

GAIN = 0.25

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def relu(x):
    return max(0.0, x)

print("=== EXPAND: every desk looks at the full notepad ===")
raw = []
for name, w in DESKS:
    note = dot(w, NOTEPAD)
    raw.append(note)
    print(f"  {name:22s} saw {note:+.3f}")

print("\n=== THINK: desks only write down what they actually saw ===")
activated = [relu(n) for n in raw]
for name, a in zip([d[0] for d in DESKS], activated):
    print(f"  {name:22s} writes {a:+.3f}")

print("\n=== COLLAPSE: notes get scaled down and written into 4 dims ===")
output = [0.0] * 4
for i, a in enumerate(activated):
    dim = i % 4
    output[dim] += GAIN * a
    print(f"  desk {i:2d} -> dim {dim}: {GAIN * a:+.3f}")

print("\n=== RESIDUAL: add the chamber's notes to the notepad ===")
new = [NOTEPAD[i] + output[i] for i in range(4)]
print(f"  before: {NOTEPAD}")
print(f"  added:  {[round(x, 3) for x in output]}")
print(f"  after:  {[round(x, 3) for x in new]}")