import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import traceback

from huggingface_hub import HfApi
from transformers import AutoTokenizer

api = HfApi()
TARGET_VOCAB = 248077
out = {"candidates": {}, "matches": []}


def save():
    with open("/kaggle/working/recon2.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


print("=== searching Qwen3.5 family ===", flush=True)
try:
    hits = list(api.list_models(search="Qwen3.5", limit=60))
    ids = [m.id for m in hits]
except Exception:
    ids = []
    out["search_error"] = traceback.format_exc()[-300:]
out["qwen35_ids"] = ids
save()
print("\n".join(ids[:30]), flush=True)

# probe smallest-looking candidates first
probe_order = [i for i in ids if any(s in i.lower() for s in ("0.6b", "0.8b", "1b", "1.7b", "2b"))] + \
              [i for i in ids if any(s in i.lower() for s in ("4b", "small", "base"))]
seen = set()
for mid in probe_order:
    if mid in seen:
        continue
    seen.add(mid)
    print(f"--- {mid} ---", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(mid)
        L = len(tok)
        entry = {"tokenizer_len": L, "vocab_match": L == TARGET_VOCAB}
        out["candidates"][mid] = entry
        if entry["vocab_match"]:
            out["matches"].append(mid)
            print(f"*** VOCAB MATCH ({L}) ***", flush=True)
    except Exception:
        out["candidates"][mid] = {"err": traceback.format_exc()[-200:]}
    save()

print("\n=== MATCHES ===", flush=True)
print(out["matches"], flush=True)
print("DONE", flush=True)