import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import traceback

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

api = HfApi()
out = {"targets": {}, "drafters": {}, "search_results": {}, "vocab_matches": [], "gguf_repos": []}


def save():
    with open("/kaggle/working/recon.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def probe_model(mid):
    """Return dict of facts about a candidate model, or raise."""
    info = api.model_info(mid)
    facts = {"exists": True, "downloads": getattr(info, "downloads", None),
             "pipeline": getattr(info, "pipeline_tag", None)}
    cfg_path = hf_hub_download(mid, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    moe_keys = {k: v for k, v in cfg.items() if any(
        s in k.lower() for s in ("expert", "moe", "num_experts"))}
    facts.update({
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "hidden_size": cfg.get("hidden_size"),
        "num_layers": cfg.get("num_hidden_layers"),
        "vocab_size": cfg.get("vocab_size"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "rope_theta": cfg.get("rope_theta"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "moe_indicators": moe_keys,
        "is_moe_suspect": bool(moe_keys),
    })
    return facts


TEST_STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "<|im_start|>user\nHello<|im_end|>",
    "1, 1, 2, 3, 5, 8, 13, 21",
]

TARGET_CANDIDATES = [
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.8-27B-Instruct",
    "Qwen/Qwen3.8-27B-Base",
]
DRAFTER_CANDIDATES = [
    "Qwen/Qwen3.5-Small-0.8B",
    "Qwen/Qwen3.5-Small-2B",
    "Qwen/Qwen3.5-Small-4B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]

print("=== searching HF for actual Qwen3.8 model IDs ===", flush=True)
try:
    hits = list(api.list_models(search="Qwen3.8", limit=40))
    out["search_results"]["qwen38"] = [m.id for m in hits]
    print("\n".join(out["search_results"]["qwen38"][:20]), flush=True)
except Exception:
    out["search_results"]["qwen38"] = f"search failed: {traceback.format_exc()[-300:]}"
save()

target_id = None
for mid in TARGET_CANDIDATES:
    print(f"--- probing target candidate {mid} ---", flush=True)
    try:
        facts = probe_model(mid)
        out["targets"][mid] = facts
        save()
        if target_id is None and not facts["is_moe_suspect"]:
            target_id = mid
            print(f"ACCEPTED as dense target: {mid}", flush=True)
    except Exception:
        out["targets"][mid] = {"exists": False,
                               "err": traceback.format_exc()[-300:]}
        save()
        print(f"failed: {mid}", flush=True)

if target_id is None:
    # fall back to any existing candidate even if MoE-suspect, for vocab work
    for mid, facts in out["targets"].items():
        if facts.get("exists"):
            target_id = mid
            break

print(f"\n=== TARGET LOCKED: {target_id} ===", flush=True)
out["locked_target"] = target_id
tok_t = AutoTokenizer.from_pretrained(target_id)
t_ids = [tok_t.encode(s) for s in TEST_STRINGS]
out["target_tokenizer"] = {"class": type(tok_t).__name__,
                           "len": len(tok_t), "vocab_size_cfg": out["targets"][target_id].get("vocab_size")}
save()
print(f"target tokenizer len={len(tok_t)}", flush=True)

print("\n=== probing drafter candidates ===", flush=True)
for did in DRAFTER_CANDIDATES:
    print(f"--- {did} ---", flush=True)
    entry = {}
    try:
        facts = probe_model(did)
        entry.update(facts)
        tok_d = AutoTokenizer.from_pretrained(did)
        d_ids = [tok_d.encode(s) for s in TEST_STRINGS]
        entry["tokenizer_len"] = len(tok_d)
        entry["identical_encoding_on_tests"] = d_ids == t_ids
        entry["per_string_match"] = [a == b for a, b in zip(t_ids, d_ids)]
        if d_ids == t_ids and entry.get("vocab_size") == out["target_tokenizer"].get("vocab_size_cfg"):
            out["vocab_matches"].append({"drafter": did, "params_hint": facts.get("num_layers"),
                                         "tokenizer_len": len(tok_d)})
    except Exception:
        entry = {"exists": False, "err": traceback.format_exc()[-300:]}
    out["drafters"][did] = entry
    save()
    print(json.dumps({k: v for k, v in entry.items() if k != "err"}, default=str)[:300], flush=True)

print("\n=== GGUF availability for target ===", flush=True)
if target_id:
    base_name = target_id.split("/")[-1]
    try:
        hits = list(api.list_models(search=f"{base_name} gguf", limit=20))
        out["gguf_repos"] = [{"id": m.id, "downloads": getattr(m, "downloads", None)} for m in hits]
    except Exception:
        out["gguf_repos"] = f"search failed: {traceback.format_exc()[-300:]}"
    for g in (out["gguf_repos"] if isinstance(out["gguf_repos"], list) else [])[:10]:
        print(g, flush=True)
save()

print("\n=== RECON SUMMARY ===", flush=True)
print(f"locked target: {target_id}", flush=True)
print(f"dense suspect: {not out['targets'].get(target_id, {}).get('is_moe_suspect')}", flush=True)
print(f"exact-vocab drafter matches: {[v['drafter'] for v in out['vocab_matches']]}", flush=True)
print(f"gguf repos found: {len(out['gguf_repos']) if isinstance(out['gguf_repos'], list) else 'n/a'}", flush=True)
print("DONE", flush=True)