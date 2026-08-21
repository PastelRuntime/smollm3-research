import json
import subprocess
import sys


def ensure_cuda_torch():
    probe = (
        "import torch; "
        "x = torch.ones(1024, 1024, device='cuda', dtype=torch.float16); "
        "y = (x + 1).sum().item(); "
        "torch.cuda.synchronize(); "
        "assert y == 2097152.0; "
        "print(torch.cuda.get_device_name(0))"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if r.returncode == 0:
        return
    for ver in ("2.4.1+cu124", "2.5.1+cu124", "2.2.2"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torch", "torchvision", "torchaudio"]
        )
        if "+" in ver:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", f"torch=={ver}",
                 "--index-url", "https://download.pytorch.org/whl/cu124"]
            )
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"torch=={ver}"])
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"base torch unsupported on this GPU, downgraded to torch {ver}", flush=True)
            return
    raise RuntimeError("no CUDA-capable torch found: " + r.stderr[-500:])


ensure_cuda_torch()
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.53,<5", "accelerate"]
)

import torch

assert torch.cuda.is_available()
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)

import gc
import time

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
WINDOW = 8192
LENGTHS = [8192, 16384, 32768, 65536]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
NEEDLE = "The special magic number mentioned in the text is 8675309."
QUESTION = "What is the special magic number mentioned in the text? The special magic number is"
ANSWER = "8675309"
INSERT_BEFORE = "\nThe needle is inserted here.\n"
INSERT_AFTER = "\nThe needle is inserted here again.\n"
FILLER = "The weather was calm and the city was quiet that day. "
MAX_NEW = 40

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
pad_id = tokenizer.eos_token_id

filler_ids = tokenizer(FILLER, add_special_tokens=False)["input_ids"]
needle_ids = tokenizer(INSERT_BEFORE + NEEDLE + INSERT_AFTER, add_special_tokens=False)["input_ids"]
question_ids = tokenizer(QUESTION, add_special_tokens=False)["input_ids"]


def build_input(context_target, depth):
    p = int(depth * (context_target - len(needle_ids)))
    body = filler_ids * (context_target // len(filler_ids) + 1)
    body = body[:context_target - len(needle_ids)]
    context_ids = body[:p] + needle_ids + body[p:]
    for _ in range(3):
        context = tokenizer.decode(context_ids, skip_special_tokens=True)
        message = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": context + "\n" + QUESTION},
        ]
        ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt")
        if ids.shape[1] <= context_target:
            return ids
        delta = ids.shape[1] - context_target
        body = body[:max(0, len(body) - delta)]
        p = min(p, len(body))
        context_ids = body[:p] + needle_ids + body[p:]
    return ids


def make_configs():
    base = AutoConfig.from_pretrained(MODEL_ID)
    swa = AutoConfig.from_pretrained(MODEL_ID)
    swa.use_sliding_window = True
    swa.sliding_window = WINDOW
    swa.layer_types = [
        "sliding_attention" if v == 1 else "full_attention" for v in swa.no_rope_layers
    ]
    return base, swa


def run_eval(name, config):
    print(f"=== loading {name} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    model.eval()
    results = []
    oom_skip = False
    for length in LENGTHS:
        if oom_skip:
            break
        for depth in DEPTHS:
            if oom_skip:
                break
            torch.cuda.reset_peak_memory_stats()
            try:
                input_ids = build_input(length - 80, depth).to("cuda")
                t0 = time.time()
                with torch.inference_mode():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=MAX_NEW,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
                gen_time = time.time() - t0
                gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                record = {
                    "config": name,
                    "length": length,
                    "depth": depth,
                    "exact_match": ANSWER in gen,
                    "generation": gen.strip(),
                    "gen_time_s": round(gen_time, 2),
                    "gen_tokens": out.shape[1] - input_ids.shape[1],
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "status": "ok",
                }
            except torch.cuda.OutOfMemoryError as e:
                record = {"config": name, "length": length, "depth": depth, "status": "oom"}
                oom_skip = True
            except Exception as e:
                record = {"config": name, "length": length, "depth": depth, "status": "error", "err": str(e)[:200]}
            results.append(record)
            print(json.dumps(record), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    return model, results


base_cfg, swa_cfg = make_configs()
meta = {
    "model": MODEL_ID,
    "window": WINDOW,
    "baseline_layer_types": base_cfg.layer_types,
    "baseline_use_sliding_window": base_cfg.use_sliding_window,
    "swa_layer_types": swa_cfg.layer_types,
    "swa_use_sliding_window": swa_cfg.use_sliding_window,
    "rope_theta": base_cfg.rope_theta,
    "num_rope_layers": sum(1 for v in base_cfg.no_rope_layers if v == 1),
    "num_nope_layers": sum(1 for v in base_cfg.no_rope_layers if v == 0),
}

model_b, results_b = run_eval("baseline", base_cfg)
del model_b
gc.collect()
torch.cuda.empty_cache()

model_s, results_s = run_eval("rnope_swa", swa_cfg)
del model_s
gc.collect()
torch.cuda.empty_cache()

parity_checked = False
for r in results_b:
    if r.get("length") == 8192 and r.get("depth") == 0.1 and r.get("status") == "ok":
        gb = r["generation"]
        for rs in results_s:
            if rs.get("length") == 8192 and rs.get("depth") == 0.1 and rs.get("status") == "ok":
                parity_checked = gb == rs["generation"]
                break
        break

output = {"meta": meta, "mask_parity_at_8k_window_covered": parity_checked, "results": results_b + results_s}
with open("/kaggle/working/results.json", "w") as f:
    json.dump(output, f, indent=2)

print("=== SUMMARY ===", flush=True)
for name in ["baseline", "rnope_swa"]:
    rows = [r for r in output["results"] if r.get("config") == name and r.get("status") == "ok"]
    ooms = [r for r in output["results"] if r.get("config") == name and r.get("status") == "oom"]
    by_len = {}
    for r in rows:
        by_len.setdefault(r["length"], []).append(r)
    print(f"{name}: {len(rows)} ok, {len(ooms)} oom", flush=True)
    for length, rr in sorted(by_len.items()):
        acc = sum(r["exact_match"] for r in rr) / len(rr)
        mem = max(r["peak_mem_gb"] for r in rr)
        print(f"  len {length}: acc {acc:.2f} ({len(rr)} runs), peak mem {mem} GB", flush=True)
print(f"mask parity (8k context, window covers all): {parity_checked}", flush=True)
print("DONE", flush=True)