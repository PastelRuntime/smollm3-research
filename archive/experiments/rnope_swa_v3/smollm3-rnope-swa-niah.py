import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import subprocess
import sys


def cuda_ok():
    probe = (
        "import torch; "
        "assert torch.cuda.is_available(), 'no cuda device'; "
        "x = (torch.ones(8, device='cuda') + 1).sum().item(); "
        "assert x == 16.0, x"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    return r.returncode == 0


if not cuda_ok():
    print("stock torch cannot launch CUDA kernels on this GPU; installing torch 2.4.1+cu124...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torch", "torchvision", "torchaudio"]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1+cu124",
         "--index-url", "https://download.pytorch.org/whl/cu124"]
    )
    assert cuda_ok(), "CUDA still broken after torch downgrade"

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "accelerate"]
)

import gc
import time
import traceback

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

print(f"torch {torch.__version__} | cuda {torch.version.cuda} | transformers {transformers.__version__}", flush=True)
print(f"GPUs visible: {torch.cuda.device_count()}", flush=True)
for i in range(torch.cuda.device_count()):
    free_b, total_b = torch.cuda.mem_get_info(i)
    print(
        f"  cuda:{i} {torch.cuda.get_device_name(i)} | sm_{torch.cuda.get_device_capability(i)[0]}{torch.cuda.get_device_capability(i)[1]} | {free_b / 2**30:.2f} GiB free of {total_b / 2**30:.2f} GiB",
        flush=True,
    )
subprocess.run(["nvidia-smi"], check=False)

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
PREFILL_CHUNK = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
pad_id = tokenizer.eos_token_id

filler_ids = tokenizer(FILLER, add_special_tokens=False)["input_ids"]
needle_ids = tokenizer(INSERT_BEFORE + NEEDLE + INSERT_AFTER, add_special_tokens=False)["input_ids"]


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


def n_gpus():
    return torch.cuda.device_count()


def log_mem(tag):
    parts = []
    for i in range(n_gpus()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        parts.append(
            f"gpu{i} free {free_b / 2**30:.2f}G alloc {torch.cuda.memory_allocated(i) / 2**30:.2f}G peak {torch.cuda.max_memory_allocated(i) / 2**30:.2f}G"
        )
    print(f"[mem {tag}] " + " | ".join(parts), flush=True)


def reset_peak():
    for i in range(n_gpus()):
        torch.cuda.reset_peak_memory_stats(i)


def peak_gb():
    return round(max(torch.cuda.max_memory_allocated(i) for i in range(n_gpus())) / 2**30, 2)


def make_configs():
    base = AutoConfig.from_pretrained(MODEL_ID)
    swa = AutoConfig.from_pretrained(MODEL_ID)
    swa.use_sliding_window = True
    swa.sliding_window = WINDOW
    swa.layer_types = [
        "sliding_attention" if v == 1 else "full_attention" for v in swa.no_rope_layers
    ]
    return base, swa


def save_results(payload):
    with open("/kaggle/working/results.json", "w") as f:
        json.dump(payload, f, indent=2)


def load_model(name, config):
    print(f"=== loading {name} (fp16, sdpa, device_map=auto across {n_gpus()} GPUs) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    p0 = next(model.parameters())
    print(f"{name}: weights loaded as {p0.dtype}", flush=True)
    if p0.dtype != torch.float16:
        print(f"{name}: WARNING dtype was not fp16, converting now", flush=True)
        model = model.to(torch.float16)
        print(f"{name}: weights now {next(model.parameters()).dtype}", flush=True)
    model.eval()
    hf_map = getattr(model, "hf_device_map", {})
    placement = {}
    for layer_idx, dev in hf_map.items():
        placement.setdefault(str(dev), []).append(layer_idx)
    for dev, layers in placement.items():
        print(f"{name}: {len(layers)} modules on {dev}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params:,} params | weights dtype {p0.dtype}", flush=True)
    print(
        f"{name}: layer_types sliding={sum(1 for t in config.layer_types if t == 'sliding_attention')} full={sum(1 for t in config.layer_types if t == 'full_attention')} | use_sliding_window={config.use_sliding_window}",
        flush=True,
    )
    log_mem(f"{name} after load")
    return model


def greedy_generate(model, input_ids, max_new):
    from transformers import DynamicCache

    input_device = model.device
    cache = DynamicCache()
    n = input_ids.shape[1]
    logits = None
    for start in range(0, n, PREFILL_CHUNK):
        chunk = input_ids[:, start:start + PREFILL_CHUNK].to(input_device)
        out = model(input_ids=chunk, past_key_values=cache, use_cache=True)
        logits = out.logits
    generated = []
    cur = logits[:, -1:].argmax(dim=-1).to(input_device)
    for _ in range(max_new):
        token = cur.item()
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        out = model(input_ids=cur, past_key_values=cache, use_cache=True)
        cur = out.logits[:, -1:].argmax(dim=-1).to(input_device)
    return generated


def eval_loop(model, name, config, output):
    results = []
    oom = False
    for length in LENGTHS:
        if oom:
            break
        for depth in DEPTHS:
            reset_peak()
            record = {"config": name, "length": length, "depth": depth, "status": "unset"}
            try:
                input_ids = build_input(length - 80, depth)
                print(f"{name} len {length} depth {depth}: input_ids {tuple(input_ids.shape)}", flush=True)
                t0 = time.time()
                with torch.inference_mode():
                    tokens = greedy_generate(model, input_ids, MAX_NEW)
                gen_time = time.time() - t0
                gen = tokenizer.decode(tokens, skip_special_tokens=True)
                record.update({
                    "exact_match": ANSWER in gen,
                    "generation": gen.strip()[:200],
                    "gen_time_s": round(gen_time, 2),
                    "gen_tokens": len(tokens),
                    "peak_mem_gb": peak_gb(),
                    "status": "ok",
                })
            except torch.cuda.OutOfMemoryError:
                log_mem(f"{name} OOM at {length}/{depth}")
                record.update({
                    "status": "oom",
                    "peak_mem_gb": peak_gb(),
                    "err": traceback.format_exc()[-800:],
                })
                print(f"{name}: OOM at length {length}; longer lengths will also OOM, stopping this config", flush=True)
                oom = True
            except Exception:
                record.update({
                    "status": "error",
                    "peak_mem_gb": peak_gb(),
                    "err": traceback.format_exc()[-800:],
                })
            results.append(record)
            print(json.dumps(record), flush=True)
            output["results"].append(record)
            save_results(output)
            gc.collect()
            for i in range(n_gpus()):
                torch.cuda.empty_cache()
    return results


def run_eval(name, config, output):
    model = load_model(name, config)
    results = eval_loop(model, name, config, output)
    del model
    gc.collect()
    for i in range(n_gpus()):
        torch.cuda.empty_cache()
    return results


base_cfg, swa_cfg = make_configs()
meta = {
    "model": MODEL_ID,
    "window": WINDOW,
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "baseline_layer_types": base_cfg.layer_types,
    "baseline_use_sliding_window": base_cfg.use_sliding_window,
    "swa_layer_types": swa_cfg.layer_types,
    "swa_use_sliding_window": swa_cfg.use_sliding_window,
    "rope_theta": base_cfg.rope_theta,
    "num_rope_layers": sum(1 for v in base_cfg.no_rope_layers if v == 1),
    "num_nope_layers": sum(1 for v in base_cfg.no_rope_layers if v == 0),
}
output = {"meta": meta, "results": []}
save_results(output)

try:
    results_b = run_eval("baseline", base_cfg, output)
except Exception:
    print(f"baseline config failed entirely:\n{traceback.format_exc()}", flush=True)
    results_b = []
save_results(output)

try:
    results_s = run_eval("rnope_swa", swa_cfg, output)
except Exception:
    print(f"rnope_swa config failed entirely:\n{traceback.format_exc()}", flush=True)
    results_s = []
save_results(output)

parity_checked = None
for r in results_b:
    if r.get("length") == 8192 and r.get("depth") == 0.1 and r.get("status") == "ok":
        for rs in results_s:
            if rs.get("length") == 8192 and rs.get("depth") == 0.1 and rs.get("status") == "ok":
                parity_checked = r["generation"] == rs["generation"]
                break
        break

output["mask_parity_at_8k_window_covered"] = parity_checked
save_results(output)

print("=== SUMMARY ===", flush=True)
for name, res in [("baseline", results_b), ("rnope_swa", results_s)]:
    rows = [r for r in res if r.get("status") == "ok"]
    ooms = [r for r in res if r.get("status") == "oom"]
    errs = [r for r in res if r.get("status") == "error"]
    by_len = {}
    for r in rows:
        by_len.setdefault(r["length"], []).append(r)
    print(f"{name}: {len(rows)} ok, {len(ooms)} oom, {len(errs)} error", flush=True)
    for length, rr in sorted(by_len.items()):
        acc = sum(r["exact_match"] for r in rr) / len(rr)
        mem = max(r["peak_mem_gb"] for r in rr)
        t = max(r["gen_time_s"] for r in rr)
        print(f"  len {length}: acc {acc:.2f} ({len(rr)} runs), peak mem {mem} GB, max gen time {t}s", flush=True)
    for r in errs:
        print(f"  ERROR len {r['length']} depth {r['depth']}: {r.get('err', '')[-200:]}", flush=True)
print(f"mask parity (8k context, window covers all): {parity_checked}", flush=True)
print("DONE", flush=True)