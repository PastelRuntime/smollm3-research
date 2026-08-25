import os

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
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torch", "torchvision", "torchaudio"]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1+cu124",
         "--index-url", "https://download.pytorch.org/whl/cu124"]
    )
    assert cuda_ok(), "CUDA still broken after torch downgrade"

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.57.6", "accelerate", "peft"]
)
r = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
    capture_output=True,
)
print("torchao removed" if r.returncode == 0 else "torchao not installed", flush=True)

import gc
import glob
import time
import traceback

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print(f"torch {torch.__version__} | cuda {torch.version.cuda} | transformers {transformers.__version__}", flush=True)
print(f"GPUs visible: {torch.cuda.device_count()}", flush=True)

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
filler_ids = tokenizer(FILLER, add_special_tokens=False)["input_ids"]
needle_ids = tokenizer(INSERT_BEFORE + NEEDLE + INSERT_AFTER, add_special_tokens=False)["input_ids"]

output = {"meta": {}, "results": []}


def save_output():
    with open("/kaggle/working/results.json", "w") as f:
        json.dump(output, f, indent=2)


def n_gpus():
    return torch.cuda.device_count()


def reset_peak():
    for i in range(n_gpus()):
        torch.cuda.reset_peak_memory_stats(i)


def peak_gb():
    return round(max(torch.cuda.max_memory_allocated(i) for i in range(n_gpus())) / 2**30, 2)


def log_mem(tag):
    parts = []
    for i in range(n_gpus()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        parts.append(f"gpu{i} free {free_b / 2**30:.2f}G peak {torch.cuda.max_memory_allocated(i) / 2**30:.2f}G")
    print(f"[mem {tag}] " + " | ".join(parts), flush=True)


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


def greedy_generate(model, input_ids, max_new, device):
    from transformers import DynamicCache

    cache = DynamicCache()
    n = input_ids.shape[1]
    logits = None
    for start in range(0, n, PREFILL_CHUNK):
        chunk = input_ids[:, start:start + PREFILL_CHUNK].to(device)
        out = model(input_ids=chunk, past_key_values=cache, use_cache=True)
        logits = out.logits
    generated = []
    cur = logits[:, -1:].argmax(dim=-1).to(device)
    for _ in range(max_new):
        token = cur.item()
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        out = model(input_ids=cur, past_key_values=cache, use_cache=True)
        cur = out.logits[:, -1:].argmax(dim=-1).to(device)
    return generated


def find_adapter(marker):
    hits = glob.glob(f"/kaggle/input/**/adapter_final/adapter_config.json", recursive=True)
    hits = [h for h in hits if marker in h]
    assert hits, f"no adapter_final found matching {marker}; scanned: {glob.glob('/kaggle/input/**/adapter_config.json', recursive=True)}"
    return hits[0].rsplit("/", 1)[0]


treat_adapter = find_adapter("phase")
ctrl_adapter = find_adapter("control")
print(f"treatment adapter: {treat_adapter}", flush=True)
print(f"control adapter:   {ctrl_adapter}", flush=True)
output["meta"].update({
    "model": MODEL_ID,
    "window": WINDOW,
    "treatment_adapter": treat_adapter,
    "control_adapter": ctrl_adapter,
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
})
save_output()


def make_config(kind):
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    if kind == "swa":
        cfg.use_sliding_window = True
        cfg.sliding_window = WINDOW
        cfg.layer_types = [
            "sliding_attention" if v == 1 else "full_attention" for v in cfg.no_rope_layers
        ]
    return cfg


CELLS = [
    {"cell": "A_control_weights_windowed_eval", "adapter": ctrl_adapter, "cfg": "swa"},
    {"cell": "B_treatment_weights_stock_eval", "adapter": treat_adapter, "cfg": "stock"},
]

for cell in CELLS:
    print(f"=== loading cell {cell['cell']} (adapter={cell['adapter']}, cfg={cell['cfg']}) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=make_config(cell["cfg"]),
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, cell["adapter"])
    model = model.to(torch.float16).merge_and_unload()
    model.eval()
    device = model.device
    log_mem(f"loaded {cell['cell']}")

    oom = False
    for length in LENGTHS:
        if oom:
            break
        for depth in DEPTHS:
            reset_peak()
            record = {"cell": cell["cell"], "config": cell["cfg"], "length": length, "depth": depth, "status": "unset"}
            try:
                input_ids = build_input(length - 80, depth)
                t0 = time.time()
                with torch.inference_mode():
                    tokens = greedy_generate(model, input_ids, MAX_NEW, device)
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
                log_mem(f"OOM at {length}/{depth}")
                record.update({"status": "oom", "peak_mem_gb": peak_gb(), "err": traceback.format_exc()[-800:]})
                oom = True
            except Exception:
                record.update({"status": "error", "peak_mem_gb": peak_gb(), "err": traceback.format_exc()[-800:]})
            output["results"].append(record)
            print(json.dumps(record), flush=True)
            save_output()
            gc.collect()
            for i in range(n_gpus()):
                torch.cuda.empty_cache()

    del model
    gc.collect()
    for i in range(n_gpus()):
        torch.cuda.empty_cache()
    log_mem(f"unloaded {cell['cell']}")

print("=== 2x2 CLOSE-OUT SUMMARY ===", flush=True)
ok = [r for r in output["results"] if r.get("status") == "ok"]
print(f"runs ok: {len(ok)}/{len(output['results'])}", flush=True)
by_cell = {}
for r in ok:
    by_cell.setdefault(r["cell"], {}).setdefault(r["length"], []).append(r)
for cell_name in sorted(by_cell):
    for length in sorted(by_cell[cell_name]):
        rr = by_cell[cell_name][length]
        acc = sum(r["exact_match"] for r in rr)
        print(f"  {cell_name} | {length//1024:>3}k: {acc}/{len(rr)}", flush=True)
print("DONE", flush=True)