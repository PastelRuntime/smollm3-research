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
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.57.6", "accelerate", "peft", "datasets"]
)

import gc
import math
import time
import traceback

import torch
import torch.nn.functional as F
import transformers
from torch.utils.checkpoint import checkpoint as ckpt
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
SEQ_LEN = 16384
N_BLOCKS = 400
GRAD_ACCUM = 4
LORA_R = 32
LORA_ALPHA = 64
LR = 2e-4
WARMUP_STEPS = 10
CE_CHUNK = 2048
MAX_BOOKS = 300

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

output = {"meta": {}, "train_log": [], "sanity_generation": None, "results": []}


def save_output():
    with open("/kaggle/working/results.json", "w") as f:
        json.dump(output, f, indent=2)


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


print("=== preparing data (pg19 streaming) ===", flush=True)
t0 = time.time()
from datasets import load_dataset

blocks = []
buf = []
n_books = 0
stream = load_dataset("deepmind/pg19", split="train", streaming=True)
for d in stream:
    text = None
    for key in ("book_text", "text", "book"):
        if isinstance(d.get(key), str) and len(d[key]) > 1000:
            text = d[key]
            break
    if text is None:
        continue
    n_books += 1
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    buf.extend(ids)
    buf.append(tokenizer.eos_token_id)
    while len(buf) >= SEQ_LEN and len(blocks) < N_BLOCKS:
        blocks.append(buf[:SEQ_LEN])
        buf = buf[SEQ_LEN:]
    if len(blocks) >= N_BLOCKS or n_books >= MAX_BOOKS:
        break
print(f"prepared {len(blocks)} blocks of {SEQ_LEN} tokens from {n_books} books in {time.time()-t0:.0f}s", flush=True)
assert len(blocks) >= 100, f"not enough training data: {len(blocks)} blocks"
output["meta"]["blocks"] = len(blocks)
output["meta"]["books"] = n_books
output["meta"]["train_tokens_m"] = round(len(blocks) * SEQ_LEN / 1e6, 2)

swa_cfg = AutoConfig.from_pretrained(MODEL_ID)
swa_cfg.use_sliding_window = True
swa_cfg.sliding_window = WINDOW
swa_cfg.layer_types = [
    "sliding_attention" if v == 1 else "full_attention" for v in swa_cfg.no_rope_layers
]
output["meta"].update({
    "model": MODEL_ID,
    "window": WINDOW,
    "seq_len": SEQ_LEN,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lr": LR,
    "grad_accum": GRAD_ACCUM,
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "swa_layer_types": swa_cfg.layer_types,
    "rope_theta": swa_cfg.rope_theta,
    "num_rope_layers": sum(1 for v in swa_cfg.no_rope_layers if v == 1),
    "num_nope_layers": sum(1 for v in swa_cfg.no_rope_layers if v == 0),
})
save_output()

print("=== loading model (fp16, sdpa, device_map=auto) ===", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=swa_cfg,
    dtype=torch.float16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
for n, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()
causal = model.get_base_model()
backbone = causal.model
lm_head = causal.lm_head
input_device = backbone.embed_tokens.weight.device
print(f"input device: {input_device}", flush=True)
log_mem("after lora wrap")

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
total_steps = math.ceil(len(blocks) / GRAD_ACCUM)


def lr_lambda(step):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda")


def block_loss(block_ids):
    input_ids = torch.tensor([block_ids], dtype=torch.long, device=input_device)
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = backbone(input_ids=input_ids, use_cache=False).last_hidden_state
    total = None
    for s in range(0, SEQ_LEN - 1, CE_CHUNK):
        e = min(s + CE_CHUNK, SEQ_LEN - 1)
        h = hidden[:, s:e]
        t = input_ids[:, s + 1:e + 1]

        def chunk_ce(h_, t_):
            logits = lm_head(h_)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                t_.reshape(-1),
                reduction="sum",
            )

        part = ckpt(chunk_ce, h, t, use_reentrant=False)
        total = part if total is None else total + part
    return total / (SEQ_LEN - 1)


print(f"=== training: {len(blocks)} blocks, {total_steps} optimizer steps ===", flush=True)
model.train()
train_start = time.time()
optimizer.zero_grad(set_to_none=True)
for bi, block in enumerate(blocks):
    reset_peak()
    loss = block_loss(block)
    scaler.scale(loss / GRAD_ACCUM).backward()
    loss_val = loss.item()
    if (bi + 1) % GRAD_ACCUM == 0 or bi == len(blocks) - 1:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    lr_now = scheduler.get_last_lr()[0]
    entry = {
        "block": bi,
        "loss": round(loss_val, 4),
        "lr": round(lr_now, 6),
        "elapsed_s": round(time.time() - train_start, 1),
        "peak_mem_gb": peak_gb(),
    }
    output["train_log"].append(entry)
    if bi % 10 == 0 or bi == len(blocks) - 1:
        print(json.dumps(entry), flush=True)
        save_output()
    if bi == len(blocks) // 2:
        model.save_pretrained("/kaggle/working/adapter_mid")
        print("saved mid checkpoint", flush=True)
print(f"training done in {(time.time() - train_start) / 60:.1f} min", flush=True)
log_mem("after training")

model.save_pretrained("/kaggle/working/adapter_final")
print("saved final adapter", flush=True)

losses = [e["loss"] for e in output["train_log"]]
output["meta"]["loss_first10_avg"] = round(sum(losses[:10]) / min(10, len(losses)), 4)
output["meta"]["loss_last10_avg"] = round(sum(losses[-10:]) / min(10, len(losses)), 4)
save_output()

model.eval()


def greedy_generate(input_ids, max_new):
    from transformers import DynamicCache

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


print("=== sanity generation (chat format survived?) ===", flush=True)
try:
    message = [{"role": "user", "content": "Explain why the sky is blue, in one short paragraph."}]
    ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt")
    with torch.inference_mode():
        tokens = greedy_generate(ids, 80)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    output["sanity_generation"] = text[:400]
    print(f"sanity: {text[:300]}", flush=True)
except Exception:
    print(f"sanity generation failed:\n{traceback.format_exc()}", flush=True)
    output["sanity_generation"] = f"FAILED: {traceback.format_exc()[-300:]}"
save_output()

print("=== NIAH eval (Phase A harness, adapter active) ===", flush=True)
oom = False
for length in LENGTHS:
    if oom:
        break
    for depth in DEPTHS:
        reset_peak()
        record = {"config": "rnope_swa_lora", "length": length, "depth": depth, "status": "unset"}
        try:
            input_ids = build_input(length - 80, depth)
            print(f"len {length} depth {depth}: input_ids {tuple(input_ids.shape)}", flush=True)
            t0 = time.time()
            with torch.inference_mode():
                tokens = greedy_generate(input_ids, MAX_NEW)
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

print("=== PHASE B SUMMARY ===", flush=True)
rows = [r for r in output["results"] if r.get("status") == "ok"]
by_len = {}
for r in rows:
    by_len.setdefault(r["length"], []).append(r)
print(f"runs ok: {len(rows)} | oom/error: {len(output['results']) - len(rows)}", flush=True)
print(f"train loss: first10 {output['meta']['loss_first10_avg']} -> last10 {output['meta']['loss_last10_avg']}", flush=True)
phase_a_swa = {8192: 1.0, 16384: 0.0, 32768: 0.0, 65536: 0.0}
for length, rr in sorted(by_len.items()):
    acc = sum(r["exact_match"] for r in rr) / len(rr)
    t = max(r["gen_time_s"] for r in rr)
    print(
        f"  len {length}: acc {acc:.2f} ({len(rr)} runs) | max gen {t}s | phase A (pre-finetune) {phase_a_swa.get(length, '?')}",
        flush=True,
    )
print("DONE", flush=True)