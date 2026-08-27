# Qwen3.5-35B-A3B sizing + residency probe (Experiment 5, Phase A)
# Pre-registration: experiments/07-drafter-moe/EXPERIMENT_5.md
# Machine: Kaggle 2x NvidiaTeslaT4 (~15 GB usable each), pinned in metadata.
#
# Output contract (per standing rules): results.json written BEFORE "DONE".
import gc
import json
import os
import subprocess
import time

RESULTS_PATH = "/kaggle/working/results.json"
# ungated byte-identical mirror of gated Qwen/Qwen3.5-35B-A3B (v2 swap, 2026-08-26)
TARGET = "unsloth/Qwen3.5-35B-A3B"
DRAFTERS = ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-0.8B"]

results = {"phase": "A", "target": TARGET, "checks": {}, "env": {}}


def save():
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


# --- env scrubbing per pre-reg pins -------------------------------------
sh("pip uninstall -y torchao 2>/dev/null")
# v3: Kaggle image's transformers predates qwen3_5_moe arch (v2 failure mode)
sh("pip install -q -U transformers accelerate bitsandbytes")
try:
    import psutil  # noqa
except ImportError:
    sh("pip install -q psutil")

results["env"]["gpu"] = sh("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader").strip()
import psutil  # noqa: E402
results["env"]["host_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)

# --- HF auth (gated model) ----------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    results["env"]["hf_token"] = "loaded from kaggle secret"
except Exception as e:
    results["env"]["hf_token"] = f"secret unavailable: {str(e)[:120]}"
save()


# --- tokenizer vocab-match check (drafter pairing validity) -------------
from transformers import AutoTokenizer  # noqa: E402

tgt_tok = AutoTokenizer.from_pretrained(TARGET)
results["checks"]["target_tokenizer_len"] = len(tgt_tok)
for d in DRAFTERS:
    try:
        dtok = AutoTokenizer.from_pretrained(d)
        same = dtok.get_vocab() == tgt_tok.get_vocab()
        results["checks"][f"{d.replace('/', '__')}_vocab_match"] = bool(same)
        del dtok
        gc.collect()
    except Exception as e:
        results["checks"][f"{d.replace('/', '__')}_error"] = str(e)[:200]
    save()

# --- load target @ 4-bit across both T4s ---------------------------------
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # noqa: E402

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    TARGET,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
results["load_seconds"] = round(time.time() - t0, 1)
results["vram_after_load"] = sh(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
).strip()
results["ram_after_load_gb"] = round(psutil.Process().memory_info().rss / 1e9, 1)
results["moe_experts"] = {
    k: getattr(getattr(model.config, k.split("_")[-1], None) if hasattr(model.config, k.split("_")[-1]) else model.config, k, None).__str__()[:80]
    for k in dir(model.config)
    if any(s in k.lower() for s in ("expert", "routed", "topk"))
}
save()

# --- ttft probe ----------------------------------------------------------
prompt = "Explain what a mixture-of-experts layer is, in two sentences."
enc = tgt_tok(prompt, return_tensors="pt").to(model.device)
t1 = time.time()
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=32, do_sample=False)
results["ttft_and_gen_seconds"] = round(time.time() - t1, 1)
results["gen_sample"] = tgt_tok.decode(out[0][enc.input_ids.shape[1]:])[:300]
results["vram_final"] = sh(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
).strip()
save()

print(json.dumps(results, indent=1))
print("DONE")
