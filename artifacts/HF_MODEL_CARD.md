---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM3-3B
library_name: peft
tags:
- lora
- rnope
- sliding-window-attention
- long-context
---

# SmolLM3-RNoPE-SWA adapters

LoRA adapters for **HuggingFaceTB/SmolLM3-3B** trained as part of a
pre-registered experiment series on sliding-window attention in hybrid
RoPE/NoPE models. SmolLM3 has 27 RoPE layers + 9 NoPE layers; these adapters
test whether long-context retrieval survives capping the RoPE layers to an
8k attention window *when the LoRA is trained under that window*.

## Contents

| Folder | What it is |
|---|---|
| `treatment/` | LoRA (rank 32) trained **with** the 8k SWA window active on RoPE layers |
| `control/`   | Same recipe, LoRA trained **without** the window |

## Result summary

- Inference-time-only windowing of the stock model destroys past-window
  retrieval (needle-in-haystack 0/5 beyond 8k) despite being 11–21% faster.
- The **treatment** adapter restores needle-in-haystack retrieval to 5/5 at
  8k / 16k / 32k / 64k under windowed inference.
- Full pre-registration, kernels, and raw results JSONs:
  https://github.com/PastelRuntime/smollm3-research

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "PastelRuntime/SmolLM3-RNoPE-SWA-Adapters", subfolder="treatment")
```
