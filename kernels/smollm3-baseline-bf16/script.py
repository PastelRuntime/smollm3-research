import json, time, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
results = {"kernel": "smollm3-baseline-bf16", "model": MODEL_ID, "precision": "bf16"}
SAVED = False

def save():
    global SAVED
    with open("/kaggle/working/results.json", "w") as f:
        json.dump(results, f, indent=2)
    SAVED = True

print("loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
model.eval()
results["device"] = torch.cuda.get_device_name(0)
results["params"] = sum(p.numel() for p in model.parameters())
results["params_human"] = f'{results["params"]:,}'
results["weights_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print("device:", results["device"], "| params:", results["params_human"])
save()

TOOLS = [
    {"name": "get_weather", "description": "Get the current weather in a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}},
    {"name": "search_notes", "description": "Search the user's personal notes",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}},
    {"name": "set_reminder", "description": "Set a reminder for the user",
     "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "What to remind about"},
                                                     "minutes_from_now": {"type": "integer", "description": "Delay in minutes"}},
                    "required": ["text"]}},
]

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def chat(messages, tools=None, max_new=150, temp=0.0):
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if tools:
        kwargs["xml_tools"] = tools
    text = tok.apply_chat_template(messages, **kwargs)
    ids = tok([text], return_tensors="pt").to(model.device)
    t = time.time()
    gen_kwargs = dict(max_new_tokens=max_new, do_sample=temp > 0)
    if temp > 0:
        gen_kwargs.update(temperature=temp, top_p=0.95)
    out = model.generate(**ids, **gen_kwargs)
    dt = time.time() - t
    new_tokens = out.shape[1] - ids.input_ids.shape[1]
    decoded = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return decoded, new_tokens / dt if dt > 0 else 0.0

def parse_tool(text):
    m = TOOL_RE.search(text)
    if not m:
        return {"parsed": False, "err": "no <tool_call> block"}
    try:
        j = json.loads(m.group(1))
        return {"parsed": True, "name": j.get("name"), "args": j.get("arguments")}
    except Exception as e:
        return {"parsed": False, "err": f"json: {e}", "raw": m.group(1)[:150]}

# --- S1: sanity generation (no tools) ---
try:
    text, tps = chat([{"role": "user", "content": "Explain why the sky is blue, in one short paragraph."}], max_new=120)
    results["s1_sanity"] = {"tokens_per_s": round(tps, 1), "text": text.strip()[:300]}
    print("S1 sanity:", results["s1_sanity"]["tokens_per_s"], "tok/s |", text.strip()[:80])
except Exception as e:
    results["s1_sanity"] = {"error": str(e)}
save()

# --- T1: tool-call format (3 prompts x 1 expected tool each) ---
T1 = [
    ("What's the weather in Copenhagen right now?", "get_weather"),
    ("Remind me in 20 minutes to check the oven.", "set_reminder"),
    ("Find my notes about the Lisbon trip.", "search_notes"),
]
t1_cases = []
for prompt, expected in T1:
    try:
        text, tps = chat([{"role": "user", "content": prompt}], tools=TOOLS, max_new=150)
        p = parse_tool(text)
        p.update({"prompt": prompt, "expected": expected,
                  "correct_name": p.get("name") == expected, "tokens_per_s": round(tps, 1),
                  "raw_head": text.strip()[:200]})
        t1_cases.append(p)
    except Exception as e:
        t1_cases.append({"prompt": prompt, "expected": expected, "error": str(e)})
results["t1_format"] = {"n": len(t1_cases),
                        "parsed_ok": sum(1 for c in t1_cases if c.get("parsed")),
                        "correct_name": sum(1 for c in t1_cases if c.get("correct_name")),
                        "cases": t1_cases}
print("T1 format:", results["t1_format"]["parsed_ok"], "/", len(t1_cases), "parsed,",
      results["t1_format"]["correct_name"], "/", len(t1_cases), "correct name")
save()

# --- T2: refusal path (tools installed, none needed) ---
try:
    text, _ = chat([{"role": "user", "content": "Write a haiku about rain. No tools needed, just write it."}], tools=TOOLS, max_new=100)
    results["t2_no_tool"] = {"emitted_tool_call": "<tool_call>" in text, "text": text.strip()[:200]}
    print("T2 no-tool:", "tool_call emitted (BAD)" if results["t2_no_tool"]["emitted_tool_call"] else "clean text (GOOD)")
except Exception as e:
    results["t2_no_tool"] = {"error": str(e)}
save()

# --- T3: argument quality on the weather call ---
try:
    text, _ = chat([{"role": "user", "content": "What's the weather in Copenhagen right now?"}], tools=TOOLS, max_new=150)
    p = parse_tool(text)
    args_ok = False
    if p.get("parsed") and isinstance(p.get("args"), dict):
        args_ok = isinstance(p["args"].get("city"), str) and len(p["args"]) <= 2
    results["t3_args"] = {"args_valid": args_ok, "detail": p}
    print("T3 args:", "valid (GOOD)" if args_ok else "check detail")
except Exception as e:
    results["t3_args"] = {"error": str(e)}
save()

# --- T4: multi-turn flow (call -> result -> react) ---
try:
    msgs = [{"role": "user", "content": "What's the weather in Copenhagen right now?"}]
    text, _ = chat(msgs, tools=TOOLS, max_new=150)
    p = parse_tool(text)
    if p.get("parsed") and p.get("name") == "get_weather":
        msgs.append({"role": "assistant", "content": text.strip()})
        msgs.append({"role": "tool", "name": "get_weather",
                     "content": json.dumps({"city": "Copenhagen", "temp_c": 12, "condition": "light rain"})})
        final, _ = chat(msgs, tools=TOOLS, max_new=150)
        results["t4_flow"] = {"first_call_ok": True, "final_mentions_weather": "rain" in final.lower() or "12" in final,
                              "final_head": final.strip()[:250]}
    else:
        results["t4_flow"] = {"first_call_ok": False, "detail": p}
    print("T4 flow:", results["t4_flow"].get("first_call_ok"))
except Exception as e:
    results["t4_flow"] = {"error": str(e)}
save()

results["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
results["wall_time_s"] = round(time.time() - T0, 1)
save()

assert SAVED and open("/kaggle/working/results.json").read(1) == "{"
print("DONE - results.json written and verified,", results["wall_time_s"], "s")
