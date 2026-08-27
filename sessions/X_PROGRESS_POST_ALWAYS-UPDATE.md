# X Post — SmolLM3 Research Progress Log

> Draft for posting on X. First-person, unpolished but organized. Edits welcome. This is not my first expirment. ALWAYS REPLACE BELOW. Dont overshare. Keep the post to the experiment itself. Share some quick humor or personality sometimes.... 

---

**Experiment 1 on SmolLM3: done.** 40/40 runs, clean data, and the result was sharper than I expected.

Quick setup: SmolLM3 has 27 RoPE layers + 9 NoPE layers. The RNoPE paper says windowing the RoPE layers (only those, not the NoPE ones) should give long-context retrieval *and* big memory savings. Nobody had tested that recipe on this model. So I did — needle-in-haystack at 8k/16k/32k/64k, baseline vs windowed, identical everything else.

**What happened:**

- Baseline: perfect retrieval at every length and depth. 5/5, 5/5, 5/5, 5/5.
- Windowed: perfect at 8k (exactly the window size)... then **0/5 everywhere past it**
- But the windowed runs were 11-21% faster and used less memory the whole way

So the honest takeaway: you can't just bolt the paper's recipe onto an already-trained model. The window has to exist *during training*. The speed win is real though — which is exactly the carrot.

**Next up (Phase B):** LoRA fine-tune WITH the windowed config, then re-run this exact eval. Two outcomes, both publishable:
1. Retrieval recovers + speed stays → "config-only long-context efficiency for SmolLM3"
2. It doesn't → "RNoPE-SWA doesn't transfer to SmolLM3, here's the ablation"

Pre-registered before running, so I can't retrofit a story onto whatever comes out.

(For anyone counting at home: it took 8 failed kernel versions to get one clean run. The model was never the problem. The environment was. Every. Single. Time. I now have opinions about P100s.)

Confirmation run going across both GPUs now, then Phase B starts. If you're into NoPE/hybrid architectures — my DMs are open. 🤝