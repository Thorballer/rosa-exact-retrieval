"""Full-attention speed and KV-cache memory on this machine, measured.

Prefill: one forward over T tokens (the cost of reading a context).
Decode:  one token at a time with a KV cache, re-timed every 16 tokens.
KV:      bytes per token, read off the cache tensors, not from a formula.
"""
import json
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import common

MODEL = "Qwen/Qwen2.5-0.5B"
torch.set_num_threads(int(__import__("os").environ.get("THREADS", "6")))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    cfg = AutoConfig.from_pretrained(MODEL)
    ids = common.corpus_tokens("code", tok, 4096)
    rows = []

    with torch.inference_mode():
        for T in (128, 256, 512, 768, 1024, 1536, 2048, 3072):
            x = torch.tensor([ids[:T]])
            m(x)  # warm
            t0 = time.perf_counter()
            out = m(x, use_cache=True)
            dt = time.perf_counter() - t0
            kv = sum(t.numel() * t.element_size()
                     for layer in out.past_key_values for t in layer)
            rows.append({"kind": "prefill", "T": T, "seconds": dt,
                         "ms_per_token": 1e3 * dt / T, "kv_bytes_per_token": kv / T,
                         "kv_bytes_total": kv})
            print(json.dumps(rows[-1]), flush=True)

        # incremental decode from a prefilled context
        for T in (512, 1024, 2048):
            x = torch.tensor([ids[:T]])
            out = m(x, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            t0 = time.perf_counter()
            steps = 16
            for _ in range(steps):
                o = m(nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1:].argmax(-1)
            dt = time.perf_counter() - t0
            rows.append({"kind": "decode", "T": T, "seconds": dt,
                         "ms_per_token": 1e3 * dt / steps})
            print(json.dumps(rows[-1]), flush=True)

    rows.append({"kind": "model", "model": MODEL, "params": sum(p.numel() for p in m.parameters()),
                 "layers": cfg.num_hidden_layers, "kv_heads": cfg.num_key_value_heads,
                 "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
                 "dtype": "float32", "device": "cpu",
                 "threads": torch.get_num_threads(), "context_window": cfg.max_position_embeddings})
    common.emit(common.HERE / "results" / "attention-speed.jsonl", rows)


if __name__ == "__main__":
    main()
