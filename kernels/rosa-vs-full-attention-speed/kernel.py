"""Speed half of the rosa-vs-full-attention experiment, on a T4.

Split out because the recall half (kernels/rosa-vs-full-attention) already produced all of its
rows; this kernel only measures cost. Two numbers per cell: how long a token takes, and how many
bytes of state it needs.

Attention: prefill over T tokens, incremental decode with a KV cache, and the attention core on
its own (one query against T keys, once per layer, so the 32k-context cost is measurable without
running a 32k forward pass). ROSA: the same C code used everywhere else in this repo.
"""
import glob
import json
import os
import subprocess
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    open(OUT, "a").write(line + "\n")


ROSA_C = r"""
// Suffix-automaton next-token predictor.
//   lookup : global open-addressed hash (state,sym)->edge   O(1) even for big alphabets
//   clone  : per-state adjacency list, so cloning never scans the table
//   mode 0 : recency  -- stamp the whole suffix chain with position i (exact ROSA;
//                        O(chain length) per token => O(n^2) on repetitive input)
//   mode 1 : first-occurrence -- no stamping at all; the deepest chain state whose
//                        first end position is earlier is the match. O(1) typical,
//                        and the quadratic disappears.
//
// Returns 0 on success, 1 on allocation failure, 2 if a capacity bound would be
// exceeded (the arrays are sized for the theoretical maxima; a guard turns the
// adversarial case into an error instead of a silent overrun).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EMPTY (-1)

static inline uint32_t hs(int32_t s, int32_t sym) {
    uint32_t h = (uint32_t)s * 2654435761u ^ (uint32_t)sym * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

int build_src_c(const int32_t *x, int n, int32_t *src, int mode) {
    // states: <= 2n-1 (root included).  edges: at most 3n-4 transitions, but this
    // implementation stores a separate entry for every clone copy of a transition,
    // so it needs the larger 4n+24 allowance; the guard below covers the rest.
    int maxs = 2 * n + 2, maxe = 4 * n + 24;
    uint32_t hcap = 1;
    while ((int)hcap < 6 * n + 16) hcap <<= 1;
    uint32_t hmask = hcap - 1;

    int32_t *link = calloc(maxs, 4), *len = calloc(maxs, 4), *end = calloc(maxs, 4);
    int32_t *head = malloc(maxs * 4), *esym = malloc(maxe * 4), *eto = malloc(maxe * 4),
            *enext = malloc(maxe * 4);
    int32_t *hkst = malloc(hcap * 4), *hksym = malloc(hcap * 4), *he = malloc(hcap * 4);
    if (!link || !len || !end || !head || !esym || !eto || !enext || !hkst || !hksym || !he) {
        free(link); free(len); free(end); free(head);
        free(esym); free(eto); free(enext); free(hkst); free(hksym); free(he);
        return 1;
    }
    for (int i = 0; i < maxs; i++) { head[i] = -1; link[i] = -1; end[i] = -1; len[i] = 0; }
    for (uint32_t i = 0; i < hcap; i++) he[i] = EMPTY;

    int ne = 0, last = 0, z = 1;

    for (int i = 0; i < n; i++) {
        int32_t t = x[i];
        if (z + 1 >= maxs) { goto fail; }
        int cur = z++;
        len[cur] = len[last] + 1; end[cur] = mode ? i : -1; head[cur] = -1;

        int p = last;
        while (p != -1) {
            uint32_t h = hs(p, t) & hmask, probe = 0;
            int e = EMPTY;
            while (he[h] != EMPTY && ++probe <= hcap) {
                if (hkst[h] == p && hksym[h] == t) { e = he[h]; break; }
                h = (h + 1) & hmask;
            }
            if (probe > hcap) goto fail;            // table full
            if (e != EMPTY) break;
            // insert edge (p, t) -> cur
            if (ne + 1 > maxe) goto fail;
            esym[ne] = t; eto[ne] = cur; enext[ne] = head[p]; head[p] = ne;
            hkst[h] = p; hksym[h] = t; he[h] = ne; ne++;
            p = link[p];
        }
        if (p == -1) {
            link[cur] = 0;
        } else {
            uint32_t h = hs(p, t) & hmask, probe = 0;
            int q = -1;
            while (he[h] != EMPTY && ++probe <= hcap) {
                if (hkst[h] == p && hksym[h] == t) { q = eto[he[h]]; break; }
                h = (h + 1) & hmask;
            }
            if (q == -1) goto fail;                 // inconsistent: extension loop found it
            if (len[p] + 1 == len[q]) {
                link[cur] = q;
            } else {
                if (z + 1 >= maxs) goto fail;
                int u = z++;
                len[u] = len[p] + 1; link[u] = link[q]; end[u] = end[q]; head[u] = -1;
                for (int e2 = head[q]; e2 != -1; e2 = enext[e2]) {
                    // each cloned transition is a NEW storage entry, so the edge array needs
                    // room for (transitions + sum of clone out-degrees)
                    if (ne + 1 > maxe) goto fail;
                    esym[ne] = esym[e2]; eto[ne] = eto[e2]; enext[ne] = head[u]; head[u] = ne;
                    uint32_t hh = hs(u, esym[e2]) & hmask, pr = 0;
                    while (he[hh] != EMPTY && ++pr <= hcap) hh = (hh + 1) & hmask;
                    if (pr > hcap) goto fail;
                    hkst[hh] = u; hksym[hh] = esym[e2]; he[hh] = ne; ne++;
                }
                while (p != -1) {
                    uint32_t hh = hs(p, t) & hmask, pr = 0;
                    int hit = EMPTY;
                    while (he[hh] != EMPTY && ++pr <= hcap) {
                        if (hkst[hh] == p && hksym[hh] == t) { hit = he[hh]; break; }
                        hh = (hh + 1) & hmask;
                    }
                    if (hit == EMPTY || eto[hit] != q) break;
                    eto[hit] = u;
                    p = link[p];
                }
                link[q] = u; link[cur] = u;
            }
        }

        if (mode == 0) {
            int v = cur, found = -1;
            while (v != -1) {
                if (found < 0 && len[v] > 0 && end[v] >= 0) found = end[v] + 1;
                end[v] = i;
                v = link[v];
            }
            src[i] = found;
        } else {
            int v = cur, found = -1;
            while (v != -1) {
                if (len[v] > 0 && end[v] >= 0 && end[v] < i) { found = end[v] + 1; break; }
                v = link[v];
            }
            src[i] = found;
        }
        last = cur;
    }
    free(link); free(len); free(end); free(head);
    free(esym); free(eto); free(enext); free(hkst); free(hksym); free(he);
    return 0;

fail:  // capacity guard: refuse rather than corrupt memory
    free(link); free(len); free(end); free(head);
    free(esym); free(eto); free(enext); free(hkst); free(hksym); free(he);
    return 2;
}

"""


def build_rosa():
    src = "/kaggle/working/rosa.c"
    open(src, "w").write(ROSA_C)
    so = "/kaggle/working/librosa.so"
    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", so, src], check=True)
    import ctypes
    h = ctypes.CDLL(so)
    h.build_src_c.restype = ctypes.c_int
    h.build_src_c.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    return h


def rosa_sources(lib, ids, mode=1):
    import ctypes
    n = len(ids)
    x = (ctypes.c_int32 * n)(*ids)
    out = (ctypes.c_int32 * n)()
    rc = lib.build_src_c(x, n, out, mode)
    if rc != 0:
        raise RuntimeError(f"build_src_c rc={rc}")
    return list(out)


def kv_bytes(past):
    """Works for legacy tuples and for Cache objects whose entries may contain None."""
    total = 0
    for layer in past:
        for t in (layer if isinstance(layer, (tuple, list)) else (layer,)):
            if t is not None and hasattr(t, "numel"):
                total += t.numel() * t.element_size()
    return total


def main():
    log({"exp": "env", "torch": torch.__version__, "device": DEV,
         "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None})
    tok = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(DEV).eval()
    cfg = AutoConfig.from_pretrained(MODEL)
    lib = build_rosa()

    files = sorted(glob.glob(os.path.join(os.path.dirname(__import__("transformers").__file__),
                                          "**", "*.py"), recursive=True))[:40]
    text = "\n\n".join(open(f, errors="ignore").read() for f in files)
    ids = tok(text, add_special_tokens=False)["input_ids"][:1_100_000]
    log({"exp": "corpus", "corpus": "code", "tokens": len(ids)})

    for T in (512, 1024, 2048, 4096, 8192):
        x = torch.tensor([ids[:T]], device=DEV)
        with torch.inference_mode():
            model(x, use_cache=True)
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x, use_cache=True)
            if DEV == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            kv = kv_bytes(out.past_key_values)
        log({"exp": "speed", "arm": "attention", "kind": "prefill", "T": T, "device": DEV,
             "dtype": str(model.dtype), "seconds": dt, "ms_per_token": 1e3 * dt / T,
             "kv_bytes_per_token": kv / T, "kv_bytes_total": kv})
        del out
        if DEV == "cuda":
            torch.cuda.empty_cache()

    for T in (1024, 4096):
        x = torch.tensor([ids[:T]], device=DEV)
        with torch.inference_mode():
            out = model(x, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        steps = 32
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(steps):
                o = model(nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1:].argmax(-1)
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        log({"exp": "speed", "arm": "attention", "kind": "decode", "T": T, "device": DEV,
             "dtype": str(model.dtype), "ms_per_token": 1e3 * dt / steps})
        del past, out
        if DEV == "cuda":
            torch.cuda.empty_cache()

    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    HK = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    dt_ = model.dtype
    for T in (1024, 4096, 16384, 32768):
        q = torch.randn(H, 1, D, device=DEV, dtype=dt_)
        k = torch.randn(H, T, D, device=DEV, dtype=dt_)
        v = torch.randn(H, T, D, device=DEV, dtype=dt_)
        reps = 20
        try:
            gqa = (H != HK)
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(reps):
                    torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=gqa)
            if DEV == "cuda":
                torch.cuda.synchronize()
            per_layer = (time.perf_counter() - t0) / reps
            log({"exp": "speed", "arm": "attention-core", "kind": "sdpa_1_query", "T": T,
                 "layers": L, "us_per_layer": 1e6 * per_layer,
                 "us_per_token_all_layers": 1e6 * per_layer * L, "device": DEV,
                 "kv_bytes_per_token": 2 * L * HK * D * torch.finfo(dt_).bits // 8})
        except Exception as e:
            log({"exp": "speed", "arm": "attention-core", "kind": "sdpa_1_query", "T": T,
                 "error": str(e)[:200]})
        del q, k, v

    for shape in ("random", "dup", "const", "code"):
        for n in (4_000, 16_000, 64_000, 256_000, 1_000_000):
            if shape == "const" and n > 64_000:
                continue
            if shape == "code":
                if len(ids) < n:
                    continue
                toks = ids[:n]
            elif shape == "const":
                toks = [7] * n
            elif shape == "dup":
                base = ids[:1024]
                toks = (base * (n // 1024 + 1))[:n]
            else:
                g = torch.Generator().manual_seed(0)
                toks = torch.randint(0, 32000, (n,), generator=g).tolist()
            t0 = time.perf_counter()
            rosa_sources(lib, toks, mode=1)
            dt = time.perf_counter() - t0
            log({"exp": "speed", "arm": "rosa", "kind": "build_and_query", "shape": shape,
                 "n": n, "seconds": dt, "us_per_token": 1e6 * dt / n, "device": "cpu"})
    log({"exp": "done"})


if __name__ == "__main__":
    main()
