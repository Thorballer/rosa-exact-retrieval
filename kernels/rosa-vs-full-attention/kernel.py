"""ROSA (compressed suffix index) vs full attention, head to head: recall and speed.

Pre-registered questions
  1. On exact long repeats, who gets the next token right -- a suffix-automaton index over
     raw token ids, or Qwen2.5-0.5B's full attention at 1k-17k context?
  2. What does each cost: microseconds per token, and bytes per token of state?

Constructions (same as experiments/rosa-vs-full-attention/recall.py, which ran this on CPU)
  dup      [A][F][A], SPAN=512, FILLER in {512, 2560, 16384} -> copy sits `dist` back
  dup-ctl  [F][A], the earlier copy deleted: identical local context, nothing to retrieve
  plain    a straight 1536-token window, no construction

Arms
  rosa    C suffix automaton (code/c/rosa.c, mode 1) predicting x[src[i]]
  full    Qwen2.5-0.5B, full causal attention (fp16 on T4)
  win512  the same model, attention hard-masked to a 512-token window (only for T <= 4096:
          a 4D mask makes SDPA materialise the whole T x T matrix)

One common mask per construction: match_len = position within the repeated span, computed
from the data, applied identically to every arm. Scores come from hidden states at the
scored positions only -- projecting all T positions through a 152k vocabulary is what made
the first two versions of this kernel OOM.
"""
import glob
import json
import os
import subprocess
import time
import urllib.request

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, PLAIN_LEN = 512, 1536
DISTS = [
    # (filler, dist, nseq, run the model arms): at dist 16896 a T x T attention matrix is
    # 8.5 GiB per sequence on a 14.5 GiB T4, so that cell measures the index only.
    (512, 1024, 16, True),
    (2560, 3072, 16, True),
    (16384, 16896, 8, False),
]
BANDS = [(1, 8), (8, 32), (32, 128), (128, 10 ** 9)]


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    open(OUT, "a").write(line + "\n")


def logrow(d):
    """Per-position rows go to the file only; 200k printed lines would flood the log."""
    open(OUT, "a").write(json.dumps(d) + "\n")


# ------------------------------------------------------------------ ROSA (C)
ROSA_C = r"""
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
    int maxs = 2 * n + 2, maxe = 4 * n + 24;
    uint32_t hcap = 1; while ((int)hcap < 6 * n + 16) hcap <<= 1;
    uint32_t hmask = hcap - 1;
    int32_t *link = calloc(maxs, 4), *len = calloc(maxs, 4), *end = calloc(maxs, 4);
    int32_t *head = malloc(maxs * 4), *esym = malloc(maxe * 4), *eto = malloc(maxe * 4),
            *enext = malloc(maxe * 4);
    int32_t *hkst = malloc(hcap * 4), *hksym = malloc(hcap * 4), *he = malloc(hcap * 4);
    for (int i = 0; i < maxs; i++) { head[i] = -1; link[i] = -1; end[i] = -1; len[i] = 0; }
    for (uint32_t i = 0; i < hcap; i++) he[i] = EMPTY;
    int ne = 0, last = 0, z = 1;
    for (int i = 0; i < n; i++) {
        int32_t t = x[i];
        if (z + 1 >= maxs) return 2;
        int cur = z++; len[cur] = len[last] + 1; end[cur] = mode ? i : -1; head[cur] = -1;
        int p = last;
        while (p != -1) {
            uint32_t h = hs(p, t) & hmask, probe = 0; int e = EMPTY;
            while (he[h] != EMPTY && ++probe <= hcap) {
                if (hkst[h] == p && hksym[h] == t) { e = he[h]; break; }
                h = (h + 1) & hmask;
            }
            if (probe > hcap) return 2;
            if (e != EMPTY) break;
            if (ne + 1 > maxe) return 2;
            esym[ne] = t; eto[ne] = cur; enext[ne] = head[p]; head[p] = ne;
            hkst[h] = p; hksym[h] = t; he[h] = ne; ne++;
            p = link[p];
        }
        if (p == -1) { link[cur] = 0; }
        else {
            uint32_t h = hs(p, t) & hmask, probe = 0; int q = -1;
            while (he[h] != EMPTY && ++probe <= hcap) {
                if (hkst[h] == p && hksym[h] == t) { q = eto[he[h]]; break; }
                h = (h + 1) & hmask;
            }
            if (q == -1) return 2;
            if (len[p] + 1 == len[q]) { link[cur] = q; }
            else {
                if (z + 1 >= maxs) return 2;
                int u = z++;
                len[u] = len[p] + 1; link[u] = link[q]; end[u] = end[q]; head[u] = -1;
                for (int e2 = head[q]; e2 != -1; e2 = enext[e2]) {
                    if (ne + 1 > maxe) return 2;
                    esym[ne] = esym[e2]; eto[ne] = eto[e2]; enext[ne] = head[u]; head[u] = ne;
                    uint32_t hh = hs(u, esym[e2]) & hmask, pr = 0;
                    while (he[hh] != EMPTY && ++pr <= hcap) hh = (hh + 1) & hmask;
                    if (pr > hcap) return 2;
                    hkst[hh] = u; hksym[hh] = esym[e2]; he[hh] = ne; ne++;
                }
                while (p != -1) {
                    uint32_t hh = hs(p, t) & hmask, pr = 0; int hit = EMPTY;
                    while (he[hh] != EMPTY && ++pr <= hcap) {
                        if (hkst[hh] == p && hksym[hh] == t) { hit = he[hh]; break; }
                        hh = (hh + 1) & hmask;
                    }
                    if (hit == EMPTY || eto[hit] != q) break;
                    eto[hit] = u; p = link[p];
                }
                link[q] = u; link[cur] = u;
            }
        }
        if (mode == 0) {
            int v = cur, found = -1;
            while (v != -1) {
                if (found < 0 && len[v] > 0 && end[v] >= 0) found = end[v] + 1;
                end[v] = i; v = link[v];
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
}
"""


def build_rosa():
    open("/kaggle/working/rosa.c", "w").write(ROSA_C)
    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", "/kaggle/working/librosa.so",
                    "/kaggle/working/rosa.c"], check=True)
    import ctypes
    h = ctypes.CDLL("/kaggle/working/librosa.so")
    h.build_src_c.restype = ctypes.c_int
    h.build_src_c.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    return h


def rosa_sources(lib, ids, mode=1):
    import ctypes
    n = len(ids)
    x = (ctypes.c_int32 * n)(*ids)
    src = (ctypes.c_int32 * n)()
    rc = lib.build_src_c(x, n, src, mode)
    if rc != 0:
        raise RuntimeError(f"build_src_c rc={rc}")
    return list(src)


def match_len(ids, i, s, cap=4096):
    """Run ending at i and at s-1: ROSA's src[i] is match_end + 1, so the match includes i."""
    m = 0
    while m < cap and i - m >= 0 and s - 1 - m >= 0 and ids[i - m] == ids[s - 1 - m]:
        m += 1
    return m


def rosa_arm(lib, ids):
    src = rosa_sources(lib, ids, mode=1)
    pred, lens, fired = [], [], []
    for i, s in enumerate(src):
        if s < 0:
            pred.append(-1); lens.append(0); fired.append(False)
        else:
            pred.append(ids[s]); lens.append(match_len(ids, i, s)); fired.append(True)
    return pred, lens, fired, src


# ------------------------------------------------------------------ corpora
BOOKS = [1342, 84, 1661, 345, 2701, 1400]


def prose_text():
    txt = []
    for n in BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
        for _ in range(3):
            try:
                raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
                i, j = raw.find("*** START OF"), raw.find("*** END OF")
                txt.append(raw[i:j] if i >= 0 and j > i else raw)
                break
            except Exception as e:
                print("retry", url, e, flush=True)
                time.sleep(3)
    return "\n\n".join(txt)


def code_text():
    """Python source from the image's own site-packages -- no download, deterministic."""
    import transformers, torch as _t
    roots = [os.path.dirname(transformers.__file__), os.path.dirname(_t.__file__)]
    files = []
    for r in roots:
        files += sorted(glob.glob(os.path.join(r, "**", "*.py"), recursive=True))
    files = files[:250]
    print("code files:", len(files), flush=True)
    return "\n\n".join(open(f, errors="ignore").read() for f in files)


# ------------------------------------------------------------------ model scoring
def window_mask(T, w, dtype):
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    ok = (j <= i) & (i - j < w)
    m = torch.zeros(T, T, dtype=dtype)
    m.masked_fill_(~ok, torch.finfo(dtype).min)
    return m[None, None].to(DEV)


def score_batch(model, seqs, idxs, window=None, chunk=4):
    """Top-1 at `idxs` for each sequence, projecting only the scored hidden states."""
    out = []
    T = len(seqs[0])
    mask = window_mask(T, window, model.dtype) if window is not None else None
    for a in range(0, len(seqs), chunk):
        part = seqs[a:a + chunk]
        x = torch.tensor(part, device=DEV)
        kw = {"attention_mask": mask} if mask is not None else {}
        with torch.inference_mode():
            hs = model.model(x, use_cache=False, **kw).last_hidden_state
            lg = model.lm_head(hs[:, idxs, :]).float()
        for j, seq in enumerate(part):
            row = []
            for k, i in enumerate(idxs):
                v = lg[j, k]
                p = torch.softmax(v, -1)
                tgt = seq[i + 1]
                row.append({"pred": int(v.argmax()), "p_target": float(p[tgt]),
                            "rank_target": int((v > v[tgt]).sum())})
            out.append(row)
        del x, hs, lg
        if DEV == "cuda":
            torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ experiments
def recall(model, tok, lib, corpus, toks, filler, dist, nseq, model_arms=True):
    step = 2 * SPAN + filler
    if len(toks) < nseq * step + 8:
        log({"exp": "skip", "corpus": corpus, "dist": dist, "why": f"only {len(toks)} tokens"})
        return
    dup, ctl = [], []
    for k in range(nseq):
        off = k * step
        A, F = toks[off:off + SPAN], toks[off + SPAN:off + SPAN + filler]
        dup.append(A + F + A)
        ctl.append(F + A)
    idx_dup = list(range(dist, dist + SPAN - 1))
    idx_ctl = list(range(filler, filler + SPAN - 1))

    for j, seq in enumerate(dup):
        pred, lens, fired, src = rosa_arm(lib, seq)
        for i in idx_dup:
            logrow({"exp": "recall", "corpus": corpus, "construction": "dup", "dist": dist,
                    "arm": "rosa", "seq": j, "pos": i, "match_len": i - dist + 1,
                    "target": seq[i + 1], "pred": pred[i], "fired": fired[i],
                    "rosa_match_len": lens[i], "src_is_true": src[i] == i - dist + 1})
        pred, lens, fired, _ = rosa_arm(lib, ctl[j])
        for i in idx_ctl:
            logrow({"exp": "recall", "corpus": corpus, "construction": "dup-ctl", "dist": dist,
                    "arm": "rosa", "seq": j, "pos": i, "match_len": 0,
                    "target": ctl[j][i + 1], "pred": pred[i], "fired": fired[i],
                    "rosa_match_len": lens[i], "src_is_true": None})

    if not model_arms:
        log({"exp": "progress", "corpus": corpus, "dist": dist, "nseq": nseq, "arms": "rosa only"})
        return

    for name, seqs, idxs in (("dup", dup, idx_dup), ("dup-ctl", ctl, idx_ctl)):
        scored = score_batch(model, seqs, idxs)
        for j, seq in enumerate(seqs):
            for i, r in zip(idxs, scored[j]):
                logrow({"exp": "recall", "corpus": corpus, "construction": name, "dist": dist,
                        "arm": "full", "seq": j, "pos": i,
                        "match_len": (i - dist + 1) if name == "dup" else 0,
                        "target": seq[i + 1], **r})
    if len(dup[0]) <= 4096:
        scored = score_batch(model, dup, idx_dup, window=SPAN, chunk=1)
        for j, seq in enumerate(dup):
            for i, r in zip(idx_dup, scored[j]):
                logrow({"exp": "recall", "corpus": corpus, "construction": "dup", "dist": dist,
                        "arm": "win512", "seq": j, "pos": i, "match_len": i - dist + 1,
                        "target": seq[i + 1], **r})
    log({"exp": "progress", "corpus": corpus, "dist": dist, "nseq": nseq})


def plain(model, tok, lib, corpus, toks, nseq=16):
    if len(toks) < nseq * PLAIN_LEN + 8:
        return
    seqs = [toks[k * PLAIN_LEN:(k + 1) * PLAIN_LEN] for k in range(nseq)]
    idx = list(range(512, PLAIN_LEN - 1))
    lens_by_seq = {}
    for j, seq in enumerate(seqs):
        pred, lens, fired, _ = rosa_arm(lib, seq)
        lens_by_seq[j] = lens
        for i in idx:
            logrow({"exp": "recall", "corpus": corpus, "construction": "plain", "dist": 0,
                    "arm": "rosa", "seq": j, "pos": i, "match_len": lens[i],
                    "target": seq[i + 1], "pred": pred[i], "fired": fired[i],
                    "rosa_match_len": lens[i], "src_is_true": None})
    scored = score_batch(model, seqs, idx)
    for j, seq in enumerate(seqs):
        for i, r in zip(idx, scored[j]):
            logrow({"exp": "recall", "corpus": corpus, "construction": "plain", "dist": 0,
                    "arm": "full", "seq": j, "pos": i, "match_len": lens_by_seq[j][i],
                    "target": seq[i + 1], **r})
    log({"exp": "progress", "corpus": corpus, "construction": "plain"})


def speed(model, cfg, tok, lib, ids):
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
        kv = sum(t.numel() * t.element_size() for layer in out.past_key_values for t in layer)
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

    # attention core only: one query against T keys, once per layer
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    HK = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    dt_ = model.dtype
    for T in (1024, 4096, 16384, 32768):
        q = torch.randn(H, 1, D, device=DEV, dtype=dt_)
        k = torch.randn(H, T, D, device=DEV, dtype=dt_)
        v = torch.randn(H, T, D, device=DEV, dtype=dt_)
        reps = 20
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(reps):
                torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=(H != HK))
        if DEV == "cuda":
            torch.cuda.synchronize()
        per_layer = (time.perf_counter() - t0) / reps
        log({"exp": "speed", "arm": "attention-core", "kind": "sdpa_1_query", "T": T,
             "layers": L, "us_per_layer": 1e6 * per_layer,
             "us_per_token_all_layers": 1e6 * per_layer * L, "device": DEV,
             "kv_bytes_per_token": 2 * L * HK * D * torch.finfo(dt_).bits // 8})
        del q, k, v

    # ROSA on the Kaggle CPU, same C code as everywhere else
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


def main():
    log({"exp": "env", "torch": torch.__version__, "device": DEV,
         "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None})
    tok = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(DEV).eval()
    cfg = AutoConfig.from_pretrained(MODEL)
    lib = build_rosa()

    corpora = {}
    for name, text in (("code", code_text()), ("prose", prose_text())):
        ids = tok(text, add_special_tokens=False)["input_ids"][:1_200_000]
        corpora[name] = ids
        log({"exp": "corpus", "corpus": name, "tokens": len(ids)})
        del text

    for name, ids_ in corpora.items():
        plain(model, tok, lib, name, ids_)
        for filler, dist, nseq, model_arms in DISTS:
            recall(model, tok, lib, name, ids_, filler, dist, nseq, model_arms)

    speed(model, cfg, tok, lib, corpora["code"])
    log({"exp": "done"})


if __name__ == "__main__":
    main()
