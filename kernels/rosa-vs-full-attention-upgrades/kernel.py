"""Do any cheap upgrades to the exact index improve recall at 1k-3k, against full attention?

The index is already at 100% on clean copies, so the headroom is in three places: an edit
adjacent to the query (which hides the source from a suffix-anchored index), an ambiguous
continuation (where tie-breaking is a policy), and ordinary text (where firing at all is
usually a mistake). This kernel builds all three constructions and scores, on identical
positions, the C variants and Qwen2.5-0.5B's full attention.

Variants come from experiments/rosa-vs-full-attention/upgrades.c (base, recent, majority,
backoff, vote, approx2/5, gate8, majg8, seed2, seed4). Attention is scored at the same
positions from its hidden states, as in kernels/rosa-vs-full-attention.
"""
import ctypes
import glob
import json
import os
import random
import subprocess

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, P_LEN, BLOCKS = 512, 12, 60
NSEQ = 16
DISTS = [(512, 1024), (2560, 3072)]
VARIANTS = [("base", 0, 0, 0), ("recent", 1, 0, 0), ("majority", 2, 0, 0), ("backoff", 3, 0, 0),
            ("vote", 4, 0, 0), ("approx2", 0, 2, 0), ("approx5", 0, 5, 0), ("gate8", 0, 0, 8),
            ("majg8", 2, 0, 8), ("maj_approx2", 2, 2, 0), ("maj_approx5", 2, 5, 0),
            ("seed2", 5, 2, 0), ("seed4", 5, 4, 0)]


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    open(OUT, "a").write(line + "\n")


def logrow(d):
    open(OUT, "a").write(json.dumps(d) + "\n")


# ---------------------------------------------------------------- C: index variants
UPGRADES_C = r"""
// Candidate engine on top of k-gram hash chains, for testing recall upgrades to the
// exact suffix index. One call per sequence; reports pred/mlen/fired per position.
//
//   variant 0  longest match, earliest occurrence      (what ROSA mode 1 does)
//   variant 1  longest match, most recent occurrence   (what ROSA mode 0 does)
//   variant 2  majority successor among near-longest candidates
//   variant 3  majority with support >= 2, else longest
//   variant 4  weighted vote over all candidates (weight = match length)
//   variant 5  seed-and-extend: anchor on 4-grams ending 0..3 tokens before the query, so an
//              edit sitting immediately before the query does not hide the source; matches
//              tolerate `maxmis` substitutions
//
//   maxmis     mismatches tolerated while extending a match (0 = exact)
//   gate       abstain when the chosen match is shorter than this (0 = never)
//
// Matches shorter than 4 tokens are not represented here: the chains anchor on 2, 4 and 8
// grams, so this engine abstains where the suffix automaton still fires. That is deliberate
// -- the point of this file is the variants, and the SAM remains the reference for short
// matches (see rosa_speed.py and recall.py).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EMPTY (-1)
#define CAP 48          // candidates kept per chain
#define MAXMIS 8
#define MATCHCAP 8192

static inline uint32_t mix(uint32_t h, int32_t v) {
    h ^= (uint32_t)v * 2654435761u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

typedef struct {
    int32_t *head;      // hash slot -> most recent position
    int32_t *prev;      // position -> previous position with the same gram
    uint32_t hcap, hmask;
} Chain;

static int chain_init(Chain *c, int n) {
    c->hcap = 1;
    while (c->hcap < (uint32_t)(4 * n + 16)) c->hcap <<= 1;
    c->hmask = c->hcap - 1;
    c->head = malloc(sizeof(int32_t) * c->hcap);
    c->prev = malloc(sizeof(int32_t) * (size_t)n);
    if (!c->head || !c->prev) return 1;
    for (uint32_t i = 0; i < c->hcap; i++) c->head[i] = EMPTY;
    for (int i = 0; i < n; i++) c->prev[i] = EMPTY;
    return 0;
}

static void chain_free(Chain *c) { free(c->head); free(c->prev); }

// returns the bucket index this position hashed to, inserting it
static inline uint32_t chain_push(Chain *c, const int32_t *x, int i, int k) {
    uint32_t h = 2166136261u;
    for (int t = 0; t < k; t++) h = mix(h, x[i - t]);
    h &= c->hmask;
    c->prev[i] = c->head[h];
    c->head[h] = i;
    return h;
}

// seed at (a,j) with `gap` tokens between the anchor and the query: compare backwards from the
// anchor and forwards across the gap, tolerating maxmis substitutions either way.
static int extend_seed(const int32_t *x, int a, int j, int gap, int maxmis, int *mis_out) {
    int mis = 0, matched = 0;
    for (int t = 1; t <= gap; t++) {
        if (x[a + t] == x[j + t]) matched++;
        else if (mis < maxmis) mis++;
        else { *mis_out = mis; return matched; }
    }
    for (int m = 0; m < MATCHCAP; m++) {
        int p1 = a - m, p2 = j - m;
        if (p1 < 0 || p2 < 0) break;
        if (x[p1] == x[p2]) matched++;
        else if (mis < maxmis) mis++;
        else break;
    }
    *mis_out = mis;
    return matched;
}

static int extend(const int32_t *x, int i, int j, int maxmis, int *mis_out) {
    int m = 0, mis = 0;
    while (m < MATCHCAP && i - m >= 0 && j - m >= 0) {
        if (x[i - m] == x[j - m]) m++;
        else if (mis < maxmis) { mis++; m++; }
        else break;
    }
    *mis_out = mis;
    return m;
}

int predict_variants(const int32_t *x, int n, int variant, int maxmis, int gate,
                     int32_t *pred, int32_t *mlen, int32_t *fired) {
    Chain c2, c4, c8;
    if (chain_init(&c2, n) || chain_init(&c4, n) || chain_init(&c8, n)) return 1;

    int32_t cand[3 * CAP], candm[3 * CAP];
    int ncand;

    for (int i = 0; i < n; i++) {
        chain_push(&c2, x, i, 2);
        if (i >= 3) chain_push(&c4, x, i, 4);
        if (i >= 7) chain_push(&c8, x, i, 8);

        ncand = 0;
        Chain *cs[3] = {&c8, &c4, &c2};
        int ks[3] = {8, 4, 2};
        for (int ci = 0; ci < 3 && ncand < 3 * CAP; ci++) {
            int k = ks[ci];
            if (i - k + 1 < 0) continue;
            uint32_t h = 2166136261u;
            for (int t = 0; t < k; t++) h = mix(h, x[i - t]);
            int j = cs[ci]->head[h & cs[ci]->hmask];
            int kept = 0;
            while (j != EMPTY && kept < CAP) {
                if (j < i) {
                    int mis;
                    int m = extend(x, i, j, maxmis, &mis);
                    cand[ncand] = j;
                    candm[ncand] = m - 3 * mis;      // mismatches cost, they do not count
                    ncand++;
                    kept++;
                }
                j = cs[ci]->prev[j];
            }
        }

        int32_t p = -1, best = -1;
        if (variant == 5) {
            int32_t bestscore = -1, bj = -1, bmatch = 0, bo = 0;
            for (int o = 0; o <= 3; o++) {
                int a = i - o;
                if (a - 3 < 0) continue;
                uint32_t h = 2166136261u;
                for (int t = 0; t < 4; t++) h = mix(h, x[a - t]);
                h &= c4.hmask;
                int j = c4.head[h];
                int kept = 0;
                while (j != EMPTY && kept < CAP) {
                    if (j < a) {
                        int mis;
                        int m = extend_seed(x, a, j, o, maxmis, &mis);
                        int32_t score = m - 3 * mis;
                        if (score > bestscore || (score == bestscore && j < bj)) {
                            bestscore = score; bj = j; bmatch = m; bo = o;
                        }
                        kept++;
                    }
                    j = c4.prev[j];
                }
            }
            if (bj >= 0) { p = x[bj + bo + 1]; best = bmatch; }
        } else if (ncand > 0) {
            if (variant == 4) {                       // weighted vote, weight = match length
                int32_t tot[3 * CAP];
                for (int a = 0; a < ncand; a++) {
                    int32_t succ = x[cand[a] + 1];
                    tot[a] = 0;
                    for (int b = 0; b < ncand; b++)
                        if (x[cand[b] + 1] == succ) tot[a] += candm[b] > 0 ? candm[b] : 0;
                }
                int32_t bestt = -1, bestm2 = -1, bj = -1;
                for (int a = 0; a < ncand; a++) {
                    if (tot[a] > bestt ||
                        (tot[a] == bestt && candm[a] > bestm2) ||
                        (tot[a] == bestt && candm[a] == bestm2 && cand[a] > bj)) {
                        bestt = tot[a]; bestm2 = candm[a]; bj = cand[a];
                    }
                }
                if (bj >= 0) { p = x[bj + 1]; best = bestm2; }
            } else {
                int32_t mmax = -1;
                for (int a = 0; a < ncand; a++) if (candm[a] > mmax) mmax = candm[a];
                if (variant == 0 || variant == 1) {
                    int32_t bj = -1;
                    for (int a = 0; a < ncand; a++) {
                        if (candm[a] != mmax) continue;
                        if (bj < 0 || (variant == 0 ? cand[a] < bj : cand[a] > bj)) bj = cand[a];
                    }
                    if (bj >= 0) { p = x[bj + 1]; best = mmax; }
                } else {                          // majority among near-longest
                    int32_t thresh = mmax - 1;
                    int bestcount = 0, bestm = -1; int32_t bj = -1;
                    for (int a = 0; a < ncand; a++) {
                        if (candm[a] < thresh) continue;
                        int32_t succ = x[cand[a] + 1];
                        int cnt = 0;
                        for (int b = 0; b < ncand; b++)
                            if (candm[b] >= thresh && x[cand[b] + 1] == succ) cnt++;
                        if (cnt > bestcount || (cnt == bestcount && candm[a] > bestm) ||
                            (cnt == bestcount && candm[a] == bestm && cand[a] > bj)) {
                            bestcount = cnt; bestm = candm[a]; bj = cand[a]; p = succ;
                        }
                    }
                    if (variant == 3 && bestcount < 2) {   // no support: fall back to longest
                        int32_t bj2 = -1;
                        for (int a = 0; a < ncand; a++)
                            if (candm[a] == mmax && (bj2 < 0 || cand[a] < bj2)) bj2 = cand[a];
                        if (bj2 >= 0) { p = x[bj2 + 1]; bestm = mmax; }
                    }
                    best = bestm;
                }
            }
        }

        int m_best = best < 0 ? 0 : best;
        if (gate > 0 && m_best < gate) { p = -1; }
        pred[i] = p;
        mlen[i] = m_best;
        fired[i] = (p >= 0) ? 1 : 0;
    }
    chain_free(&c2); chain_free(&c4); chain_free(&c8);
    return 0;
}

"""


def build_upgrades():
    open("/kaggle/working/upgrades.c", "w").write(UPGRADES_C)
    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", "/kaggle/working/libupgrades.so",
                    "/kaggle/working/upgrades.c"], check=True)
    h = ctypes.CDLL("/kaggle/working/libupgrades.so")
    h.predict_variants.restype = ctypes.c_int
    h.predict_variants.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                                   ctypes.POINTER(ctypes.c_int32)]
    return h


def predict(h, ids, variant, maxmis, gate):
    n = len(ids)
    x = (ctypes.c_int32 * n)(*ids)
    pred = (ctypes.c_int32 * n)()
    ml = (ctypes.c_int32 * n)()
    fired = (ctypes.c_int32 * n)()
    rc = h.predict_variants(x, n, variant, maxmis, gate, pred, ml, fired)
    if rc != 0:
        raise RuntimeError(f"predict_variants rc={rc}")
    return list(pred), list(ml), list(fired)


# ---------------------------------------------------------------- corpora
BOOKS = [1342, 84, 1661, 345, 2701, 1400]


def prose_text():
    import urllib.request, time
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
    import transformers, torch as _t
    roots = [os.path.dirname(transformers.__file__), os.path.dirname(_t.__file__)]
    files = []
    for r in roots:
        files += sorted(glob.glob(os.path.join(r, "**", "*.py"), recursive=True))
    files = files[:250]
    return "\n\n".join(open(f, errors="ignore").read() for f in files)


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


# ---------------------------------------------------------------- constructions
def edited(span, at, width, rng, vocab):
    out = list(span)
    for p in at:
        for t in range(p, min(p + width, len(out))):
            new = out[t]
            while new == out[t]:
                new = rng.choice(vocab)
            out[t] = new
    return out


def build(toks, rng, filler, dist, kind):
    """Returns (sequences, scored positions, targets per sequence)."""
    step = 2 * SPAN + filler
    seqs, idxs, targets = [], [], []
    if kind == "ambig":
        for k in range(NSEQ):
            off = k * (4 * (P_LEN + 1) + 8)
            P = toks[off:off + P_LEN]
            xa, xb = toks[off + P_LEN], toks[off + P_LEN + 1]
            seq, idx = [], []
            for b in range(BLOCKS):
                seq += P + [xa] + P + [xa] + P + [xb] + P
                idx.append(len(seq) - 1)
                seq.append(toks[off + P_LEN + 2 + (b % 64)])
            seq.append(xa)
            seqs.append(seq)
            idxs.append(idx)
            targets.append({"majority": {i: xa for i in idx},
                            "recent": {i: xb for i in idx}})
        return seqs, idxs, targets

    vocab = toks[:20000]
    for k in range(NSEQ):
        off = k * step
        A, F = toks[off:off + SPAN], toks[off + SPAN:off + SPAN + filler]
        if kind == "dup":
            seq = A + F + A
            ed = set()
        elif kind == "edit2":
            seq = A + F + edited(A, [SPAN // 3, 2 * SPAN // 3], 2, rng, vocab)
            ed = set()
            for p in (SPAN // 3, 2 * SPAN // 3):
                ed.update(range(dist + p, dist + p + 2))
        elif kind == "edit5":
            seq = A + F + edited(A, [SPAN // 3, 2 * SPAN // 3], 5, rng, vocab)
            ed = set()
            for p in (SPAN // 3, 2 * SPAN // 3):
                ed.update(range(dist + p, dist + p + 5))
        elif kind == "edit_near":
            seq = A + F + edited(A, list(range(1, SPAN, 8)), 1, rng, vocab)
            ed = {dist + p for p in range(1, SPAN, 8)}
        else:
            raise ValueError(kind)
        if kind == "edit_near":
            idx = [i for i in range(dist + 1, dist + SPAN - 1)
                   if (i - 1) in ed and i not in ed and (i + 1) not in ed]
        else:
            idx = [i for i in range(dist, dist + SPAN - 1) if i not in ed and i + 1 not in ed]
        seqs.append(seq)
        idxs.append(idx)
        targets.append({i: seq[i - dist + 1] for i in idx})
    return seqs, idxs, targets


def main():
    log({"exp": "env", "torch": torch.__version__, "device": DEV,
         "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None})
    tok = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(DEV).eval()
    h = build_upgrades()
    rng = random.Random(0)

    corpora = {}
    for name, text in (("code", code_text()), ("prose", prose_text())):
        corpora[name] = tok(text, add_special_tokens=False)["input_ids"][:1_200_000]
        log({"exp": "corpus", "corpus": name, "tokens": len(corpora[name])})
        del text

    for corpus, toks in corpora.items():
        for filler, dist in DISTS:
            for kind in ("dup", "edit2", "edit5", "edit_near", "ambig"):
                if kind == "ambig" and dist != 1024:
                    continue                      # distance is irrelevant to a 12-token context
                seqs, idxs, targets = build(toks, rng, filler, dist, kind)

                # index variants, scored on the same positions
                for name, variant, maxmis, gate in VARIANTS:
                    hit = fired = n = 0
                    hit_maj = hit_rec = 0
                    secs = 0.0
                    for j, seq in enumerate(seqs):
                        import time as _t
                        t0 = _t.perf_counter()
                        pred, ml, f = predict(h, seq, variant, maxmis, gate)
                        secs += _t.perf_counter() - t0
                        idx = idxs[j]
                        tgt = targets[j]
                        if isinstance(tgt, dict) and "majority" in tgt:
                            hit_maj += sum(1 for i in idx if pred[i] == tgt["majority"][i])
                            hit_rec += sum(1 for i in idx if pred[i] == tgt["recent"][i])
                        else:
                            hit += sum(1 for i in idx if pred[i] == tgt[i])
                        fired += sum(1 for i in idx if f[i])
                        n += len(idx)
                    row = {"exp": "upgrade", "corpus": corpus, "construction": kind, "dist": dist,
                           "variant": name, "n": n, "fired": fired, "seconds": secs}
                    if kind == "ambig":
                        row["hit_majority"] = hit_maj
                        row["hit_recent"] = hit_rec
                    else:
                        row["hit"] = hit
                    log(row)

                # full attention at the same positions
                idx = idxs[0]
                same = all(idxs[j] == idx for j in range(len(seqs)))
                if not same:
                    raise SystemExit("scored positions differ across sequences")
                scored = score_batch(model, seqs, idx, chunk=2 if len(seqs[0]) > 2048 else 4)
                for j, seq in enumerate(seqs):
                    tgt = targets[j]
                    for i, r in zip(idx, scored[j]):
                        row = {"exp": "attn", "corpus": corpus, "construction": kind,
                               "dist": dist, "seq": j, "pos": i, "pred": r["pred"],
                               "p_target": r["p_target"], "rank_target": r["rank_target"]}
                        if isinstance(tgt, dict) and "majority" in tgt:
                            row["target_majority"] = tgt["majority"][i]
                            row["target_recent"] = tgt["recent"][i]
                        else:
                            row["target"] = tgt[i]
                        logrow(row)
                log({"exp": "progress", "corpus": corpus, "dist": dist, "kind": kind})
                del seqs, idxs, targets, scored
                if DEV == "cuda":
                    torch.cuda.empty_cache()
    log({"exp": "done"})


if __name__ == "__main__":
    main()
