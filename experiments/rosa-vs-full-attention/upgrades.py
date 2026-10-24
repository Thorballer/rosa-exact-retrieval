"""Can the index's recall be improved at 1k-3k? Five families of cheap upgrade, measured.

Where the headroom is: on clean copies the index is already at 100.0% at 1,024 and 3,072, so
nothing can improve it there. What is left is (a) copies with small edits, where exact
matching breaks, (b) ambiguous continuations, where tie-breaking is a policy choice, and
(c) deciding not to fire on short matches.

Variants (all in upgrades.c, no training, no floats):

  base      longest match, earliest occurrence        = ROSA mode 1
  recent    longest match, most recent occurrence     = ROSA mode 0
  majority  most frequent successor among near-longest candidates
  backoff   majority with support >= 2, else base
  vote      weighted vote over candidates, weight = match length
  approx2   base but matches tolerate 2 mismatches (each costs 3 tokens of match)
  approx5   same with 5 mismatches
  gate8     base, abstain below an 8-token match
  majg8     majority, abstain below an 8-token match
  maj_approx2 / maj_approx5   majority on tolerant matches

Constructions (SPAN=512, distances 1,024 and 3,072, 16 sequences per corpus):
  dup        clean [A][F][A]
  edit2      [A][F][A'] with two 2-token substitutions in A'
  edit5      [A][F][A'] with two 5-token substitutions in A'
  edit_near  a single-token edit immediately before the query, every 8 positions -- exact
             matching then has a 1-2 token match, approximate matching can bridge it
  ambig      [P a][P a][P b][P ?] blocks: at the fourth P three prior occurrences exist
             with successors a, a, b -- majority a, most recent b. The query is labelled
             twice, once against each truth, so a policy's preference is directly readable
  plain      a straight 1536-token window (precision and coverage)
"""
import argparse
import ctypes
import json
import random
import statistics
import subprocess
import time

import common

SPAN, PLAIN_LEN = 512, 1536
BUILD = common.BUILD

VARIANTS = [
    ("base", 0, 0, 0),
    ("recent", 1, 0, 0),
    ("majority", 2, 0, 0),
    ("backoff", 3, 0, 0),
    ("vote", 4, 0, 0),
    ("approx2", 0, 2, 0),
    ("approx5", 0, 5, 0),
    ("gate8", 0, 0, 8),
    ("majg8", 2, 0, 8),
    ("maj_approx2", 2, 2, 0),
    ("maj_approx5", 2, 5, 0),
    ("seed2", 5, 2, 0),
    ("seed4", 5, 4, 0),
]


def lib():
    src = common.HERE / "upgrades.c"
    so = BUILD / "libupgrades.dylib"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["clang", "-O3", "-march=native", "-shared", "-fPIC",
                        "-o", str(so), str(src)], check=True)
    h = ctypes.CDLL(str(so))
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


def edited(span, at, width, rng, vocab):
    out = list(span)
    for p in at:
        for t in range(p, min(p + width, len(out))):
            new = out[t]
            while new == out[t]:
                new = rng.choice(vocab)
            out[t] = new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseq", type=int, default=16)
    ap.add_argument("--corpora", default="code,prose")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()

    tok = common.tokenizer()
    h = lib()
    rng = random.Random(0)
    rows = []

    for corpus in a.corpora.split(","):
        need = a.nseq * (2 * SPAN + 2560) + 8192
        toks = common.corpus_tokens(corpus, tok, need, strict=False)
        vocab = toks[:20000]                       # source of replacement ids

        for filler, dist in ((512, 1024), (2560, 3072)):
            step = 2 * SPAN + filler
            nseq = min(a.nseq, max(1, (len(toks) - 8) // step))
            built = {"dup": [], "edit2": [], "edit5": [], "edit_near": []}
            for k in range(nseq):
                off = k * step
                A, F = toks[off:off + SPAN], toks[off + SPAN:off + SPAN + filler]
                built["dup"].append(A + F + A)
                built["edit2"].append(A + F + edited(A, [SPAN // 3, 2 * SPAN // 3], 2, rng, vocab))
                built["edit5"].append(A + F + edited(A, [SPAN // 3, 2 * SPAN // 3], 5, rng, vocab))
                near = edited(A, list(range(1, SPAN, 8)), 1, rng, vocab)
                built["edit_near"].append(A + F + near)

            for constr, width in (("dup", 0), ("edit2", 2), ("edit5", 5), ("edit_near", 1)):
                for j, seq in enumerate(built[constr]):
                    ed = set()
                    if constr == "edit_near":
                        # the edit sits at i-1 for every 8th position: score exactly those
                        ed = {dist + p for p in range(1, SPAN, 8)}
                        idx = [i for i in range(dist + 1, dist + SPAN - 1)
                               if (i - 1) in ed and i not in ed and (i + 1) not in ed]
                    else:
                        for p in (SPAN // 3, 2 * SPAN // 3):
                            ed.update(range(dist + p, dist + p + width))
                        idx = [i for i in range(dist, dist + SPAN - 1)
                               if i not in ed and i + 1 not in ed]
                    target = {i: seq[i - dist + 1] for i in idx}
                    for name, variant, maxmis, gate in VARIANTS:
                        t0 = time.perf_counter()
                        pred, ml, fired = predict(h, seq, variant, maxmis, gate)
                        dt = time.perf_counter() - t0
                        rows.append({"exp": "upgrade", "corpus": corpus, "construction": constr,
                                     "dist": dist, "variant": name, "seq": j,
                                     "n_scored": len(idx), "seconds": dt, "tokens": len(seq),
                                     "nseq": nseq,
                                     "hit": sum(1 for i in idx if pred[i] == target[i]),
                                     "fired": sum(1 for i in idx if fired[i]),
                                     "mean_mlen": statistics.mean(ml[i] for i in idx)})
                print(f"{corpus} d={dist} {constr}: {len(built[constr])} seqs", flush=True)

        # ---- ambiguous continuation: same text, two labelled truths -----------
        P_LEN, BLOCKS = 12, 60
        for filler, dist in ((512, 1024), (2560, 3072)):
            results = []
            nseq_amb = min(a.nseq, max(1, len(toks) // (BLOCKS * (4 * P_LEN + 3) + 80)))
            for k in range(nseq_amb):
                off = k * (4 * (P_LEN + 1) + 8)
                P = toks[off:off + P_LEN]
                xa, xb = toks[off + P_LEN], toks[off + P_LEN + 1]
                # a unique separator after each block stops a match from extending across
                # blocks, so every query sees exactly the three priors inside its own block
                seq, idx = [], []
                for b in range(BLOCKS):
                    block = P + [xa] + P + [xa] + P + [xb] + P
                    seq += block
                    idx.append(len(seq) - 1)
                    seq.append(toks[off + P_LEN + 2 + (b % 64)])
                seq.append(xa)
                for name, variant, maxmis, gate in VARIANTS:
                    t0 = time.perf_counter()
                    pred, ml, fired = predict(h, seq, variant, maxmis, gate)
                    dt = time.perf_counter() - t0
                    results.append({"exp": "upgrade", "corpus": corpus, "construction": "ambig",
                                    "dist": dist, "variant": name, "seq": k,
                                    "n_scored": len(idx), "seconds": dt, "tokens": len(seq), "nseq": nseq_amb,
                                    "hit_majority": sum(1 for i in idx if pred[i] == xa),
                                    "hit_recent": sum(1 for i in idx if pred[i] == xb),
                                    "fired": sum(1 for i in idx if fired[i]),
                                    "mean_mlen": statistics.mean(ml[i] for i in idx)})
            rows += results
            print(f"{corpus} d={dist} ambig done", flush=True)

        # ---- plain window -----------------------------------------------------
        nseq_plain = min(a.nseq, len(toks) // PLAIN_LEN)
        plain = [toks[k * PLAIN_LEN:(k + 1) * PLAIN_LEN] for k in range(nseq_plain)]
        for j, seq in enumerate(plain):
            idx = list(range(512, PLAIN_LEN - 1))
            for name, variant, maxmis, gate in VARIANTS:
                pred, ml, fired = predict(h, seq, variant, maxmis, gate)
                rows.append({"exp": "upgrade", "corpus": corpus, "construction": "plain",
                             "dist": 0, "variant": name, "seq": j, "n_scored": len(idx),
                             "seconds": 0.0, "tokens": len(seq),
                             "hit": sum(1 for i in idx if pred[i] == seq[i + 1]),
                             "fired": sum(1 for i in idx if fired[i]),
                             "mean_mlen": statistics.mean(ml[i] for i in idx)})
        print(f"{corpus}: {len(rows)} rows so far", flush=True)

    common.emit(common.HERE / "results" / f"upgrades-{a.tag}.jsonl", rows)


if __name__ == "__main__":
    main()
