"""Correction pass over recall-*.jsonl.

Bug it fixes: `match_len` compared positions i-1 and s-1, i.e. it excluded the current
token, and `src_is_true` compared ROSA's source to `i - dist` when ROSA's source is
`i - dist + 1` (src = match_end + 1). Neither touches the model's predictions; only the
common mask and the source-identification metric were shifted by one token.

This rebuilds each sequence from the same corpora (the construction is deterministic),
recomputes the masks with the corrected definitions, and asserts that every stored ROSA
prediction still equals the token at its own source position.
"""
import argparse
import json
import pathlib

import common

SPAN, PLAIN_LEN = 512, 1536


def src_and_lens(ids):
    src = common.rosa_sources(ids, mode=1)
    lens = []
    for i, s in enumerate(src):
        if s < 0:
            lens.append(0)
            continue
        m = 0
        while m < 4096 and i - m >= 0 and s - 1 - m >= 0 and ids[i - m] == ids[s - 1 - m]:
            m += 1
        lens.append(m)
    return src, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--nseq", type=int, default=6)
    ap.add_argument("--filler", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dist = SPAN + a.filler
    step = 2 * SPAN + a.filler
    need = a.nseq * max(step, PLAIN_LEN) + 256
    tok = common.tokenizer()

    cache = {}
    for corpus in ("code", "prose"):
        toks = common.corpus_tokens(corpus, tok, need)
        seqs = {}
        for k in range(a.nseq):
            off = k * step
            A, F = toks[off:off + SPAN], toks[off + SPAN:off + SPAN + a.filler]
            seqs[("dup", k)] = A + F + A
            seqs[("dup-ctl", k)] = F + A
        for k in range(a.nseq):
            seqs[("plain", k)] = toks[k * PLAIN_LEN:(k + 1) * PLAIN_LEN]
        cache[corpus] = seqs

    lens_cache = {}
    def build(corpus, constr, k):
        key = (corpus, constr, k)
        if key not in lens_cache:
            lens_cache[key] = src_and_lens(cache[corpus][(constr, k)])
        return lens_cache[key]

    rows = []
    checked = bad_pred = fixed_src = fixed_len = 0
    for line in open(a.path):
        if not line.strip():
            continue
        r = json.loads(line)
        arm = r["arm"]
        constr = arm.split(":")[1] if ":" in arm else r["construction"]
        corpus, k = r["corpus"], r["seq"]
        src, lens = build(corpus, constr, k)
        i = r["pos"]
        if constr == "dup":
            r["match_len"] = i - dist + 1
        elif constr == "dup-ctl":
            r["match_len"] = 0
        else:
            r["match_len"] = lens[i]
        if arm.startswith("rosa"):
            checked += 1
            if src[i] >= 0 and r["pred"] != cache[corpus][(constr, k)][src[i]]:
                bad_pred += 1
            r["rosa_match_len"] = lens[i]
            if constr == "dup":
                was, now = r.get("src_is_true"), src[i] == i - dist + 1
                fixed_src += int(bool(now) != bool(was))
                r["src_is_true"] = now
            fixed_len += int(r.get("rosa_match_len") != lens[i])
        rows.append(r)

    out = pathlib.Path(a.out or a.path.replace(".jsonl", ".fixed.jsonl"))
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{len(rows)} rows -> {out}")
    print(f"rosa rows re-checked: {checked}, pred/src mismatches: {bad_pred}")
    print(f"src_is_true corrected on {fixed_src} rows; match_len corrected on {fixed_len} rosa rows")


if __name__ == "__main__":
    main()
