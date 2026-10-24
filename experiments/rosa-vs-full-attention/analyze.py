"""Markdown tables from either results format (local recall-*.jsonl or Kaggle results.jsonl).

    python analyze.py results/recall-main.jsonl ...
    python analyze.py /tmp/kout/results.jsonl --dist 1024

Bootstrap CIs are over sequences, not positions: positions inside one sequence are not
independent draws (they are, quite literally, copies of the same text).
"""
import argparse
import json
import pathlib
import random
from collections import defaultdict

BANDS = [("1-8", 1, 8), ("8-32", 8, 32), ("32-128", 32, 128), ("128+", 128, 10 ** 9)]
ARMS = ["rosa", "full", "win512"]


def norm(r):
    """One row -> (corpus, construction, dist, arm, seq, band, correct)."""
    arm = r.get("arm")
    if ":" in arm:
        arm, construction = arm.split(":", 1)
    else:
        construction = r.get("construction")
    dist = r.get("dist")
    if dist is None:
        dist = 1024 if construction in ("dup", "dup-ctl") else 0
    m = r.get("match_len", 0)
    b = "none"
    for name, lo, hi in BANDS:
        if lo <= m < hi:
            b = name
            break
    return (r.get("corpus"), construction, dist, arm, r.get("seq"),
            b, int(r.get("pred") == r.get("target")))


def boot(vals, n=2000, seed=0):
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def seq_means(rows):
    """accuracy per sequence (macro), so the CI reflects between-sequence variance."""
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r[-3]].append(r[-1])
    return [sum(v) / len(v) for v in by_seq.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--band", default="all")
    a = ap.parse_args()
    rows = []
    for f in a.files:
        p = pathlib.Path(f)
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("exp") not in (None, "recall"):
                continue
            rows.append(norm(o))
    print(f"{len(rows)} scored positions from {', '.join(a.files)}\n")

    keys = sorted({(r[0], r[1], r[2]) for r in rows})
    for corpus, construction, dist in keys:
        print(f"### {corpus} / {construction}" + (f" / distance {dist}" if dist else ""))
        print("| band | n | " + " | ".join(ARMS) + " |")
        print("|---|---|" + "---|" * len(ARMS))
        for bname, _, _ in BANDS + [("none", 0, 1)]:
            cells, n = [], 0
            for arm in ARMS:
                sub = [r for r in rows if r[0] == corpus and r[1] == construction
                       and r[2] == dist and r[3] == arm and r[5] == bname]
                if not sub:
                    cells.append("-")
                    continue
                n = len(sub)
                acc = sum(r[-1] for r in sub) / len(sub)
                lo, hi = boot(seq_means(sub))
                cells.append(f"{acc*100:.1f}% [{lo*100:.0f}-{hi*100:.0f}]")
            if any(c != "-" for c in cells):
                print(f"| {bname} | {n} | " + " | ".join(cells) + " |")
        cells = []
        for arm in ARMS:
            sub = [r for r in rows if r[0] == corpus and r[1] == construction
                   and r[2] == dist and r[3] == arm]
            if not sub:
                cells.append("-")
                continue
            acc = sum(r[-1] for r in sub) / len(sub)
            lo, hi = boot(seq_means(sub))
            cells.append(f"**{acc*100:.1f}%** [{lo*100:.0f}-{hi*100:.0f}] (n={len(sub)})")
        print("| TOTAL | | " + " | ".join(cells) + " |\n")


if __name__ == "__main__":
    main()
