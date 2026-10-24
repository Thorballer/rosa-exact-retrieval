"""Turn results/recall-*.jsonl into the tables used in the write-up."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BANDS = [("1-8", 1, 8), ("8-32", 8, 32), ("32-128", 32, 128), ("128+", 128, 10 ** 9), ("none", 0, 1)]


def band(m):
    for name, lo, hi in BANDS:
        if lo <= m < hi:
            return name
    return "none"


def load(tag="main"):
    rows = []
    for p in sorted((HERE / "results").glob(f"recall-{tag}*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def agg(rows, key):
    out = {}
    for r in rows:
        b = key(r)
        d = out.setdefault(b, {"n": 0, "hit": 0, "fired": 0, "p": [], "rank": [], "src_ok": 0, "src_n": 0})
        d["n"] += 1
        d["hit"] += int(r["pred"] == r["target"])
        d["fired"] += int(bool(r.get("fired", True)))
        if "p_target" in r:
            d["p"].append(r["p_target"]); d["rank"].append(r["rank_target"])
        if r.get("src_is_true") is not None:
            d["src_n"] += 1; d["src_ok"] += int(r["src_is_true"])
    res = {}
    for b, d in out.items():
        res[b] = {
            "n": d["n"],
            "acc": d["hit"] / d["n"] if d["n"] else 0.0,
            "fired_frac": d["fired"] / d["n"] if d["n"] else 0.0,
            "p_target": (sum(d["p"]) / len(d["p"])) if d["p"] else None,
            "mean_rank": (sum(d["rank"]) / len(d["rank"])) if d["rank"] else None,
            "src_acc": (d["src_ok"] / d["src_n"]) if d["src_n"] else None,
        }
    return res


def table(rows, corpus, arm, by="band"):
    sub = [r for r in rows if r["corpus"] == corpus and r["arm"] == arm]
    a = agg(sub, lambda r: band(r["match_len"]) if by == "band" else "all")
    return a


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    rows = load(tag)
    print(f"{len(rows)} rows\n")
    for corpus in ("code", "prose"):
        for constr in ("dup", "dup-ctl", "plain"):
            arms = [a for a in ("rosa", "full", "win512") if any(
                r["arm"] == f"{a}:{constr}" and r["corpus"] == corpus for r in rows)]
            if not arms:
                continue
            print(f"== {corpus} / {constr}")
            hdr = f"{'band':>8} {'n':>6} " + " ".join(f"{a:>16}" for a in arms)
            print(hdr)
            for bname, lo, hi in BANDS:
                cells, n = [], None
                for a in arms:
                    t = table(rows, corpus, f"{a}:{constr}")
                    d = t.get(bname)
                    if d:
                        n = d["n"] if n is None else n
                        extra = f" (f{d['fired_frac']:.2f})" if a == "rosa" else (
                            f" r{d['mean_rank']:.0f}" if d["mean_rank"] is not None else "")
                        cells.append(f"{d['acc']*100:6.1f}%{extra:>9}")
                    else:
                        cells.append(f"{'-':>16}")
                if any(c.strip() != "-" for c in cells):
                    print(f"{bname:>8} {n:>6} " + " ".join(cells))
            # overall
            cells = []
            for a in arms:
                t = table(rows, corpus, f"{a}:{constr}", by="all")
                d = t.get("all")
                cells.append(f"{d['acc']*100:6.1f}% (n={d['n']})" if d else "-")
            print(f"{'TOTAL':>8} {'':>6} " + " ".join(f"{c:>16}" for c in cells))
            print()


if __name__ == "__main__":
    main()
