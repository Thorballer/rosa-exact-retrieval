"""Speed + memory of the ROSA compressed suffix index, measured in C (no Python in the loop).

Each cell runs in its own process so getrusage's high-water mark belongs to that cell only;
bytes/token is the cell's peak RSS minus the same binary's peak RSS on a 16-token input.

    python rosa_speed.py                # full sweep -> results/rosa-speed.jsonl
"""
import json
import struct
import subprocess

import common

BENCH = common.BUILD / "rosa_bench"
CELLS = [("random", "random"), ("dup", "dup"), ("const", "const"), ("code", "codefile")]
NS = [4_000, 16_000, 64_000, 256_000]
OUT = common.HERE / "results" / "rosa-speed.jsonl"


def build():
    if not BENCH.exists():
        subprocess.run(["clang", "-O3", "-march=native", "-o", str(BENCH), "rosa_bench.c",
                        "../../code/c/rosa.c"], check=True, cwd=common.HERE)


def run(shape, n, mode, stdin=None):
    p = subprocess.run([str(BENCH), shape, str(n), str(mode)], input=stdin or b"",
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{shape} n={n}: {p.stderr.decode()[-300:]}")
    return json.loads(p.stdout.decode().strip().splitlines()[-1])


def main():
    build()
    base = {m: run("const", 16, m)["maxrss_bytes"] for m in (0, 1)}
    rows = []
    for mode in (1, 0):
        for shape, drv in CELLS:
            if mode == 0 and shape in ("const", "dup") and True:
                pass  # included: mode-0's quadratic is the point of the table
            for n in NS:
                if mode == 0 and shape in ("const", "dup") and n > 16_000:
                    continue  # mode 0 is O(n^2) there; 256k would take hours
                stdin = b""
                if shape == "code":
                    toks = common.corpus_tokens("code", common.tokenizer(), n, strict=False)
                    if len(toks) < n:
                        continue  # corpus runs out before this length
                    stdin = struct.pack(f"<{n}i", *toks[:n])
                r = run(drv, n, mode, stdin)
                r["shape_label"] = shape
                r["bytes_per_token_rss"] = (r["maxrss_bytes"] - base[mode]) / n
                rows.append(r)
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                  for k, v in r.items()}), flush=True)
    common.emit(OUT, rows)


if __name__ == "__main__":
    main()
