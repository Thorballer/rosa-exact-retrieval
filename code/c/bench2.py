"""Is ROSA slow because of Python, or because of the algorithm?"""
import ctypes, os, random, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "librosa_speed.dylib")
subprocess.run(["clang", "-O3", "-march=native", "-shared", "-fPIC",
                "-o", SO, os.path.join(HERE, "rosa.c")], check=True)
lib = ctypes.CDLL(SO)
lib.build_src_c.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
lib.build_src_c.restype = ctypes.c_int


def build_src_c(x, mode=0):
    n = len(x)
    arr = (ctypes.c_int32 * n)(*x); out = (ctypes.c_int32 * n)()
    assert lib.build_src_c(arr, n, out, mode) == 0
    return list(out)


def build_src_py(x):
    n = len(x); s = 2 * n + 1
    b = [None] * s; c = [-1] * s; d = [0] * s; e = [-1] * s
    b[0] = {}; g = 0; z = 1
    src = [-1] * n
    for i, t in enumerate(x):
        r = z; z += 1; b[r] = {}; d[r] = d[g] + 1; p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r; p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z; z += 1; b[u] = b[q].copy(); d[u] = d[p] + 1; c[u] = c[q]; e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u; p = c[p]
                c[q] = c[r] = u
        v = g = r; a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = e[v] + 1; break
            v = c[v]
        src[i] = a
        v = g
        while v != -1 and e[v] < i:
            e[v] = i; v = c[v]
    return src


CORPUS = os.environ.get("ROSA_CORPUS") or (sys.argv[1] if len(sys.argv) > 1 else None)
if CORPUS is None:
    for cand in ("corpus.txt", "~/.hermes/CONTRIBUTING.md"):
        if os.path.exists(os.path.expanduser(cand)):
            CORPUS = os.path.expanduser(cand)
            break
if CORPUS is None:
    raise SystemExit("no corpus: pass a UTF-8 text file as argv[1] or set ROSA_CORPUS")


def corpus():
    """Word-level token stream. Any English prose file works; the recorded logs used a
    ~7.6k-word markdown document, so absolute timings depend on the input you pass."""
    words = re.findall(r"\S+", open(CORPUS, encoding="utf-8", errors="ignore").read())
    vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    return [vocab[w] for w in words]


def shapes(n):
    rnd = random.Random(0)
    ids = corpus()
    return [("random(1k sym)", [rnd.randrange(1000) for _ in range(n)]),
            ("constant run", [0] * n),
            ("repeated 28-blk", (list(range(28)) * (n // 28 + 1))[:n]),
            ("real prose(word)", (ids * (n // len(ids) + 1))[:n])]


if __name__ == "__main__":
    rnd = random.Random(1); bad = 0
    for _ in range(200):
        k = rnd.choice([2, 3, 5, 50])
        x = [rnd.randrange(k) for _ in range(rnd.randint(1, 40))]
        if build_src_c(x, 0) != build_src_py(x):
            bad += 1
    print(f"correctness (recency mode): {200 - bad}/200 identical to Python reference", flush=True)

    print(f"\n{'shape':16s} {'n':>6} {'py us/tok':>10} {'C rec':>7} {'C first':>8} {'x vs py':>8}",
          flush=True)
    for label, data in shapes(32768):
        for n in (4096, 32768):
            x = data[:n]
            t0 = time.perf_counter(); build_src_py(x); tp = time.perf_counter() - t0
            t0 = time.perf_counter(); build_src_c(x, 0); tr = time.perf_counter() - t0
            t0 = time.perf_counter(); build_src_c(x, 1); tf = time.perf_counter() - t0
            print(f"{label:16s} {n:6d} {tp / n * 1e6:10.2f} {tr / n * 1e6:7.2f} "
                  f"{tf / n * 1e6:8.2f} {tp / max(tr, 1e-9):7.1f}x", flush=True)

    ids = corpus() * 4
    n = len(ids)
    rec = build_src_c(ids, 0); fir = build_src_c(ids, 1)
    hit_r = sum(1 for i, s in enumerate(rec) if 0 <= s < n and ids[s] == ids[i]) / n
    hit_f = sum(1 for i, s in enumerate(fir) if 0 <= s < n and ids[s] == ids[i]) / n
    disagree = sum(1 for a, b in zip(rec, fir) if a != b) / n
    print(f"\nreal prose, {n} word tokens:")
    print(f"  next-word accuracy  recency {hit_r:.3f} | first-occurrence {hit_f:.3f}")
    print(f"  different continuation chosen: {disagree:.1%} of positions")
