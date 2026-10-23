"""Does C fix ROSA's speed?  Pure-Python SAM vs C SAM, same algorithm."""
import ctypes, os, random, re, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "librosa_speed.dylib")
subprocess.run(["clang", "-O3", "-march=native", "-shared", "-fPIC",
                "-o", SO, os.path.join(HERE, "rosa.c")], check=True)
lib = ctypes.CDLL(SO)
lib.build_src_c.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int32)]
lib.build_src_c.restype = ctypes.c_int


def build_src_c(x):
    n = len(x)
    arr = (ctypes.c_int32 * n)(*x); out = (ctypes.c_int32 * n)()
    assert lib.build_src_c(arr, n, out) == 0
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


def data_shapes(n):
    rnd = random.Random(0)
    blk = list(range(28))
    words = re.findall(r"\S+", open(os.path.expanduser("~/.hermes/CONTRIBUTING.md"),
                                    encoding="utf-8", errors="ignore").read())
    vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    ids = [vocab[w] for w in words]
    return {
        "random(1k sym)": [rnd.randrange(1000) for _ in range(n)],
        "constant run": [0] * n,
        "repeated 28-blk": (blk * (n // 28 + 1))[:n],
        "real prose(word)": (ids * (n // len(ids) + 1))[:n],
    }


if __name__ == "__main__":
    rnd = random.Random(1); bad = 0
    for _ in range(200):
        k = rnd.choice([2, 3, 5, 50])
        x = [rnd.randrange(k) for _ in range(rnd.randint(1, 40))]
        if build_src_c(x) != build_src_py(x):
            bad += 1
    print(f"correctness: {200 - bad}/200 identical to the Python reference", flush=True)

    sh = data_shapes(32768)
    print(f"\n{'shape':18s} {'n':>7} {'py(s)':>8} {'C(s)':>8} {'speedup':>8} {'C µs/tok':>9}", flush=True)
    for label, full in sh.items():
        for n in (2048, 4096):
            x = full[:n]
            t0 = time.perf_counter(); py = build_src_py(x); tp = time.perf_counter() - t0
            t0 = time.perf_counter(); cc = build_src_c(x); tc = time.perf_counter() - t0
            assert py == cc
            print(f"{label:18s} {n:7d} {tp:8.3f} {tc:8.4f} {tp / tc:7.1f}x {tc / n * 1e6:9.2f}", flush=True)
    print("\nC only, scaling:", flush=True)
    for label, full in sh.items():
        for n in (16384, 32768):
            x = full[:n]
            t0 = time.perf_counter(); build_src_c(x); tc = time.perf_counter() - t0
            print(f"  {label:18s} n={n:6d}: {tc:7.3f}s  ({tc / n * 1e6:6.2f} µs/token)", flush=True)
