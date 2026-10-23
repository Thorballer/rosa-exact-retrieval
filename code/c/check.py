"""Correctness check for rosa.c against a brute-force implementation of the ROSA spec.

Mode 0 (recency) must reproduce exactly:
    src[i] = j+1, where (m, j) maximises m subject to j < i and x[j-m:j] == x[i-m:i],
             ties on m broken by the largest j; -1 if no previous occurrence exists.
Mode 1 (first-occurrence) is a different tie-break by design (it trades exact recency for
constant time on repetitive input), so it is only checked for "does not regress prediction
accuracy much", never for equality.

    python3 check.py            # ~10 s
"""
import ctypes
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "librosa_check.dylib")
subprocess.run(["clang", "-O2", "-shared", "-fPIC", "-o", SO, os.path.join(HERE, "rosa.c")],
               check=True)
lib = ctypes.CDLL(SO)
lib.build_src_c.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
lib.build_src_c.restype = ctypes.c_int


def build(x, mode=0):
    n = len(x)
    a = (ctypes.c_int32 * n)(*x)
    o = (ctypes.c_int32 * n)()
    rc = lib.build_src_c(a, n, o, mode)
    return rc, list(o)


def brute(x):
    n = len(x)
    src = [-1] * n
    for i in range(n):
        best_m, best_j = 0, -1
        for j in range(i):
            m = 0
            while m < i + 1 and j - m >= 0 and x[j - m] == x[i - m]:
                m += 1
            if m > best_m or (m == best_m and m > 0 and j > best_j):
                best_m, best_j = m, j
        src[i] = best_j + 1 if best_m > 0 else -1
    return src


def cases():
    rnd = random.Random(0)
    for alpha in (1, 2, 3, 5, 50, 1000):
        for n in (1, 2, 3, 8, 17, 64, 129, 300):
            yield f"random a={alpha} n={n}", [rnd.randrange(alpha) for _ in range(n)]
    for n in (64, 300):
        yield f"constant n={n}", [7] * n
        yield f"period-3 n={n}", ([0, 1, 2] * n)[:n]
        yield f"period-28 n={n}", (list(range(28)) * n)[:n]
        yield f"countdown n={n}", list(range(n))[::-1]
        yield f"runs n={n}", [i // 5 for i in range(n)]


def main():
    bad = 0
    total = 0
    for label, x in cases():
        total += 1
        rc, got = build(x, 0)
        if rc != 0:
            print(f"FAIL(rc={rc}) {label}")
            bad += 1
            continue
        want = brute(x)
        if got != want:
            wrong = [i for i in range(len(x)) if got[i] != want[i]]
            print(f"MISMATCH {label}: {len(wrong)}/{len(x)} positions, first {wrong[0]} "
                  f"(got {got[wrong[0]]}, want {want[wrong[0]]})")
            bad += 1
    print(f"mode 0: {total - bad}/{total} cases identical to brute force")

    # mode 1: same automaton, cheaper tie-break.  Report how often it picks a different
    # (equally long) earlier occurrence, and whether it loses prediction accuracy.
    rnd = random.Random(1)
    agree = differ = hit0 = hit1 = tot = 0
    for _ in range(40):
        n = 400
        x = [rnd.randrange(8) for _ in range(n)]
        _, s0 = build(x, 0)
        _, s1 = build(x, 1)
        for i in range(n - 1):
            if s0[i] >= 0 and s1[i] >= 0:
                tot += 1
                if s0[i] == s1[i]:
                    agree += 1
                else:
                    differ += 1
            if s0[i] >= 0 and x[s0[i]] == x[i + 1]:
                hit0 += 1
            if s1[i] >= 0 and x[s1[i]] == x[i + 1]:
                hit1 += 1
    print(f"mode 1: same continuation as mode 0 in {agree / max(agree + differ, 1):.1%} of "
          f"positions; next-token correct {hit0 / tot:.3f} (mode 0) vs {hit1 / tot:.3f} (mode 1)")


if __name__ == "__main__":
    sys.exit(main())
