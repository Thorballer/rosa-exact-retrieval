"""Shared harness: corpora, tokenizer, ROSA via ctypes, bucketing.

Nothing here trains anything. Both arms are evaluated on the same token ids at the
same positions, bucketed by one common mask (methodology.md #6).
"""
import ctypes
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)

# ---------------------------------------------------------------- ROSA (C, ctypes)

def rosa_lib():
    src = REPO / "code" / "c" / "rosa.c"
    lib = BUILD / "librosa.dylib"
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            ["clang", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(lib), str(src)],
            check=True,
        )
    h = ctypes.CDLL(str(lib))
    h.build_src_c.restype = ctypes.c_int
    h.build_src_c.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    return h


def rosa_sources(tokens, mode=1):
    """src[i] = position whose successor ROSA predicts for i+1, or -1."""
    h = rosa_lib()
    n = len(tokens)
    x = (ctypes.c_int32 * n)(*tokens)
    src = (ctypes.c_int32 * n)()
    rc = h.build_src_c(x, n, src, mode)
    if rc != 0:
        raise RuntimeError(f"build_src_c rc={rc}")
    return list(src)


def match_len(tokens, i, s, cap=4096):
    """Length of the run ending at position i and at ROSA's source-minus-one (s-1).

    ROSA's src[i] is match_end + 1, so the run to compare is tokens[i-m] vs tokens[s-1-m]:
    it includes the current token, which an off-by-one here silently drops (see repost.py).
    """
    m = 0
    while m < cap and i - m >= 0 and s - 1 - m >= 0 and tokens[i - m] == tokens[s - 1 - m]:
        m += 1
    return m


# ---------------------------------------------------------------- corpora

def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


def corpus_tokens(kind, tok, n_tokens, strict=True):
    """Code = this repo's own kernels; prose = Project Gutenberg. Token ids in Qwen space."""
    if kind == "code":
        text = "\n\n".join(
            p.read_text(errors="ignore") for p in sorted((REPO / "kernels").glob("*/kernel.py"))
        )
    elif kind == "prose":
        text = (REPO / "code" / "c" / "corpus.txt").read_text(errors="ignore")
        # corpus.txt is word-level Moby Dick; the tokenizer handles raw text fine
    else:
        raise ValueError(kind)
    ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    if len(ids) < n_tokens:
        if strict:
            raise RuntimeError(f"{kind}: only {len(ids)} tokens")
        return ids
    return ids[:n_tokens]


def dup_sequence(tokens, span, filler_len, offset=0):
    """[A][F][A]  -- copy 2 sits distance span+filler_len > any window < distance."""
    A = tokens[offset:offset + span]
    F = tokens[offset + span:offset + span + filler_len]
    return A + F + A, span, len(A) + len(F)


def control_sequence(tokens, span, filler_len, offset=0):
    """[F][A] -- the same scored positions with the earlier copy deleted."""
    A = tokens[offset:offset + span]
    F = tokens[offset + span:offset + span + filler_len]
    return F + A, span, len(F)


# ---------------------------------------------------------------- metrics

BANDS = [(1, 8), (8, 32), (32, 128), (128, 10 ** 9)]


def band(m):
    for lo, hi in BANDS:
        if lo <= m < hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}"
    return "<1"


def bucket_by_match(preds, truth, lens):
    out = {}
    for p, t, m in zip(preds, truth, lens):
        b = band(m)
        d = out.setdefault(b, [0, 0])
        d[1] += 1
        d[0] += int(p == t)
    return {b: {"acc": hit / n, "n": n} for b, (hit, n) in out.items()}


def emit(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def log(*a):
    print(*a, flush=True)
