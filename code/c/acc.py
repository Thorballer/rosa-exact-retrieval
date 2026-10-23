"""Correctly-aligned next-token accuracy: recency vs first-occurrence."""
import re, os
from bench2 import build_src_c, corpus

ids = corpus() * 4
n = len(ids)
for name, mode in (("recency", 0), ("first-occurrence", 1)):
    src = build_src_c(ids, mode)
    hit = tot = 0
    for i in range(n - 1):
        if src[i] >= 0:
            tot += 1
            hit += (ids[src[i]] == ids[i + 1])
    print(f"{name:18s} predicted {tot / n:.1%} of positions, next-word accuracy {hit / n:.3f} "
          f"(on predicted {hit / tot:.3f})")
