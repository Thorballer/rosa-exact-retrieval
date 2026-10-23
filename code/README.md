# Code

Two things live here: the C suffix-automaton predictor used for every cost number in the write-up,
and the polling wrapper used to drive Kaggle during training runs. The training and evaluation code
itself is in `../kernels/` — one self-contained script per experiment.

## `c/rosa.c`

A 113-line suffix automaton that emits, for each position, the source index of the longest previous
occurrence of the current suffix (or −1). Two modes:

| mode | behaviour | cost |
|---|---|---|
| `0` — recency | reproduces the original pseudocode: stamp every state on the suffix-link chain with the current position | O(chain) per token → **quadratic on constant runs and repeated blocks** |
| `1` — first-occurrence | no stamping; the walk stops at the first chain state whose first end position is earlier | O(1) typical, **flat in n even on constant runs** |

Mode 1 trades exact-recency tie-breaking for a 4–80× constant factor on real inputs; measured effect
on predictions over 30,376 word tokens is next-word accuracy 0.766 → 0.764, with a different
continuation chosen at 9.9% of positions. If you need exact recency, use a link-cut tree (what
`rosa-torch` does); mode 1 is the cheap approximation.

Storage: an open-addressed hash for `(state, symbol) → edge`, plus a per-state adjacency list so
cloning never scans the table.

## `c/bench.py`, `c/bench2.py`, `c/acc.py`

`bench2.py` is the useful one: it recompiles `rosa.c` with `clang -O3 -march=native`, checks the C
output against the Python reference (200/200 positions), then times four input shapes at 4k and 32k
tokens in both modes. `bench.py` is the earlier version that only had mode 0. `acc.py` measures what
the tie-break change costs in prediction quality.

`bench.log` and `bench2.log` are the recorded outputs from the authoring machine. Re-running gives
the same *shape* and slightly different absolute numbers (machine, load, cache state) — the ranges
quoted in the write-up (0.3–0.7 µs/token for mode 1, 0.9–1500 µs/token for mode 0 depending on
input) come from these logs.

**One convention to be careful with.** `bench2.py` prints `next-word accuracy 0.003`, which is
accuracy over *all* positions; `acc.py` prints `0.764`, which is accuracy over the positions where
ROSA fired (89% of them). The write-up quotes the `acc.py` number and says so. Do not mix them.

Requires `clang` and Python 3 (stdlib only).

## `scripts/monitor.sh`

The wrapper used for every Kaggle run: poll `kaggle kernels status`, break on COMPLETE/ERROR/CANCEL,
download the output directory, print the resulting JSONL. Kept because the retry/poll behaviour is
part of how the results were produced, and because `kaggle kernels output` silently fails on a
non-final status, which is how several runs looked "empty" until the status check was added.
