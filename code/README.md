# Code

Two things live here: the C suffix-automaton predictor used for every cost number in the write-up,
and the polling wrapper used to drive Kaggle during training runs. The training and evaluation code
itself is in `../kernels/` — one self-contained script per experiment.

## `c/rosa.c`

A suffix automaton that emits, for each position, the source index of the longest previous occurrence
of the current suffix (or −1). Two modes:

| mode | behaviour | cost |
|---|---|---|
| `0` — recency | reproduces the original pseudocode: stamp every state on the suffix-link chain with the current position | O(chain) per token → **quadratic on constant runs and repeated blocks** |
| `1` — first-occurrence | no stamping; the walk stops at the first chain state whose first end position is earlier | O(1) typical, **flat in n even on constant runs** |

Mode 1 trades exact-recency tie-breaking for a large constant-factor win on repetitive inputs. On
8-symbol random data it picks a different continuation than mode 0 at 58% of positions while
next-token accuracy is identical (0.126 vs 0.127) — the tie-break is arbitrary among equally long
matches. On word-level prose the difference is 11% of positions and 0.761 vs 0.759 accuracy. If you
need exact recency, use a link-cut tree (what `rosa-torch` does); mode 1 is the cheap approximation.

Storage: an open-addressed hash for `(state, symbol) → edge`, plus a per-state adjacency list so
cloning never scans the table. Arrays are sized for the theoretical maxima (2n−1 states, 4n+24 edge
slots, 6n hash slots); every write is guarded, and the function returns 2 rather than overrunning if
an adversarial input ever exceeded them.

Returns: `0` ok, `1` allocation failure, `2` capacity bound hit.

## `c/check.py`, `c/bench.py`, `c/bench2.py`, `c/acc.py`

`check.py` validates mode 0 against a brute-force implementation of the spec — 58 cases, alphabets
1–1000, lengths 1–300, plus constant runs, period-3, period-28, countdown and run-length inputs.
**58/58 identical.** It also reports how mode 1 differs from mode 0 (tie-break rate and next-token
accuracy). Start here; it takes ~10 s.

`bench2.py` is the main benchmark: it recompiles `rosa.c` with `clang -O3 -march=native`, checks C
against the Python reference on 200 random cases, then times four input shapes at 4k and 32k tokens in
both modes. `bench.py` is the earlier version, mode 0 only. `acc.py` measures what the tie-break change
costs in prediction quality.

Recorded output (`bench2.log`, bench.py's `bench.log`, `acc.log`), word-level prose, mode 1 vs mode 0:

| input | Python | C mode 0 (recency) | C mode 1 (first-occurrence) |
|---|---|---|---|
| random symbols, 32k | 2.48 µs/tok | 0.72 µs/tok | **0.48 µs/tok** |
| real prose (words), 32k | 1.24 µs/tok | 0.66 µs/tok | **0.57 µs/tok** |
| repeated 28-token block, 32k | 46.5 µs/tok | 3.43 µs/tok | **0.32 µs/tok** |
| constant run, 32k | 1,437.7 µs/tok | 136.1 µs/tok | **0.33 µs/tok** |

Absolute values move ±50% with machine and load; the *shape* is the result — mode 0 is quadratic on
repetitive input, mode 1 is flat everywhere. The prose corpus is `corpus.txt` (first 9,000 words of
Moby Dick, public domain); any UTF-8 text works, by `argv[1]` or `ROSA_CORPUS`.

**One convention to be careful with.** `bench2.py` prints `next-word accuracy 0.000`, which is
accuracy over *all* positions; `acc.py` prints `0.759`, which is accuracy over the positions where
ROSA fired (89.7% of them). The write-up quotes the `acc.py` number and says so. Do not mix them.

Requires `clang` and Python 3 (stdlib only).

## `scripts/monitor.sh`

The wrapper used for every Kaggle run: poll `kaggle kernels status`, break on COMPLETE/ERROR/CANCEL,
download the output directory, print the resulting JSONL. Kept because the poll/retry behaviour is
part of how the results were produced, and because `kaggle kernels output` silently returns nothing on
a non-final status, which is how several runs looked "empty" until the status check was added.
