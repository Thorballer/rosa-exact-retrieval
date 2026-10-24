# ROSA vs full attention: speed and recall, head to head

The rest of this repository tests ROSA against *windowed* attention inside a frozen model, mostly
through perplexity. This experiment asks the blunter question: put the compressed suffix index and
full causal attention on the same token stream, score the same positions, and see who wins on
**recall** and on **speed**.

Short answer:

- **Recall on verbatim repeats:** a tie. Full attention copies as well as the index at 1k and 3k
  distance (98.6–99.0% vs 100.0%), because a 0.5B transformer's induction heads are doing this job
  inside their own context window. The index is exact at every distance tested (1,024 / 3,072 /
  16,896) and needs nothing trained.
- **Recall on ordinary text:** attention wins by 2–5×. Code 73.9% vs 37.7%, prose 42.5% vs 8.5%.
  Exact matching has nothing to offer where the input does not literally repeat.
- **Speed:** on a T4 in fp16, one forward pass costs 140 µs/token at 1k context and 752 µs/token at
  8k, against 0.39–0.73 µs/token for the index — **350× to 1,900×**. On this experiment's laptop CPU
  the gap is ~21,000×. State is **12,288 bytes/token** of KV cache (fp16) against 146–178 bytes/token
  of index: **~80×**.

So the index wins on cost by two to five orders of magnitude and ties on recall exactly where it
applies — but "where it applies" is a narrower regime than attention's, and a modern model's own
context window already covers most of it.

## Design

Every arm is scored on the same positions, bucketed by one mask computed from the data
(`docs/methodology.md` #6).

Constructions (SPAN = 512):

| construction | sequence | what it measures |
|---|---|---|
| `dup` | `[A][F][A]`, FILLER 512 / 2560 / 16384 | exact recall: every position in the second copy has a verbatim match `dist` back |
| `dup-ctl` | `[F][A]` | same local context, earlier copy deleted — the leave-the-copy-out control |
| `plain` | a straight 1536-token window | ordinary prediction, no construction |

Arms:

| arm | what it is |
|---|---|
| `rosa` | the C suffix automaton in `code/c/rosa.c` (mode 1), predicting `x[src[i]]` |
| `full` | `Qwen/Qwen2.5-0.5B`, full causal attention over the whole sequence (32k context) |
| `win512` | the same model and positions, attention hard-masked to a 512-token window |

`win512` is what keeps the comparison from being a formality: it is the arm the rest of this repo
calls "attention", and it is where reach shows up.

Metric: top-1 next-token accuracy. On `dup` the target token *is* the token following the
corresponding source position, so "top-1 correct" means "copied the right token from `dist` back".
Buckets are by match length, which on `dup` equals the position inside the repeated span.

## Recall — Qwen2.5-0.5B fp16 on a T4 (16 sequences per cell, 8,176 positions)

95% CIs are bootstraps over *sequences*, not positions: positions inside one sequence are copies of
the same text, not independent draws.

| corpus | construction | distance | rosa | full attention | win512 |
|---|---|---|---|---|---|
| code | dup | 1,024 | **100.0%** [100–100] | 98.8% [99–99] | 53.4% [48–60] |
| code | dup | 3,072 | **100.0%** [100–100] | 98.6% [98–99] | 51.3% [47–56] |
| code | dup | 16,896 | **100.0%** [100–100] | not run¹ | not run¹ |
| code | dup-ctl | 1,024 | 34.9% [29–42] | 70.7% [67–75] | — |
| code | dup-ctl | 3,072 | 41.7% [32–52] | 73.5% [69–79] | — |
| code | plain | — | 37.7% [31–45] | **73.9%** [70–78] | — |
| prose | dup | 1,024 | **100.0%** [100–100] | 99.0% [99–99] | 25.7% [24–28] |
| prose | dup | 3,072 | **100.0%** [100–100] | 98.9% [99–99] | 26.9% [25–29] |
| prose | dup | 16,896 | **100.0%** [100–100] | not run¹ | not run¹ |
| prose | dup-ctl | 1,024 | 7.5% [6–9] | 40.2% [39–42] | — |
| prose | dup-ctl | 3,072 | 10.2% [9–11] | 40.8% [39–42] | — |
| prose | plain | — | 8.5% [7–11] | **42.5%** [41–45] | — |

¹ At 16,896 distance one attention matrix is 14 heads × 17,408² × 2 bytes = 8.5 GiB per sequence;
SDPA falls back to a materialised matrix on a 14.5 GiB T4 and OOMs. That cell measures the index
only, which is itself the point: the index does not care how far back the match is.

Retrieval-specific effect — `dup` minus its own `dup-ctl`, so local-context prediction is subtracted:

| corpus | distance | rosa | full attention |
|---|---|---|---|
| code | 1,024 | **+65.1 pts** | +28.1 pts |
| code | 3,072 | **+58.3 pts** | +25.1 pts |
| prose | 1,024 | **+92.5 pts** | +58.8 pts |
| prose | 3,072 | **+89.8 pts** | +58.1 pts |

The index's recall is more *of* the mechanism; attention's headline number leans harder on the fact
that code and prose are predictable in place.

By match length, `code / dup / 1024`:

| match length | n | rosa | full attention | win512 |
|---|---|---|---|---|
| 1–8 | 112 | 100.0% | 60.7% | 14.3% |
| 8–32 | 384 | 100.0% | 95.1% | 38.0% |
| 32–128 | 1,536 | 100.0% | 99.2% | 46.4% |
| 128+ | 6,144 | 100.0% | 99.7% | 56.8% |

Attention needs roughly 32 tokens of verbatim match before it copies as reliably as the index, and it
climbs smoothly; the index is at 100% from a one-token match because "the previous occurrence of this
suffix" is the thing it computes.

Index-arm facts from the same rows:

- fires on **100%** of `dup` positions, picks the true source position **100.0%** of the time on code
  and **99.99%** on prose — the recall is the mechanism, not luck;
- on `plain` code it fires on 78.7% of positions, but only 2.4% of those have a match ≥16 tokens
  (median match 2 tokens);
- on `plain` prose it fires on 66.3%, and 0.0% of positions have a match ≥16 tokens.

## Recall — same harness on a laptop CPU (fp32, 6 sequences per cell, distance 1,024)

Kept because it is the only arm where the model has no GPU advantage and because the corpus differs
(this repo's own kernels rather than the Python in a Kaggle image).

| corpus | construction | rosa | full attention | win512 |
|---|---|---|---|---|
| code | dup | **100.0%** | 99.1% [99–99] | 38.9% [32–45] |
| code | dup-ctl | 16.1% [10–24] | 59.0% [49–68] | — |
| code | plain | 20.2% [16–25] | **67.6%** [65–70] | — |
| prose | dup | **100.0%** | 99.5% [99–100] | 29.9% [22–44] |
| prose | dup-ctl | 11.2% [7–17] | 42.1% [34–55] | — |
| prose | plain | 10.6% [6–18] | **42.1%** [35–53] | — |

Same ordering, same magnitude; the code corpus here is harder for attention (67.6% plain vs 73.9%),
which is why the local `dup-ctl` numbers sit lower.

## Speed — measured, both machines

ROSA, build-and-query, single core, no GPU:

| machine | shape | n = 4k | 16k | 64k | 256k | 1M |
|---|---|---|---|---|---|---|
| Intel i5-8500B, `-march=native` | random (vocab 32k) | 0.09 µs | 0.11 µs | 0.15 µs | 0.25 µs | — |
| | duplicated span | 0.09 µs | 0.09 µs | 0.10 µs | 0.12 µs | — |
| | constant run | 0.08 µs | 0.09 µs | 0.09 µs | 0.12 µs | — |
| | real code tokens | 0.20 µs | 0.23 µs | 0.24 µs | — | — |
| Kaggle CPU (T4 box), `-O3` | random | 0.52 µs | 0.45 µs | 0.52 µs | 0.62 µs | 0.73 µs |
| | duplicated span | 0.39 µs | 0.40 µs | 0.40 µs | 0.48 µs | 0.58 µs |
| | constant run | 0.40 µs | 0.39 µs | 0.40 µs | — | — |
| | real code tokens | 0.40 µs | 0.40 µs | 0.45 µs | — | — |

Per-token cost tracks the *fire rate*, not n: a position with no match returns immediately, one with
a match walks the suffix-link chain. That is why random data drifts upward as n grows (more matches
become available) while the duplicated span stays flat. Memory is 178 bytes/token requested
(146–175 bytes/token peak RSS above baseline) — this is the build-time automaton, not a compacted
persistent form.

The known algorithmic cliff, for scale (mode 0, exact-recency): a constant run costs **3.72 µs/token**
at n = 4k and **42.07 µs/token** at n = 16k, i.e. quadratic and 483× mode 1. Mode 1 exists to remove
it.

Full attention, Qwen2.5-0.5B:

| machine | T | prefill ms/token | KV bytes/token |
|---|---|---|---|
| T4, fp16, batch 1 | 512 | 0.145 | 12,288 |
| | 1,024 | 0.140 | 12,288 |
| | 2,048 | 0.215 | 12,288 |
| | 4,096 | 0.372 | 12,288 |
| | 8,192 | 0.752 | 12,288 |
| Intel i5, fp32, 6 threads | 512 | 5.67 | 24,576 |
| | 1,024 | 5.27 | 24,576 |
| | 1,536 | 5.13 | 24,576 |
| | 2,048 | 5.69 | 24,576 |
| | 3,072 | 6.02 | 24,576 |

Head-to-head, per token, at equal context:

| comparison | ratio |
|---|---|
| T4 fp16 forward vs index, T = 1k | 140 µs / 0.40 µs ≈ **350×** |
| T4 fp16 forward vs index, T = 8k | 752 µs / 0.40 µs ≈ **1,900×** |
| i5 fp32 forward vs index, T = 1.5k | 5.13 ms / 0.24 µs ≈ **21,000×** |
| state per token, fp16 KV vs index | 12,288 / ~150 ≈ **80×** |
| state per token, fp32 KV vs index | 24,576 / ~150 ≈ **164×** |

Caveats that belong in the same breath as those ratios:

- A forward pass is the whole model. The honest framing is "the retrieval mechanism costs
  1/350th–1/20,000th of the forward pass it sits inside", not "attention is slow at retrieving".
- The automaton is serial and CPU-only (`docs/prior-art.md`), which is exactly why the GPU row is two
  orders of magnitude less flattering than the CPU row: attention gets 36× faster on a T4, ROSA gets
  ~2× *slower* (the Kaggle CPU is weaker than the laptop).
- Naive batch-1 decode on the T4 measured 27–36 ms/token. That is Python and full-vocabulary
  logits, not attention; it is reported only so nobody mistakes it for an attention cost.
- The standalone SDPA microbenchmark in `kernels/rosa-vs-full-attention-speed` fell back to a
  slow path (`enable_gqa` on torch 2.11/T4) and reported ~5 ms/token of "attention" at T = 1k, which
  is 30× the theoretical cost. It is in the raw results and **not** used above; the prefill slope is
  the usable measurement (≈85 µs per token per additional 1k of context at batch 1).

## What this does and does not show

- **Does:** on guaranteed verbatim repeats the index is exactly right at any distance, for ~150 bytes
  and a few hundred nanoseconds per token, with no training and no GPU. Full attention matches it at
  1k–3k distance and cannot be run at 17k on a 14.5 GiB card.
- **Does:** outside the repeat regime the index is not "slightly worse", it is 2–5× worse on plain
  prediction — and on code it fires often (78.7% of positions) on matches too short to be worth
  anything, which is the mechanism behind the harm measured in `rosa-generalize-aggressive`.
- **Does not:** support "ROSA replaces attention". A 0.5B model already does exact copy inside its
  context window; the index's advantage is cost and unbounded distance, not capability.
- **Does not:** measure a hybrid. The architecture that would use both — exact candidate generation,
  learned rerank, learned gate, explicit NULL — is argued in `docs/prior-art.md` and is not tested
  here.

## Corrections and dead ends

- **Off-by-one in the first pass.** `match_len` compared positions i-1 and s-1, dropping the current
  token, and `src_is_true` compared against `i - dist` when ROSA's source is `i - dist + 1`
  (`src = match_end + 1`). `repost.py` rebuilds the masks and the source metric and asserts that
  every stored ROSA prediction still equals the token at its own source position (24,540 rows, 0
  mismatches). Model predictions were never affected.
- **Three OOMs on Kaggle before the harness was right:** a 32×1536×151,936 fp32 logits tensor (29 GiB),
  then a 4-sequence chunk at T = 17,408 (21 GiB), then SDPA materialising the T×T matrix. The fix that
  stays: project only the *scored* hidden states through `lm_head`, not all T positions.
- **A window arm that could not work.** Passing global `position_ids` to a 512-token input fails
  because the rotary cache is sized to the input length (index 1024 out of bounds for 512). Replaced
  by an additive 4D mask (`j <= i and i - j < 512`) on the full sequence, which is what windowed
  attention actually is.

## Improving recall at 1k–3k: what worked, and what it cost

On clean copies there is no headroom — the index is at 99.8% at both distances. So the upgradable
recall only exists where exact matching *fails*. Three such regimes were built, and five families of
cheap upgrade tested against them (`upgrades.c`, `upgrades.py`, kernel
`rosa-vs-full-attention-upgrades`). No training, no floats, same positions for every arm.

| construction | what breaks | exact index | best upgrade |
|---|---|---|---|
| `edit2` / `edit5` | two 2- or 5-token edits, a third of the span away | 99.2–99.4% | **nothing** — exact matching already recovers past the edit |
| `edit_near` | one token edited *immediately before* the query, every 8 positions | **0.2–0.7%** | **seed2/seed4: 98.2–98.4%** |
| `ambig` | three prior occurrences, successors (a, a, b) | 2.5–4.1% (majority truth) | **vote: 95.1–98.3%** |

Full attention, same corpus and positions:

| construction | distance | exact index | best upgrade | full attention |
|---|---|---|---|---|
| `edit_near`, code | 1,024 | 0.7% | **98.2%** (seed2) | 71.7% |
| `edit_near`, prose | 1,024 | 0.2% | **98.4%** (seed) | 67.9% |
| `edit_near`, code | 3,072 | 0.7% | **98.3%** (seed2) | 66.9% |
| `edit_near`, prose | 3,072 | 0.7% | **98.4%** (seed) | 66.5% |
| `ambig`, code | — | 4.1% / 3.0% (maj/rec) | **95.1% / 0.4%** (vote) | 12.6% / 4.1% |
| `ambig`, prose | — | 2.5% / 0.7% | **98.3% / 0.0%** (vote) | 14.1% / 1.6% |

**Upgrade 1 — seed-and-extend.** A suffix-anchored index cannot find the source when the token
immediately before the query is corrupted: every gram ending at the query contains the corruption,
so there are no candidates at all (the index abstains, or fires on a 1-token match and is wrong).
Anchoring 4-grams at offsets 0–3 *before* the query and extending with ≤2 mismatches recovers
**98.2–98.4%**, 27–31 points ahead of full attention, which pays for its own fuzziness with a forward
pass and still gets a third of these wrong.

**Upgrade 2 — length-weighted voting.** When several prior occurrences disagree, "longest match wins"
is not a tie: the tokens *preceding* each occurrence differ, so the longest suffix is decided by one
extra matched token, and that token is the previous occurrence's successor. A weighted vote over all
candidates (weight = match length, PPM-style) picks the majority continuation instead: **95.1–98.3%**
where longest-match gets 2.5–4.1%, and 80+ points ahead of attention. It is a policy, not a truth:
against the *most recent* continuation it scores 0.0–0.4%, where the base rule gets 0.7–3.0%.

Both are close to free where they are not needed, and neither helps where the index already works:

| | base | majority | vote | seed2 | seed4 | gate8 |
|---|---|---|---|---|---|---|
| `dup`, code, 1,024 | **99.8** | 99.7 | 99.4 | 99.4 | 99.4 | 98.6 |
| `edit2`, code, 1,024 | **99.3** | 99.0 | 98.2 | 98.6 | 98.6 | 95.8 |
| `edit5`, code, 1,024 | **99.2** | 99.0 | 98.1 | 97.4 | 95.0 | 95.8 |
| `plain`, code (local run) | **12.4** | 12.8 | 12.8 | 2.9 | 0.8 | 0.5 (1% fire) |

Abstention (`gate8`: fire only on ≥8-token matches) does what it says and nothing more: fire rate on
plain code drops 54% → 1%, accuracy 12.4% → 0.5%. Precision without a reranker is not a win.

### What the upgrades cost

Microseconds per token, single core, C (`upgrades_bench.c`, one process per cell):

| shape | n | base | majority | vote | seed2 | seed4 | gate8 | SAM, mode 1 |
|---|---|---|---|---|---|---|---|---|
| random (no repeats) | 4k | 0.09 | 0.09 | 0.09 | 0.12 | 0.12 | 0.09 | 0.09 |
| | 16k | 0.10 | 0.10 | 0.10 | 0.12 | 0.12 | 0.10 | 0.11 |
| | 64k | 0.10 | 0.10 | 0.10 | 0.13 | 0.13 | 0.10 | 0.15 |
| duplicated span | 4k | 5.1 | 5.2 | 4.9 | 12.9 | 12.8 | 5.1 | 0.09 |
| | 16k | 117 | 163 | 135 | 276 | 383 | 123 | 0.10 |
| | 64k | >8 s | >8 s | >8 s | >8 s | >8 s | >8 s | 0.10 |
| constant run | 4k | 455 | 291 | 484 | 1,147 | 825 | 457 | 0.08 |

This is the finding that matters more than any accuracy number above. The suffix automaton is
O(1) amortised per token *because* it returns one source per position from a suffix-link walk.
A candidate engine enumerates up to 48 occurrences per chain and extends each one, so its cost is
O(candidates × match length): on repetitive input — the exact input this mechanism exists for — it is
**50–1,000× the sampler's cost, and quadratic in n**. The seed variants are 2–3× worse than the base
candidate engine there. Neither upgrade preserves the property that made the index cheap, and a
serious implementation would need the same first-occurrence trick (depth-limited walks, or a
suffix-link tree) to get the recall without the cliff.

## Files

| file | role |
|---|---|
| `recall.py` | local (CPU) recall run: `--nseq 6 --corpora code,prose` |
| `rosa_speed.py`, `rosa_bench.c` | ROSA throughput and memory, one process per cell for a clean RSS |
| `attention_speed.py` | prefill / decode / KV on this machine |
| `analyze.py` | markdown tables with bootstrap CIs, reads either results format |
| `repost.py` | the corrected mask / source-position pass |
| `upgrades.c`, `upgrades.py`, `upgrades_bench.c` | the recall upgrades: variants, constructions, cost |
| `results/upgrades-main.jsonl` | local cross-check of every variant (3,068 rows) |
| `results/kaggle-t4-upgrades.jsonl` | T4 run: variants and full attention, same positions (18 MB) |
| `results/upgrades-cost.jsonl` | per-variant cost sweep |
| `results/recall-main.fixed.jsonl.gz` | local CPU run, corrected (55,212 rows) |
| `results/kaggle-t4-recall.jsonl.gz` | T4 run, both corpora, three distances (245,344 rows) |
| `results/rosa-speed.jsonl`, `results/attention-speed.jsonl` | local speed cells |
| `kernels/rosa-vs-full-attention/` | the T4 recall kernel |
| `kernels/rosa-vs-full-attention-speed/` | the T4 speed kernel |
| `kernels/rosa-vs-full-attention-upgrades/` | the T4 upgrade kernel: five constructions, 13 variants, both arms |
