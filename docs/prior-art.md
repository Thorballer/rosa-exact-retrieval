# Prior art, and how to place an "exact match" mechanism

Nothing in the ROSA family is new as a data structure. The novelty on offer is *integration*: putting
a suffix index inside a transformer stack as a differentiable-ish side channel, and paying for it in
CPU cycles and bytes rather than FLOPs and KV cache. This page is the positioning material, so a
write-up does not overclaim.

## The taxonomy

| structure | answers | cost |
|---|---|---|
| suffix automaton (SAM) | "longest suffix of what I have seen that occurred before", online, all key lengths at once via one suffix-link walk | ≤2n−1 states, amortised O(1) per appended symbol |
| suffix array / FM-index | same queries over static text, sublinear search, self-index | O(n) build, indexed queries |
| LZ77 match finder / hash chains | the practical version: fixed window, per-order hash tables, verify | O(1) expected per probe, needs a fixed k |
| PPM / PAQ / cmix | variable-order context mixing; the match model *is* the LZ77 predictor; strongest classical compressors | linear-ish, memory hungry |
| kNN-LM / RETRO / neural cache | float keys, ANN search, external datastore | datastore + per-token search |
| induction head | the learned, fuzzy, window-bounded version of the same function | inside attention |

ROSA is the first row, wired into a layer stack. The LZ77 match model in PAQ/PPM has run exactly this
predictor for twenty years; infini-gram runs the suffix-array version at scale; kNN-LM runs the
neural version. What is new is the *coupling* and the cost profile, not the mechanism.

## The arguments that decide design questions

- **Equality of discrete symbols is what makes indexing possible.** Continuous vectors have no "same
  suffix", so a linear layer in front of a suffix automaton is a no-op — you get a broken index, not
  fuzzy retrieval. You must quantise (argmax to a codebook) first, and then you have a *learned
  hash*. Measured here: quantising hidden states gives retrieval at chance (0.007–0.029); quantising
  the token symbol gives 0.93–1.0.
- **A learned hash overfits like an embedding.** It learns which things to collide on from the
  training distribution, so it gains on seen categories and degrades under shift. Fuzzier codes,
  worse transfer — and the mechanism needs K ≫ the number of distinct symbols for codes to be
  usable at all (at K = 4096, only 81% of 192 tokens got their own code).
- **Sublinear indexing requires a fixed metric.** A metric you are still training admits only linear
  scans, so "soft matching over the full context" is attention with extra steps and no softmax
  normalisation.
- **Attention is soft retrieval over a bounded window.** The bound is why long context costs O(T²);
  that is the price of fuzziness, not a wart to be removed. A hard index buys unbounded reach and
  gives up fuzziness, which is exactly the trade measured in `docs/findings.md` (F7: helps beyond the
  window, costs 14–17% where the window suffices).
- **The architecture that works**: exact candidate generation → learned rerank over K candidates →
  learned read gate → explicit NULL. Fuzziness where it is cheap (K candidates), exactness where it
  is cheap (the index). Everything measured here that *helped* fits this shape; the parts that failed
  (multi-order mixing, hard length gates, metadata verifiers) were attempts to improve the rerank
  step, and plain "longest match, most recent" beat them all.
- **A symbolic copier aces copy benchmarks by construction** (MQAR, NIAH). 100% NIAH is a statement
  about the benchmark, not about capability. Measured here on a purpose-built copy task: 100% at
  every distance versus 13% for full attention — genuinely categorical, and genuinely a copy task.

## Cost figures worth reusing (measured)

- SAM ≈ **1.18 states/token** on word-level prose, ≤2n−1 regardless of match length. A hash index over
  all k ≤ 16-grams is **~15× n keys**. Compressed index versus dictionary — that is what the
  automaton buys.
- Word-level exact-context match on English prose: **~6%** next-word accuracy (20–25% at
  subword/token level, which is the number usually quoted). Always state the tokenisation.
- C implementation with hashed transitions: **0.33–0.7 µs/token**, flat in n, including constant runs
  once the recency stamp is replaced by a first-occurrence timestamp. Pure Python: 1.5–1483 µs/token
  depending on input shape. `rosa-torch`'s stateful path measured 25–60 µs/token, i.e. ~90% of the
  practical cost is library and interface overhead, not the automaton.
- Per-token memory: SAM ~10–40 bytes versus **~12 KB** for a Qwen2.5-0.5B KV cache (24 layers ×
  2 KV heads × 64 dim × 2 tensors × 2 bytes) — roughly 300–1000× less.
- **The automaton is serial.** GPUs cannot host it; budget CPU-side streaming cost. In training runs
  this appeared as 13× slower wall-clock per step.

## External evidence worth citing (attribute, never assert)

- **ROSA-Tuning** (arXiv 2602.02499, independent): window attention + ROSA beat global attention on
  PG-19 at 16k context on perplexity (17.63 vs 18.96; window attention alone 74.5). Its own later
  update reports the advantage collapsing to marginal versus window attention on LongBench, which the
  authors attribute to the discretisation structure's limited generalisation. That trajectory is
  exactly what the measurements here predict: a memorisation channel that is excellent at exact copy
  and unsafe where matches are unreliable.
- **kNN-LM**: WikiText-103 perplexity ~18.7 → ~16.1. Real, modest, datastore-bound — and the same
  control discipline (compare against a fixed/random datastore) is what keeps the number honest.
- **`rosa-torch`** (github.com/aabbdev/rosa, MIT): the reference implementation. Verified here against
  its own `reference_rosa` on 280/280 positions; inherits the off-by-one convention (`y[i]` predicts
  token `i+1`); replaced the original snippet's quadratic recency stamp with a link-cut tree; throws
  no light on whether the mechanism helps a model, because it ships no training results.

## Two things the write-up of any such mechanism must state

1. **The regime.** "Matches or beats attention" is only meaningful scoped to verbatim repetition
   beyond the window. Outside that regime this family of mechanism measures at ~1.0× and is
   sometimes worse.
2. **The control.** Any perplexity improvement from an added channel must be reported net of an
   identically-trained scrambled-source or constant-vector arm. Without it the reported gain is
   mostly the channel, and it will be wrong by roughly 6×.
