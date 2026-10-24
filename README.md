# Exact suffix retrieval inside a transformer: an independent evaluation of ROSA

ROSA — *Rapid Online Suffix Automaton* — is the neurosymbolic exact-match mechanism proposed in
the RWKV-8 write-up (Oct 11, 2025) as a route "beyond attention": index the token stream with a
suffix automaton and, at every position, copy the token that followed the longest previous
occurrence of the current suffix. No dot product, no softmax, no floats, no KV cache.

This repository is an attempt to measure what that is actually worth inside a language model,
rather than on a copying benchmark. It contains the 25 Kaggle kernels that produced the numbers,
every raw results file — including the invalid, degenerate and crashed runs — and the C
implementation used for the cost measurements.

## Bottom line

The mechanism is real, exact, and cheap. Its benefit is narrow, and most of the large gains
reported for this class of mechanism — including the ones this project produced in its own first
half — are not retrieval.

Inside a **frozen Qwen2.5-0.5B** with a 512-token attention window, an exact-retrieval sidecar is
worth:

| regime | retrieval-specific effect (PPL, vs an identically-trained fixed-vector control) |
|---|---|
| long verbatim match **outside** the window | **1.3–1.9× better** |
| match **inside** the window | **0.86× — worse than not having it** |
| ordinary prose (mean match 0.78 tokens) | **~1.0× — inert** |
| matches present but unreliable (surface-renamed) | **2.4× worse**, absent a working verifier |
| any of the above, measured against no control at all | 5–8× "better" — this is the artifact |

The index itself: **exact** (400/400 against a brute-force reference), **linear** (≤2n−1 states,
~1.18 states/token), and **0.3–0.7 µs/token in C** once one algorithmic cliff is removed. Memory
is ~10–40 bytes/token against ~12 KB/token for a KV cache — roughly **300–1000× cheaper context
memory**, and that is the one advantage that does not depend on the regime.

What did not survive: "ROSA makes recall learnable that attention cannot learn" (a phase
transition that flips with the LR schedule), "8.6× perplexity from exact retrieval" (85% of it was
the trainable pathway), and the metadata-conditioned verifier that was supposed to make loose
matching safe (beaten by its own ablation across three seeds).

Read [`docs/findings.md`](docs/findings.md) for the full result set with the corrections that
produced it, [`docs/methodology.md`](docs/methodology.md) for the eight ways this project's own
experiments misled it, and [`docs/experiments.md`](docs/experiments.md) for a kernel-by-kernel
register including every run that had to be thrown away.

## What ROSA is

Given a token stream `x`, ROSA returns `y` where

```
y[i] = x[j+1]   if there exists j < i and m ≥ 0 with x[j-m:j] == x[i-m:i]
               (maximise m; break ties by largest j)
     = -1       otherwise
```

— the longest previous occurrence of the current suffix, and what followed it. One suffix-automaton
walk per token answers every match length at once (`≤2n−1` states, amortised O(1) per appended
symbol). It is parameter-free, non-differentiable, and lives happily on the CPU next to a GPU doing
the neural work.

Two things follow, and they drive everything in this repo:

- **It is a perfect induction head.** Attention learns a fuzzy, window-bounded version of exactly
  this function; ROSA does it exactly, with unbounded reach, for free.
- **It can only fire on verbatim repetition.** "Where did this exact sequence appear before" is a
  question about surface strings, so any benefit is conditional on the input containing long
  literal repeats — code, logs, chat history, boilerplate, self-quoting documents.

The reference implementation assessed here is [`aabbdev/rosa`](https://github.com/aabbdev/rosa)
(`rosa-torch`, MIT). Its behaviour was verified against its own brute-force `reference_rosa` on
280/280 positions, and it inherits the off-by-one convention of the original snippet: `y[i]`
predicts token `i+1`, because the current token is inserted into the automaton before the query.

## What was measured

Two harnesses, both on a single Kaggle T4:

- **Synthetic** (`d=96`, 2 layers, trained from scratch): associative recall where the paired value
  is always outside a deliberately small attention window, with attention, windowed attention, and
  windowed/attention + ROSA arms, plus harm controls where copying is always wrong.
- **Real model** (`Qwen/Qwen2.5-0.5B`, frozen): attention replaced by a 512-token sliding window so
  there is a genuine reach deficit, a ROSA sidecar injected at layers 6/12/18 with a learned value
  projection and a learned read gate, gradients confirmed to reach only the sidecar
  (`llm_params_with_grad: 0`). Evaluated on Python source, on prose, and on a duplicated-chunk
  construction that guarantees long matches (n = 30,690 scored positions, 97% with matches ≥32
  tokens).

The single most important design decision: **every arm is compared against a control that has the
same trainable pathway but no real retrieval content.** Without that, a residual pathway alone
improves a frozen model's perplexity by 5–8×, which is easy — and, in this project, temporarily
successful — to mistake for the mechanism working.

| experiment | question | verdict |
|---|---|---|
| `rosa-vs-attention-copy-recall` | does ROSA beat attention on synthetic recall? | **bistable / schedule artifact** |
| `rosa-long-context-distance` | does reach hold at 4k with 32–4000-token gaps? | **void** — baseline never left chance |
| `rosa-symbolic-vs-attention` | token-indexed ROSA vs attention, distance-resolved | **100% at every distance, 3/3 seeds** vs attention 13% |
| `rosa-gate-identifiable` | can a learned gate veto bad retrievals? | **yes, when the decision is decidable from the input** |
| `rosa-codebook-ksweep-orig` | is the learned codebook broken or mis-sized? | **code-utilisation problem; monotone in K** |
| `rosa-real-lm-sidecar` | does it help a frozen LM? | 8.1× gross, **1.3–1.4× retrieval-specific** |
| `rosa-killtest-prose-vs-code` | code vs prose, with a fixed-vector control | code ~1.3×, prose **1.10×** |
| `rosa-definitive-match-length` | does the gain grow with match length? | **prediction falsified** — it shrinks and reverses |
| `rosa-generalize-aggressive` | does it survive surface renaming? | **2.4× worse than no sidecar** |
| `rosa-verifier-3seeds` | can a metadata verifier make that safe? | **dead** — ablation wins on all 3 seeds |
| `rosa-definitive-match-length` / `rosa-aux-and-multilayer` | accuracy levers without memory cost | 3-layer injection **1.12×** for 2.4M params; aux head confounded with 410M params |
| `rosa-lora-vs-frozen` | does any of it survive fine-tuning? | fine-tuning dominates ~10:1; sidecar effect ±5% and **negative** with a live sidecar |

## ROSA against full attention, directly

`experiments/rosa-vs-full-attention/` puts the index and Qwen2.5-0.5B's full attention on the same
token stream and scores the same positions. On verbatim repeats they tie at 1k–3k distance (100.0%
vs 98.6–99.0%), and the index is still 100% at 16,896 where attention will not fit on a T4; on
ordinary text attention wins 2–5× (code 73.9% vs 37.7%). Cost per token: 0.39–0.73 µs for the index
against 140–752 µs for an fp16 forward pass on that T4 (350× to 1,900×), and ~150 bytes of state
against 12,288 bytes of KV cache.

## Where it is categorically better — and why that is weaker than it sounds

Synthetic associative recall: 512-token context, 8 key→value pairs, 8 query keys, 128-token
attention window, chance 1.6%, `d=96`, 2 layers, 3 seeds, token-indexed ROSA (no learned codebook):

| arm | accuracy | by retrieval distance: <128 / 128–256 / 256–384 / 384+ |
|---|---|---|
| full attention | 13.3% | 13 / 13 / 13 / 14 |
| windowed attention (128) | 12.9% | **25 / 24 / 2 / 1** |
| window + ROSA | **100%** | 100 / 100 / 100 / 100 |
| attention + ROSA | **100%** | 100 / 100 / 100 / 100 |

Two things are worth taking from this. First, the windowed-attention distance profile — 25% inside
the window, chance beyond it — is the cleanest measurement of what "reach" means in this project.
Second, on this task the difference is categorical, not marginal: attention holding the whole
512-token context still saturates at 13%, because it has to *learn* the copy from 16 supervised
examples, while ROSA gets it by construction.

That is also the caveat. This task is exactly the function ROSA computes, so it is a test of the
index, not evidence of value on real text — which is the whole reason the real-model experiments
below exist. A harm control (targets re-assigned so copying is always wrong) puts every ROSA arm at
chance accuracy but at **loss 6.2–6.7 against attention's 4.2–4.4**: a closed gate on top of a
confidently wrong pathway, which is the plausible engine of a long-context collapse. Give the gate
a decidable signal — a flag token stating whether copying is correct — and it separates cleanly:
**gate 0.97–0.99 on copy vs 0.017–0.020 on harm**, copy accuracy 1.000, both attention baselines at
6–13%.

## The one number that matters

Duplicated-chunk construction, frozen Qwen2.5-0.5B, 512-token window, perplexity on the second half
by match length (n = 30,690 positions):

| arm | 1–8 (n=210) | 8–32 (720) | 32–128 (2,880) | 128–512 (11,520) | 512+ (15,360) | all |
|---|---|---|---|---|---|---|
| no sidecar | 617.8 | 97.4 | 29.2 | 18.02 | 16.41 | 19.18 |
| **real retrieval** | **173.6** | **37.6** | **14.8** | **11.61** | 17.27 | **15.18** |
| const-vector control | 331.4 | 58.9 | 20.8 | 14.97 | 14.78 | 16.18 |
| scrambled-source control | 362.0 | 63.9 | 21.2 | 15.20 | 14.65 | 16.28 |

Retrieval-specific gain (const ÷ real) is **1.91× at 1–8 tokens, 1.57× at 8–32, 1.41× at 32–128,
1.29× at 128–512, and 0.86× beyond 512** — it shrinks where I predicted it would grow, because in
this construction match length and remaining in-window context move together. Aggregate
retrieval-specific effect: **1.07×**.

That is the honest shape of the result: exact retrieval substitutes for context the model cannot
reach, and costs ~5–14% where it can.

## Reproducing

```bash
# every experiment is a self-contained Kaggle script kernel
ls kernels/                      # 25 kernels, each with its kernel.py + kernel-metadata.json
cat results/README.md            # which raw run is canonical, which are void and why

# the cost numbers run locally in ~30 s (needs clang)
cd code/c && python3 check.py && python3 bench2.py && python3 acc.py
```

`code/c/rosa.c` is a compact C suffix-automaton predictor with two modes: `mode 0` reproduces
ROSA exactly (per-position recency stamp), `mode 1` replaces the recency stamp with a
first-occurrence timestamp and removes the only quadratic cliff in the algorithm. `bench.log` and
`bench2.log` are the recorded outputs.

## Scope, and what this is not

- Everything real-model was measured on a **frozen** 0.5B model. A null there is ambiguous: it does
  not show that retrieval cannot be learned, only that a frozen backbone plus a sidecar does not
  learn it in 400 steps. The one fine-tuning run (`rosa-lora-vs-frozen`) is a single seed with an
  eval-hook bug that took four attempts to get right, and its conclusion is that fine-tuning
  dominates everything else by ~10:1.
- Nothing here tests ROSA **replacing** attention at scale. The closest result is window+ROSA
  against full attention on a synthetic task.
- These are mixed and negative results on purpose. The value of the repo is the control methodology
  plus the cost profile, not a headline.

## Layout

```
kernels/<kaggle-slug>/       27 kernels, exactly as pushed
results/<slug>[*].jsonl      raw JSONL output, one line per measurement
code/c/                      C suffix automaton, brute-force check, benchmarks, logs
code/scripts/monitor.sh      Kaggle poll/harvest wrapper used throughout
docs/findings.md             full findings, with the correction chain
docs/experiments.md          kernel register: question, result, verdict, failure mode
docs/methodology.md          the traps that cost a day
docs/prior-art.md            taxonomy and external evidence
experiments/rosa-vs-full-attention/  ROSA index vs full attention: recall and speed, head to head
```

## License

MIT — see [`LICENSE`](LICENSE). The ROSA mechanism is due to the RWKV-8 ROSA write-up; the
reference implementation belongs to [`aabbdev/rosa`](https://github.com/aabbdev/rosa). Everything
in this repo is an independent evaluation of both, and reuses no code from either.
