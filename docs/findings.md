# Findings

Everything below is measured, not asserted. Each section names the kernel that produced it, so a
number can be traced to `kernels/<slug>/kernel.py` and its raw output in `results/`.

Two harnesses throughout:

- **Synthetic** — `d=96`, 2 layers, trained from scratch on associative recall: 512-token context,
  8 key→value pairs, 8 query keys, 128-token window, chance 1.56%, Kaggle T4, 3 seeds unless noted.
- **Real model** — `Qwen/Qwen2.5-0.5B`, **frozen**, attention replaced by a 512-token sliding
  window, ROSA sidecar injected at layers 6/12/18 with a learned value projection and a learned
  read gate. Gradient isolation verified each run (`llm_params_with_grad: 0`, non-zero sidecar
  gradient). Evaluated on Python source (`in` = positions inside the window, `out` = beyond it),
  on Gutenberg prose, and on a duplicated-chunk construction that guarantees long matches.

---

## F1. The index is exact, linear, and small

Against a brute-force implementation of the stated definition, ROSA's output is **400/400
identical** (`rosa-symbolic-vs-attention`, oracle check; the reference implementation's own stateful
path also matched its `reference_rosa` on 280/280 positions).

Size measurements on 7,594 word tokens of prose:

| structure | size | per token |
|---|---|---|
| suffix automaton | 8,938 states, 16,322 transitions | **1.18 states/token** |
| hash index, all k ≤ 8-grams | 55,149 keys | 7.3× |
| hash index, all k ≤ 16-grams | 115,801 keys | 15.2× |

The automaton is a *compressed* index: ≤2n−1 states regardless of match length, against a
dictionary of k-grams that grows with the window. Per token it costs ~10–40 bytes; a KV cache for
Qwen2.5-0.5B costs ~12 KB (24 layers × 2 KV heads × 64 dim × 2 tensors × 2 bytes). That is roughly
**300–1000× less context memory**, and it is the one advantage of this mechanism that is not
regime-dependent.

## F2. The cost: one algorithmic cliff, and it is the case you care about

`code/c/rosa.c`, word-level prose, single machine, verified identical to the Python reference
(200/200) in both modes:

| input | Python | C (exact recency) | C (first-occurrence) |
|---|---|---|---|
| random symbols, 32k | 1.55 µs/tok | 0.46 µs/tok | **0.36 µs/tok** |
| real prose (words), 32k | 3.78 µs/tok | 0.94 µs/tok | **0.68 µs/tok** |
| repeated 28-token block, 32k | 78.7 µs/tok | 6.20 µs/tok | **0.52 µs/tok** |
| constant run, 32k | 1,482.9 µs/tok | 122.7 µs/tok | **0.33 µs/tok** |

The cliff is the `while e[v] < i: e[v] = i` most-recent-occurrence stamp in the original
pseudocode: it walks the suffix-link chain on every token, so it is **quadratic in exactly the
inputs where the mechanism earns its keep** — constant runs, boilerplate, repeated blocks. In
Python it is 1483 µs/token at 32k on a constant run, 80× the random-input cost.

Replacing the recency stamp with a **first-occurrence** timestamp (no propagation; the walk stops at
the first chain state whose first end is earlier) removes it: **0.33 µs/token, flat in n**. Measured
effect on predictions over 30,376 word tokens: next-word accuracy **0.766 vs 0.764**, different
continuation chosen at 9.9% of positions. A link-cut tree is the exact-recency fix (what
`rosa-torch` uses); first-occurrence is the cheap one.

Also worth internalising: automaton work is **serial**, so GPUs do not help it. In the training
runs this showed up as **13× slower wall-clock per step** (162 s vs 12 s for the same 1200-step
synthetic run) because the automaton runs in Python on the CPU. Budget CPU-side streaming cost, not
FLOPs.

## F3. On a copying task it is categorically better — and that is the weak result

`rosa-symbolic-vs-attention`, token-indexed ROSA (index raw token ids, no learned codebook),
3 seeds per arm:

| arm | accuracy | <128 | 128–256 | 256–384 | 384+ |
|---|---|---|---|---|---|
| full attention | 13.3% | 13 | 13 | 13 | 14 |
| windowed attention (128) | 12.9% | **25** | **24** | 2 | 1 |
| window + ROSA | **100%** | 100 | 100 | 100 | 100 |
| attention + ROSA | **100%** | 100 | 100 | 100 | 100 |

loss 0.000 on every ROSA cell, every seed, every distance bucket. Attention holding the entire
512-token context still saturates at 13%, because it has to learn the copy from 16 supervised
examples; ROSA gets it by construction. The windowed-attention row (25% inside its window, chance
beyond) is the cleanest measurement in this project of what "reach" buys and what a window costs.

**The caveat is the whole caveat**: this task *is* the function ROSA computes. A 100%-vs-13% gap on
a copy benchmark is evidence about the index, not about value on text.

Earlier attempts at this same question were misleading in both directions. `rosa-vs-attention-copy-recall`
(1200 steps) showed full+ROSA at 100% on 3/3 seeds against full attention at 19%, which reads as
"ROSA makes recall learnable that attention cannot learn". Rerunning the identical arm at **800
steps** gave 100% with **no ROSA at all** (and full+ROSA only 2/3). Twelve runs across arms landed
at ~19% or ~100% with nothing in between: the task is bistable and the LR schedule decides which
side an arm reaches. Any comparison that straddles that cliff measures the schedule.

## F4. The read gate learns to veto — but only when the decision is decidable

`rosa-gate-identifiable`. Harm control: targets re-assigned so that copying the retrieved token is
always wrong.

- With **no signal in the input** (identical context, different label), every ROSA arm sits at
  chance accuracy but at **loss 6.2–6.7 versus attention's 4.1–4.4** — a closed gate on top of a
  confidently wrong pathway. An earlier conclusion that "the gate does not veto" was an artifact of
  this unidentifiable control.
- Add a **flag token** immediately before the queries stating whether copying is correct, and the
  gate separates cleanly:

| arm | copy acc | gate on copy | harm acc | gate on harm |
|---|---|---|---|---|
| window + ROSA | **1.000** | **0.97–0.99** | 1.4–1.6% | **0.017–0.020** |
| attention + ROSA | **1.000** | 0.86–0.98 | 1.4–1.5% | 0.017–0.020 |
| full / window attention | 6.7–12.9% | — | 1.3–1.6% | — |

So the veto is learnable. Before judging any gate, check that the decision is decidable from the
input — otherwise you are measuring the control, not the gate.

## F5. The learned codebook is a utilisation problem, not a broken mechanism

`rosa-codebook-ksweep-orig` — the original kernel, K parameterised, 2500 steps, 192 distinct
tokens, chunked evaluation, oracle = 1.0:

| index | retrieval position accuracy (2 seeds) | codes actually used | task accuracy |
|---|---|---|---|
| raw token ids | **1.000 / 1.000** | 192 | 1.000 |
| K = 64 | 0.628 / 0.600 | 32–33 | 0.74–0.77 |
| K = 256 | 0.731 / 0.862 | 66–70 | 0.95–0.96 |
| K = 1024 | 0.936 / 0.933 | 114–119 | 0.992–0.994 |
| K = 4096 | 0.971 / 0.966 | 154–156 | 0.998–0.999 |

Monotone in K, saturating where codes-used approaches the 192 distinct tokens. At K = 4096 only 81%
of tokens get their own code, which is why retrieval plateaus at 0.97 rather than 1.0. Scaling to a
150k vocabulary therefore needs millions of codes — in practice: **index tokens, do not quantise
activations**.

Two earlier attempts at this measurement were void, both from this project's own errors: 600 steps
(the sweep at 600 steps put retrieval at chance for *every* K — undertraining, not a property of
the design) and a reimplementation with weights shared across layers (which reported 0.054/0.032 at
2500 steps, and was measuring the reimplementation). K = 4096 also OOMs on a T4 at batch 8 unless
evaluation is chunked.

An earlier, more damaging error sits behind this: measuring with a 64-symbol codebook quantised from
**hidden states** gave retrieval at chance (0.006–0.024 against chance 0.021) *while the module fired
on 70–99% of positions*, and for a while that was reported as "the discretization is broken". Hidden
states carry context: the same token in two positions quantises differently and never matches.
Quantise the **symbol** (the embedding), not the activation, and the same index retrieves at 0.93.

## F6. Real model: 85% of the gross gain is the pathway, not the retrieval

`rosa-killtest-prose-vs-code`, frozen Qwen2.5-0.5B, 512-token window, Python source corpus. All
arms bucketed by **one** common out-of-window position mask (25,765 in-window positions, 1,373
out-of-window):

| arm | ppl in-window | ppl out-of-window |
|---|---|---|
| no sidecar | 15.095 | 136.024 |
| exact retrieval (real) | **6.608** | **16.863** |
| scrambled-source control | 7.012 | 23.807 |
| constant-vector control | 6.849 | 21.552 |

- Gross effect: **8.07×** out-of-window, 2.28× in-window.
- Retrieval-specific effect (control ÷ real): **1.41×** out-of-window against the scrambled control,
  **1.28×** against the constant control, **1.04–1.06×** in-window.
- So ~85% of the 8× is a trainable residual pathway that any control arm reproduces.

That distinction was invisible in the first real-model run, which reported 136.0 → 16.8 as a
retrieval win. The bug that hid it: each arm was bucketed by the source *that arm used*, giving the
scrambled arm 4,665 "out" positions against 1,373 for the real arms — the control looked better than
the treatment because it was scored on a different set of positions.

Prose (`rosa-prose-verified`, Gutenberg, match mean 0.78 tokens, max 11): no sidecar 187.96/1279.81
(in/out), real 35.04/59.03, scrambled 36.71/65.14 → retrieval-specific **1.10×** out-of-window. On
ordinary prose, long matches barely exist and the mechanism is close to inert.

## F7. The definitive result: the gain shrinks with match length, then reverses

`rosa-definitive-match-length`. Second copy of the same 1024-token chunk, match at distance 1024
(twice the window), n = 30,690 scored positions, 97% with matches ≥32 tokens, 87.6% ≥128:

| arm | 1–8 (n=210) | 8–32 (720) | 32–128 (2,880) | 128–512 (11,520) | 512+ (15,360) | all |
|---|---|---|---|---|---|---|
| no sidecar | 617.79 | 97.38 | 29.23 | 18.02 | 16.41 | 19.18 |
| **real** | **173.61** | **37.61** | **14.80** | **11.61** | 17.27 | **15.18** |
| const | 331.42 | 58.88 | 20.78 | 14.97 | 14.78 | 16.18 |
| scrambled | 362.00 | 63.93 | 21.22 | 15.20 | 14.65 | 16.28 |

Retrieval-specific gain (const ÷ real), by band: **1.91× / 1.57× / 1.40× / 1.29× / 0.86×**;
aggregate **1.07×**.

The registered prediction was that the gain would *grow* with match length. It shrinks, and past
512 tokens it reverses into a **loss** — `real` (17.27) is worse than `const` (14.78) and worse than
no sidecar at all (16.41). The reasoning behind the prediction was wrong because it confounded two
variables: in this construction match length *is* position within the duplicated half, so the
long-match band is also the band where the model already has a full window of the copy to attend to.

The claim the data supports is bounded:

> Exact suffix retrieval is worth roughly 1.3–1.9× perplexity on tokens that have a long verbatim
> match **and** sit outside the attention window. It is neutral to harmful once the window already
> covers the material, and does nothing on text without long repeats.

On the no-match control (chunkA ++ chunkB, no useful retrieval) `real` 12.11 ≈ `const` 12.19 against
`none` 13.71 — i.e. the entire gain there is the pathway, as expected.

## F8. Unreliable matches are worse than no retrieval at all

`rosa-generalize-aggressive`. Take Python spans, rename **123 identifiers per span** consistently
throughout (semantics and structure preserved, surface destroyed), retrain nothing, evaluate the
sidecar trained on originals. Mean exact-match length drops 237 → 3.8 tokens.

| condition | ppl |
|---|---|
| no sidecar | 121.75 |
| exact index (K = 192) | **297.44** (2.44× worse) |
| coarse index (K = 256) | **269.89** (2.22× worse) |

A read gate trained where retrieval is always valid never learns to distrust it: it is calibrated to
the training distribution's match quality, not to validity. This is the mechanism that would produce
a long-context collapse of the ROSA-Tuning kind.

The first attempt at this construction was invalid: renaming one identifier per span left 97% of
tokens identical (token-set IoU 0.972, verbatim runs of 237 tokens intact), so "exact matching is
broken" was false — the test had not broken exact matching. The second attempt became invalid a
different way: because the renamed copy is nearly aligned, the true correspondence sits close to
`src = j − span`, so "within 4 tokens" partly measures proximity to the diagonal. Permute statement
order to move correspondence off-diagonal before claiming anything about coarse-symbol transfer.

## F9. The metadata verifier that was supposed to fix F8 is dead

The proposed fix: feed the gate match length, distance and a match-present flag, and train the
sidecar on 50/50 valid and invalid matches.

First run (`rosa-verifier-offdiag`, SPAN = WIN = 512): verifier 106.67 vs no-sidecar 116.69 vs
no-metadata ablation 125.13 on renamed text — read at the time as "harm converted into a small
gain". It was construction noise: with SPAN == WIN the duplicated half was reachable by the window
itself, so the "valid" control collapsed to ppl 2.72 and the whole comparison was degenerate.

Rerun with a non-degenerate control (`rosa-verifier-3seeds`, SPAN = 1024 > WIN = 512, 3 seeds,
means shown):

| condition (mean match) | none | verifier | **no-metadata ablation** |
|---|---|---|---|
| `dup`, valid (512.5 tok) | **13.58** | 12.99 | 13.37 |
| `renamed`, invalid (4.59) | 23.05 | 14.29 | **13.13** |
| `renamed_rot`, invalid, off-diagonal | 26.96 | 16.42 | **15.03** |

- The sidecar gives **no gain on valid matches** (13.58 → 12.99/13.37).
- The ablation beats the metadata arm on both invalid conditions in **all three seeds**.
- The gate never discriminates; it opens *more* on bad matches (0.118–0.133) than on good
  (0.104–0.112).
- The large gains on `renamed` (23 → 13) are the trainable pathway, available with no metadata at
  all (n=12 per arm per seed, so treat the exact ordering with care, but the direction is stable and
  the pre-registered criterion — gate separates valid from invalid — fails outright).

A verifier that does not separate the cases it exists to separate is not a verifier. Nothing in this
design makes loose matching safe, which remains the blocking limitation.

## F10. Accuracy levers: plain top-1 wins; only multi-layer injection pays

`rosa-accuracy-levers` — frozen Qwen2.5-0.5B, SPAN > WIN, 400 steps, PPL on the second half:

| condition | none | **top1** | top4 (multi-order mix) | top1 + length gate | top4 + length gate |
|---|---|---|---|---|---|
| `dup` (valid) | 13.58 | **12.61** | 14.07 | 21.74 | 37.59 |
| `renamed` (invalid) | 23.05 | **15.05** | 15.78 | 18.07 | 17.40 |
| `renamed_rot` (invalid) | 26.96 | **17.48** | 18.25 | 21.16 | 20.50 |

- Mixing candidates from different orders (PPM-style backoff, zero extra storage) is **worse**
  everywhere: the learned scorer does not beat "longest match, most recent", it adds noise.
- A hard "don't fire unless the best match is ≥ 8 tokens" rule is much worse still, because it
  changes what the sidecar sees during training and the model adapts to a starved channel.
- Retrieval-specific gain on valid matches stays ~1.08×. The sidecar still helps most on *invalid*
  conditions — the pathway signature again.

`rosa-aux-and-multilayer` — injecting at 3 layers instead of 1, plus an auxiliary head trained to
predict the next token from the mixed representation:

| condition | none | base (1 layer) | aux | multi (3 layers) | **aux+multi** |
|---|---|---|---|---|---|
| `dup` (valid) | 13.58 | 12.83 | 13.30 | 11.41 | **10.50** |
| `renamed` (invalid) | 23.05 | 14.56 | 14.57 | 11.59 | **10.90** |
| `renamed_rot` (invalid) | 26.96 | 16.79 | 16.42 | 13.19 | **12.64** |

Multi-layer injection is the genuinely cheap win: **~2.4M parameters, zero per-token memory, 1.12×
on valid matches**. The aux head buys a further 8% but costs **~410M parameters** (three
896→151,936 heads at 136M each, i.e. the `lora_params: 412,360,326` line in the log) — and the
version that reuses the model's own `lm_head` so aux adds zero parameters crashed on a dtype
mismatch (`rosa-content-vs-capacity`) and never produced a number. So the 8% is **confounded with
capacity** and should not be cited as a content effect until the free-head variant runs. If a
configuration had to ship today it would be `multi` without aux.

## F11. Fine-tuning dominates everything, and with a live sidecar the sidecar hurts

`rosa-lora-vs-frozen`, frozen vs LoRA (r=8, α=16) on layers 6/12/18, 400 steps, PPL:

| condition | LoRA | LoRA + sidecar (training *and* eval) | frozen + sidecar |
|---|---|---|---|
| `dup` (valid) | **8.507** | 10.759 | 11.573 |
| `renamed` (invalid) | **7.764** | 8.341 | 11.271 |
| `renamed_rot` (invalid) | **8.741** | 9.171 | 13.049 |
| `in_dist` (held-out code) | **6.915** | 8.632 | 8.200 |

- 400 LoRA steps take code perplexity from **19.03 → 6.92**. The sidecar's entire effect is a few
  percent on top of that: adaptation is ~10× larger than retrieval. Any claim that retrieval is the
  interesting lever has to reckon with this.
- With the sidecar live at evaluation, **LoRA + sidecar is worse than LoRA alone on all four
  conditions**. Injection costs ~5–25% where the model can already reach the material — the same
  sign as F7's 0.86× band, now at fine-tuning scale.
- The earlier run of this kernel (`results/rosa-lora-vs-frozen.out7.jsonl`) had hooks removed before
  evaluation, so it compared models *trained* with a sidecar against models *evaluated* without one:
  `frozen+side` reproduced the no-sidecar baseline exactly (13.58 / 23.05 / 26.96 / 19.03), which is
  the proof the hooks were dead. In that configuration the sidecar looked mildly *positive*
  (`dup` 8.218 vs 8.507, `in_dist` 6.652 vs 6.915) — i.e. training-time influence only. Both runs
  are included; the `.out8` run is the valid comparison.
- LoRA config and the 412M "lora" parameter log line are worth a caveat: the field counts all
  trainable parameters in the arm, and the aux heads dominate it. LoRA alone adds far fewer.

---

## What survives

1. **Index construction**: exact (400/400 vs brute force), linear (≤2n−1 states, ~1.18 states/token),
   **0.3–0.7 µs/token in C** once the recency stamp is replaced by a first-occurrence timestamp, flat
   in n including on constant runs.
2. **Memory**: ~10–40 bytes/token against ~12 KB/token KV cache — ~300–1000× less per token. Regime-
   independent, and the strongest true claim in the project.
3. **Retrieval-specific accuracy**: **1.3–1.9×** perplexity on positions with long verbatim matches
   outside the attention window (11.5k–15.4k positions per band), ~1.0–1.1× on prose, **0.86×** (a
   loss) where the window already covers the material.
4. **Synthetic copy/recall**: categorical, not marginal — 100% at every distance to 500 tokens on a
   task where full attention saturates at 13%, and a windowed-attention distance profile (25% inside,
   chance outside) that cleanly demonstrates what reach means.
5. **The gate veto works** when the decision is decidable from the input (0.98 vs 0.02).
6. **A control methodology**: gain must be decomposed as (trainable channel) × (content). A frozen
   backbone cannot tell a useful injection from a merely trainable one, and the difference is 5–8×.

## What is dead

- "ROSA makes recall learnable that attention cannot learn" — a phase transition that flips with
  step count (19% at 1200 steps vs 100% at 800 with no ROSA at all).
- "8.6× / 8.1× perplexity from exact retrieval" — 85% of it is a trainable pathway that the
  scrambled/constant controls reproduce.
- "The learned codebook retrieves at chance, so the discretization is broken" — that was quantising
  hidden states instead of symbols; quantise the symbol and retrieval goes 0.007 → 0.93.
- "The gain must grow with match length" — falsified; it shrinks and reverses.
- The metadata-conditioned verifier: beaten by its own no-metadata ablation on both invalid
  conditions in all three seeds, gate opens more on bad matches than good.
- Coarse-symbol transfer at K = 256 (~600 tokens/code): 30.7% position accuracy versus 47.8% for
  exact, further from correspondence overall. Plausible to retry at large K, not established.
- "Multi-order candidate mixing" (PPM-style top-k) as an accuracy lever: worse than plain top-1 in
  every condition measured here.

## Open

- **Does any of it survive real fine-tuning?** All real-model work used a frozen backbone, so a null
  is ambiguous. The single LoRA run says fine-tuning dominates by ~10:1, and the aux-head
  parameter-economics question (410M vs a free `lm_head` reuse) is unanswered because that kernel
  crashed.
- **Match length vs in-window coverage.** F7 falsified the naive prediction because the two moved
  together. The experiment that separates them — hold coverage fixed, vary only distance — has not
  been run.
- **Coarse codes as a generalisation mechanism.** Not tested off-diagonal at large K.
- **One unexplained anomaly**: 188 in-window perplexity for a 0.5B model on Gutenberg novels
  (`rosa-prose-verified`, no sidecar) is far from what a 0.5B model should score. The comparison
  within that run is internally consistent (same data, same mask, three arms), but the absolute
  number should not be cited until someone reproduces it.
- **Nothing here tests ROSA replacing attention at scale.** The closest result is window+ROSA vs
  full attention on a synthetic task.
