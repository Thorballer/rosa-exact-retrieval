# Methodology: eight ways these experiments lied

This project produced roughly as many wrong answers as right ones, and every wrong answer was
plausible when it arrived. These are the specific traps, in the order they cost time. Each one is
cheap to avoid once named.

## 1. The LR schedule decides a bistable task, not the mechanism

The same arm, same seed, same task gave **19% at 1200 steps and 100% at 800**. Twelve runs across
four arms landed at ~19% or ~100% with nothing in between: associative recall at this size is
bistable, and the OneCycle schedule's total-step count moves the phase transition.

- Report **eval loss**, not argmax accuracy, whenever the task can be all-or-nothing.
- Pin the schedule, and require that the baseline arm demonstrably solves the task before drawing
  any conclusion about an added mechanism.
- "Arm A reached 100% and arm B didn't" is not a result unless the reverse ordering never occurs on
  a rerun with a different step count.

## 2. A module's token ids are not the task's token ids

`hard_rosa_predicted_tokens` lives in codebook space (vocab 64); the task's labels were vocab ids
0–31. Comparing them reported ~1% "candidate accuracy" that meant nothing in either direction.

- Score **positions** (`hard_rosa_source_index`) rather than tokens: positions are space-free.
- Validate the oracle first — `src position == true position` should be 1.000 before any training.

## 3. Quantise the symbol, not the activation

Indexing a learned codebook over **hidden states** gave retrieval accuracy 0.006–0.024 against 0.021
chance while the module fired on 70–99% of positions. The conclusion "the discretization is broken"
was wrong and cost a day.

Hidden states carry context, so the same token at two positions quantises to different codes and the
suffixes never match. Index the token embedding instead and retrieval goes to 0.93. Any "the codebook
doesn't work" result should be re-run with a symbol-space index before it is believed.

## 4. The control must have the pathway, not just the absence of the mechanism

A frozen LM plus **any** trained residual pathway improves perplexity 5–8×. Without a control that
keeps the pathway and removes the content, the pathway effect is indistinguishable from retrieval —
and this project reported it as retrieval twice (an 8.6× and an 8.1× "win", of which 85% and ~78%
respectively were the pathway).

The two controls that work:

- **scrambled source** — same sidecar, trained and evaluated on wrong positions;
- **constant vector** — same shapes, no content at all.

Report the ratio between those and the real arm, never the ratio to "no sidecar". Gain decomposes as
(trainable channel) × (content); measure the factors separately.

## 5. A degenerate control reads as a huge effect in whichever direction you like

With `SPAN == WIN == 512`, the "out of reach" copy was reachable by the attention window itself. The
`dup` baseline collapsed to perplexity 2.72 against 19.18 with `SPAN = 1024 > WIN = 512`, and a
verifier that looked like it had converted 2.4× harm into a small gain turned out to be noise.

- Always check the baseline number against a known-good run of the same construction.
- For any "beyond the window" claim, assert `SPAN > WIN` in the design, in the log, and in the paper.

## 6. Bucket every arm by ONE common position mask

Bucketing each arm by the source *that arm used* gave the scrambled arm 4,665 "out-of-window"
positions against 1,373 for the real arms, and made the control look better than the treatment.

Compute the comparison mask once from ground truth, reuse it for every arm, and log `n` per bucket so
a mismatched mask is visible.

## 7. Do not let the correct answer sit on the diagonal

Two constructions here were accidentally trivial:

- Renaming **one** identifier per span left 97% of tokens unchanged (token-set IoU 0.972) with
  verbatim runs of 237 tokens intact — so "exact matching is broken" was false and the test measured
  nothing.
- In a nearly-aligned copy, the true correspondence is close to `src = j − span`, so scoring
  "within 4 tokens" partly measures proximity to the diagonal rather than transfer.

Rename aggressively (123 identifiers per span → mean match 237 → 3.8 tokens) and permute statement
order to push correspondence off-diagonal before claiming generalisation.

## 8. Confounded variables describe the same thing twice

The prediction "retrieval gain must grow with match length" was falsified because in a
duplicated-document construction **match length is position**, and position is also how much
in-window context exists. Long matches and starved positions moved together, so the trend measured
the confound.

Separate them by holding in-window coverage fixed and varying only match distance. That experiment
was never run here, which is why the finding is stated as a band (1.3–1.9× outside the window) rather
than a law.

## Also worth recording

- **Natural data has no statistical power for this question.** In 1,373 naturally out-of-window
  positions only **32** had a match ≥16 tokens. Build sequences where the match is guaranteed and
  abundant (duplicate a chunk → 87% of 30,690 positions with matches ≥128), then the answer is
  quantitative instead of a coin flip.
- **Undertrained sweeps are void, not negative.** A K sweep at 600 steps put retrieval at chance for
  every K; the same sweep at 2500 steps gave 0.59/0.80/0.93. Sweep results below the training budget
  tell you about the budget.
- **Modify the original kernel; do not reimplement it.** A reimplementation with weights shared
  across layers produced 0.054 where the original produced 0.93. Minimum-diff edits to a known-good
  kernel are the only way a sweep number means anything.
- **A frozen backbone cannot license a claim about learning.** All real-model results here are frozen
  sidecar experiments; they measure whether an injection helps a frozen network, not whether a
  mechanism can be learned. State which one you measured.
- **Plumbing will beat you to the conclusion.** In this project the real failures were: fp16/fp32
  mismatches (three separate times), a hook returning a tuple into a layer that expects a bare
  tensor, a `hasattr(model, "base_model")` check that is true for every HF model, a gather indexing
  the candidate axis with position values (an out-of-bounds write the GPU reports several ops
  later), and evaluation that silently ran with the sidecar hooks removed. The last one is the
  dangerous kind: the numbers look sane and reproduce a known baseline, which is exactly how you
  detect it (if `frozen+side == none` exactly, the hooks are dead).
- **Log the units.** `retrieval_pos_acc`, `matched_frac`, `fire_rate`, `gate_at_q` — every number in
  this repo says what space it lives in. The ones that did not (candidate accuracy, unsupervised
  perplexity absolutes) are the ones that were wrong.
