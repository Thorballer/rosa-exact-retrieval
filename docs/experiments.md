# Experiment register

25 Kaggle kernels, in the order they were run. Each entry: what it tested, what came back, and —
for the ones that had to be thrown away — the specific reason. Void and degenerate runs are kept
because the failure modes are half the result; several of them initially looked like findings.

Kernels are under `kernels/<slug>/`, raw output under `results/<slug>[*].jsonl`. All runs were on a
single Kaggle T4 under a private account; `code/scripts/monitor.sh` is the poll-and-harvest wrapper.

Legend: **valid** = result used in `docs/findings.md`; **void** = measured nothing; **degenerate** =
measured something, but the control made the comparison meaningless; **crashed** = no result.

---

### 1. `rosa-vs-attention-copy-recall` — valid but superseded
Synthetic associative recall: `[k₁v₁…k₁₆v₁₆][perm(k)]`, 48 tokens, attention window 12 (paired value
always outside it), `d=96`, 2 layers, arms `full` / `window` / `win+rosa` / `full+rosa`, tasks
`copy` and `harm`, adaptive step count to a time budget.

Result (1200 steps): full 19.1%, window 13.1%, win+rosa 18.2%, **full+rosa 100% (3/3 seeds)**,
harm control at chance for every ROSA arm. This was reported as "ROSA makes recall learnable that
attention cannot learn".

Why it is superseded: at 800 steps the same arms gave full attention 100% on one seed and full+rosa
only 2/3 — the task is bistable and the OneCycle schedule decides which side an arm lands on.
`rosa-controls-capacity-leak` is the rerun that exposed this. Transferable output: the cost line
(162 s vs 12 s per 1200 steps, +62% params) and the harm-control behaviour.

### 2. `rosa-candidate-diagnostic` — void (wrong metric space)
Attempted to score ROSA's candidate token against the target token. The module's `predicted_tokens`
live in a 64-symbol codebook space while task labels are vocab ids 0–31, so the reported ~1%
"candidate accuracy" was comparing unrelated spaces. Fixed in #3 by scoring positions instead.

### 3. `rosa-retrieval-position-diagnostic` — valid, and decisive
Replaced the token comparison with `retrieval_pos_acc` (does ROSA's chosen *source position* equal
the true value's position — positions are space-free), plus `matched_frac` (does it fire at all).

Result with a hidden-state codebook: fires on 70–99% of positions, correct source position on
**0.6–2.4%** of queries against ~2.1% chance — including in the arm that scored 100% accuracy. That
is the measurement behind "the discretization, not the indexing, is the broken part" — later
overturned by #9 (quantise the symbol, not the activation).

Also the first honest negative on the long-context question: it showed the earlier gains came from
attention, not retrieval. Two earlier runs of this kernel errored on a one-line unpacking bug
(`sample()` started returning 4 values; one call site still took 3).

### 4. `rosa-controls-capacity-leak` — valid, and it deflates #1
800 steps, arms `full` / `fullwide` (parameter-matched) / `win+rosa` / `full+rosa`, tasks `copy`,
`shuffled`, `norepeat`, with an oracle per task first (copy 1.0, shuffled 0.045, norepeat 0.029).

Result: at 800 steps `full` reached 100% on one seed with **no ROSA at all**, and `full+rosa` only
2/3. Nothing about `shuffled` or `norepeat` separated the arms either. This is where the
"ROSA unlocks learnability" claim died. It also established the reporting convention used
throughout the rest of the project: report loss, not argmax accuracy, on bistable tasks.

### 5. `rosa-long-context-distance` — void (task not learnable)
Designed to measure the distance curve directly: train at length 384, evaluate at 4096, retrieval
distances 32 / 256 / 1024 / 2048 / 4000, window 256, no positional embeddings, 3 seeds.

Result: **every arm, every distance, `acc` 0–8% and loss 4.12–4.26 against uniform ln(64) = 4.159**;
training loss went 6.2 → 4.15 and flattened. The baseline never left chance, so there was no
collapse to observe. The design gave the task 16 supervised examples and 1000 steps — my failure,
not a fact about ROSA's reach. Nothing from this kernel is citable, and the open question ("does it
hold at 4k?") was answered later, differently, by #11.

### 6. `rosa-symbolic-vs-attention` — valid (after a device bug)
Abandoned the learned codebook entirely: index **raw token ids** with an exact suffix automaton, keep
a learned value projection and read gate, and measure by retrieval distance. 512-token context, 8
pairs, window 128, chance 1.56%.

First push failed on every ROSA arm with a CUDA/CPU gather mismatch the GPU only reported several ops
later (`Expected all tensors to be on the same device`). After the fix, 3 seeds per arm:

| arm | acc | <128 | 128–256 | 256–384 | 384+ |
|---|---|---|---|---|---|
| full | 13.3% | 13 | 13 | 13 | 14 |
| window | 12.9% | 25 | 24 | 2 | 1 |
| win+rosa | **100%** | 100 | 100 | 100 | 100 |
| full+rosa | **100%** | 100 | 100 | 100 | 100 |

loss 0.000 everywhere on the ROSA cells. Harm control: ROSA arms at chance accuracy but loss
6.17–6.68 against attention's 4.24–4.43 (confident wrongness with the gate closed).

### 7. `rosa-gate-identifiable` — valid, and it overturned a conclusion
Same task with a **flag token** immediately before the queries stating whether copying is correct,
so the gate's decision is decidable from the input.

Result: `win+rosa` copy accuracy **1.000** with gate 0.972–0.989, and on harm 1.4–1.6% accuracy with
gate 0.017–0.020. Both attention baselines sit at 6.7–12.9% copy accuracy. So the veto is learnable;
the earlier "the gate does not veto" was an artifact of an unidentifiable control.

### 8. `rosa-real-lm-sidecar` — valid (third attempt)
First real-model test: Qwen2.5-0.5B frozen, attention replaced by a 512-token window, ROSA sidecar
with learned value/gate, gradients isolated to the sidecar. Two failures first — a dtype mismatch
(sidecar fp32, frozen LM fp16) and a hook contract bug (this `transformers` version returns a bare
tensor from the decoder layer, so a tuple-returning hook got fed to the next layer as the hidden
state).

Valid run: ppl_in 15.095 → 6.632, ppl_out 136.024 → 16.785. Reported at the time as an 8× retrieval
win. The control (#10) breaks that.

### 9. `rosa-codebook-token-vs-hidden` — valid, and it resolved the codebook question in principle
Index source sweep at 2500 steps, 192 distinct tokens, oracle 1.0:

| index | retrieval pos acc | codes used |
|---|---|---|
| raw token ids | 1.000 | 192 |
| hidden-state codebook (hid:256) | **0.007–0.029** | 33 |
| token-embedding codebook, K=64 | 0.587–0.610 | 32 |
| K=256 | 0.743–0.856 | 67–69 |
| K=1024 | 0.925–0.936 | 112–117 |

Hidden states carry context, so the same token quantises differently at different positions and their
codes never match. Quantise the symbol instead and retrieval comes alive. (`rosa-codebook-ksweep-orig`,
#23, later re-ran this sweep properly on the original kernel.)

### 10. `rosa-killtest-prose-vs-code` — valid, and the control that deflated #8
Same harness, four arms (`none`, `real`, `scrambled`, `const`), one common position mask across arms,
code corpus. Result: 136.024 → 16.863 (real) but also → 23.807 (scrambled) and → 21.552 (const), i.e.
retrieval-specific 1.28–1.41× out-of-window and 1.04–1.06× in-window. ~85% of the gross gain is the
pathway.

The first attempt at the control bucketed each arm by the source *that arm used* (4,665 out-positions
for the scrambled arm against 1,373 for the real arms), which made the control look better than the
treatment. Fixed in `realmodel/out5` and reused everywhere after.

Prose arm: the first prose corpus was unverified (it scored 367.69 in-window with no sidecar, which
is not a plausible number for anything) so the corpus was rebuilt from Project Gutenberg with an
alphabetic-fraction check and a decoded sample — hence `rosa-prose-verified` (#12).

### 11. `rosa-definitive-match-length` — valid, and it falsified the registered prediction
Second copy of the same 1024-token chunk (match distance 1024 = twice the window), n = 30,690 scored
positions, 97% with matches ≥32 tokens. Four arms, perplexity bucketed by match length.

Result: retrieval-specific gain **1.91× / 1.57× / 1.40× / 1.29× / 0.86×** across the 1–8, 8–32,
32–128, 128–512 and 512+ bands; aggregate 1.07×. The gain shrinks where the prediction said it would
grow, and reverses to a loss where the window already covers the material. This is the result the
project is built on — see `docs/findings.md` F7 for the confound that produced the wrong prediction.

### 12. `rosa-prose-verified` — valid, with one unexplained absolute number
Gutenberg corpus (Moby Dick, Pride & Prejudice, Sherlock Holmes, Frankenstein, Dracula), alphabetic
fraction 0.955–0.977, verified decoded sample. Match-length probe: mean 0.78 tokens, max 11, 0% of
positions with a match ≥16.

Result: no sidecar 187.955 / 1279.813 (in/out), real 35.043 / 59.029, scrambled 36.710 / 65.139 →
retrieval-specific 1.10× out-of-window, 1.05× in-window. The 188 in-window absolute is far from what
a 0.5B model should score on novels and could not be reconciled; the within-run comparison is
internally consistent but the absolute value is flagged, not cited.

### 13. `rosa-generalize-renamed` — void (construction failed)
Goal: does the index generalise across surface changes? Took a 512-token Python span and renamed one
identifier consistently (`foo` → `zqoofx`).

Result: exact-match mean stayed at **237.21 tokens**, `frac_ge8: 0.965`, token-set IoU **0.972** — 97%
of tokens unchanged, so exact matching was never broken and the test measured nothing. Then it
crashed anyway on a half-precision layer-norm (`expected scalar type Half but found Float`).

### 14. `rosa-generalize-aggressive` — valid (negative)
Rerun with an honest construction: **123 identifier renames per span**, mean exact match 237 → 3.81
tokens.

Result: no sidecar 121.75, exact index **297.44** (2.44× worse), coarse index K=256 **269.89**
(2.22× worse). Wrong-match injection is worse than no retrieval — the mechanism behind a
long-context collapse. Secondary caveat recorded: the renamed copy is nearly aligned, so "within 4
tokens" partly measures proximity to the diagonal; permute statement order before claiming anything
about coarse-symbol transfer.

### 15. `rosa-ksweep` — void (undertrained)
K sweep at 600 steps. Retrieval position accuracy was at chance for every K (0.001–0.025 against
oracle 1.0), and K=4096 OOM'd on a T4 (6.25 GiB allocation for the code logits). Conclusion drawn at
the time — "codebook granularity doesn't help" — was wrong: the same sweep at 2500 steps gave
0.59/0.80/0.93 for K=64/256/1024.

### 16. `rosa-ksweep-v2` — void (reimplementation drift)
2500 steps, but reimplemented with weights shared across layers instead of the original's two blocks
per layer. Result: 0.054 (K=256) and 0.032 (K=1024), and K=4096 OOM'd again at evaluation (800
sequences × 4096 codes = 6.4 GB). Two attempts, two self-inflicted reasons to throw the number away —
which is why #23 uses the original kernel with a minimum diff.

### 17. `rosa-verifier-offdiag` — degenerate (SPAN == WIN)
First test of the metadata-conditioned verifier: gate sees hidden state + match length, distance,
match-present flag; sidecar trained 50/50 on valid (original spans) and invalid (surface-renamed)
matches. Result: renamed 116.69 (none) → **106.67** (verifier) → 125.13 (no-metadata ablation), read
as "harm converted into a gain".

Degenerate because SPAN = WIN = 512: the duplicated half was reachable by the window itself, so the
`dup` baseline collapsed to ppl 2.72 and the valid-match arm was not out of reach. Any comparison in
this run can be read as a large effect in either direction.

### 18. `rosa-verifier-fixed-control` — valid, single seed
Same design with SPAN = 1024 > WIN = 512 (rot 340, 80 pairs). Result: `dup` 12.32 → 12.90 (verifier)
/ 12.12 (ablation); `renamed` 22.48 → 14.27 / 15.11; `renamed_rot` 25.98 → 16.42 / 16.96. The gate
on valid-match tokens 0.129 vs invalid 0.146 — no separation. The #17 advantage did not reproduce.

### 19. `rosa-verifier-3seeds` — valid, and it kills the verifier
The same kernel wrapped in a seed loop (0, 1, 2), every row seed-tagged, 12 eval sequences per cell.
Means: `dup` 13.58 / 12.99 / 13.37; `renamed` 23.05 / 14.29 / **13.13**; `renamed_rot` 26.96 / 16.42 /
**15.03** (none / verifier / ablation). The ablation beats the metadata arm on both invalid conditions
in all three seeds, and the gate opens *more* on bad matches (0.118–0.133) than good (0.104–0.112).

### 20. `rosa-accuracy-levers` — valid (negative for two of three levers)
`top1` (single continuation), `top4` (multi-order candidate mixing, PPM-style, zero extra storage),
and length-gated variants. Result: top1 is best everywhere (`dup` 12.61 vs none 13.58), top4 is worse
(14.07), the ≥8-token length gate is much worse (21.74), and top4+gate is worse still (37.59). A hard
fire-rate restriction changes what the sidecar sees during training and the model adapts to a starved
channel.

One run of this kernel died on a gather that indexed the candidate axis with position values — an
out-of-bounds write the GPU only surfaced several ops later, which is why the traceback pointed at a
comparison rather than the gather.

### 21. `rosa-aux-and-multilayer` — valid (with a parameter caveat)
Three injection taps (layers 6/12/18) instead of one, plus an auxiliary next-token head on the mixed
representation. Result: `dup` 13.58 / 12.83 (base) / 13.30 (aux) / 11.41 (multi) / **10.50**
(aux+multi); same ordering on both invalid conditions. Multi-layer injection is a real, cheap gain
(~2.4M params); the aux variant adds ~410M params (three 896→151,936 heads) for a further 8%, so that
step is confounded with capacity.

First push failed: the aux head was fed the raw 896-dim mixed vector into a cross-entropy expecting
vocab logits (`reshape(-1, 151936)` on 916,608 elements).

### 22. `rosa-multi-content-control` — void (OOM)
The content-vs-capacity control for #21: identical capacity for `real` / `const` / `scrambled`. OOM'd
because all three arms' 408M-parameter aux heads were kept alive at once (12.94 GiB allocated of 14.56
GiB, then a 594 MiB allocation failed). Then the attempt to fix it by reusing the model's own `lm_head`
became #25.

### 23. `rosa-codebook-ksweep-orig` — valid, resolves the codebook question
The **original** codebook kernel with a minimum diff: K parameterised (64/256/1024/4096) and `score()`
chunked so K=4096 does not allocate 12.6 GB at evaluation. Everything else — two blocks per layer,
2500 steps, batch 8, aux weight 0.1, same data generator — untouched.

Result: 0.628/0.600 (K=64), 0.731/0.862 (256), 0.936/0.933 (1024), 0.971/0.966 (4096) retrieval
position accuracy over 2 seeds, with codes-used 32/68/116/155 out of 192. Monotone in K, saturating as
codes approach the distinct-token count → a code-utilisation problem, not a broken mechanism.

### 24. `rosa-lora-vs-frozen` — valid (fifth attempt)
Fine-tuning versus frozen, with a capacity-matched sidecar: LoRA r=8 α=16 on layers 6/12/18, 400
steps.

Four earlier attempts failed on: a stale `hasattr(model, "base_model")` check (True for every HF
model, so the helper reached for `.model` on a `Qwen2Model`); a gather with the wrong axis; an
undefined `hooks` name at cleanup; and fp32/fp16 disagreement in the LoRA wrapper's two matmuls
(`.to(dtype)` silently converted `A`/`B`, and the local harness caught what five Kaggle round-trips
did not).

Two valid runs, and both are kept:
- `results/rosa-lora-vs-frozen.out7.jsonl` — sidecar removed before evaluation, so it measures
  training-time influence only: `frozen+side` reproduces the no-sidecar baseline exactly
  (13.58 / 23.05 / 26.96 / 19.03), proving the hooks were dead; `lora+side` looks mildly positive
  (`dup` 8.218 vs 8.507).
- `results/rosa-lora-vs-frozen.out8.jsonl` — sidecar live at evaluation (the valid comparison):
  **LoRA alone beats LoRA+sidecar on all four conditions** (8.507 / 7.764 / 8.741 / 6.915 vs
  10.759 / 8.341 / 9.171 / 8.632), and fine-tuning beats everything by ~10:1 (19.03 → 6.92).

### 25. `rosa-content-vs-capacity` — crashed
The experiment that would have decided whether #21's aux gain was content or capacity: aux head
reusing the model's own `lm_head` (zero added parameters), each arm evaluated then freed, `real` vs
`const` vs `scrambled` at identical capacity. Died immediately on fp16/fp32 in the reused head
(`expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`). The `none` baselines
it logged (13.58 / 23.05 / 26.96 / 19.033) match every other run, so the harness was correct; the
aux variant simply never produced a number.

### 26. `rosa-vs-full-attention` (+ `-speed`) — valid, and it is the fair fight
Not a frozen-sidecar or windowed-attention test: the C suffix index against **full** causal attention,
same tokens, same positions, one common mask. `dup` = `[A][F][A]`, distance 1,024 / 3,072 / 16,896;
`dup-ctl` = the same local context with the earlier copy deleted; `plain` = no construction. Arms:
`rosa`, `full` (Qwen2.5-0.5B, 32k context), `win512` (same model, additive window mask).

Result (T4 fp16, 16 seq/corpus/cell, 8,176 positions): `dup` at 1,024 — rosa **100.0%**, full
**98.8%** (code), 99.0% (prose), win512 53.4% / 25.7%. At 3,072: 100.0 / 98.6 / 98.9. At 16,896 the
index is still 100.0% and the model arm cannot run (one attention matrix is 8.5 GiB/sequence on a
14.5 GiB card). Controls: `dup-ctl` rosa 34.9% (code) / 7.5% (prose), full 70.7% / 40.2%; `plain`
rosa 37.7% / 8.5%, full 73.9% / 42.5%. So retrieval-specific gain is rosa +58–93 pts against
attention +25–59 pts: the index wins on *how much* of its accuracy is retrieval, ties on accuracy, and
loses badly wherever the text does not literally repeat.

Cost: ROSA 0.39–0.73 µs/token on the T4 box and 0.08–0.24 µs/token on the laptop, versus 140 µs/token
(fp16, T=1k) to 752 µs/token (T=8k) for a forward pass; KV cache 12,288 B/token fp16 against 146–178
B/token of index. Per-token index cost tracks the fire rate, not n; memory is O(n) in the context and
stays flat in match length.

Three Kaggle OOMs before the harness was right (32×1536×151,936 fp32 logits = 29 GiB; 4-sequence chunk
at T=17,408 = 21 GiB; SDPA materialising T×T). The fix that stays: project only the scored hidden
states through `lm_head`. One local off-by-one (`match_len` dropping the current token, `src_is_true`
comparing to `i-dist` instead of `i-dist+1`) is corrected by `repost.py`, which re-derives the masks
and asserts that every stored prediction still equals the token at its own source. Local CPU run and
full T4 run both shipped.
