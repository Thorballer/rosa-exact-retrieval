# Raw results

One JSONL file per kernel run, exactly as downloaded from Kaggle: one JSON object per line, one line
per measurement. No post-processing — where a run failed, the failure is in the file as a `fatal` /
`error` object.

`<slug>.jsonl` is the first harvest of that kernel; `<slug>.outN.jsonl` are later harvests of the same
kernel after a re-push. **Later is not always better** — read the status column.

| file | status | note |
|---|---|---|
| `rosa-vs-attention-copy-recall.jsonl` | superseded | reported 100% full+ROSA vs 19% attention; the 800-step rerun (`rosa-controls-capacity-leak`) shows the task is bistable |
| `rosa-candidate-diagnostic.jsonl` | **void** | `candidate_acc` compares codebook-space token ids to vocab ids |
| `rosa-retrieval-position-diagnostic.jsonl` | partial | 1 line: run died on an unpacking bug |
| `rosa-retrieval-position-diagnostic.out2.jsonl` | valid | `retrieval_pos_acc` with a hidden-state codebook: chance-level (0.007) |
| `rosa-controls-capacity-leak.jsonl` | valid | 800 steps, capacity + harm controls; deflates the 100%-vs-19% claim |
| `rosa-long-context-distance.jsonl` | **void** | baseline never left chance → the design measured nothing |
| `rosa-symbolic-vs-attention.jsonl` | **void** | all ROSA arms died on a CUDA/CPU device mismatch |
| `rosa-symbolic-vs-attention.out2.jsonl` | valid | token-indexed ROSA: 100% at every distance, 3/3 seeds |
| `rosa-gate-identifiable.jsonl` | valid | flag-token experiment: gate 0.97–0.99 vs 0.017–0.020 |
| `rosa-real-lm-sidecar.jsonl` | partial | dtype failure in the sidecar |
| `rosa-real-lm-sidecar.out4.jsonl` | partial | hook contract failure |
| `rosa-real-lm-sidecar.out5.jsonl` | valid | first clean real-model result, bucketing since fixed |
| `rosa-codebook-token-vs-hidden.jsonl` | valid | raw ids vs hidden-state vs token-embedding codebook, 2500 steps |
| `rosa-killtest-prose-vs-code.jsonl` | valid | 4 arms on code + prose; the control that deflated the 8× |
| `rosa-definitive-match-length.jsonl` | **valid — the headline result** | duplicated chunk, 30,690 positions, bucketed by match length |
| `rosa-prose-verified.jsonl` | valid, one flagged absolute | verified Gutenberg corpus; 188 in-window ppl unexplained |
| `rosa-generalize-renamed.jsonl` | **void** | one rename per span left 97% of tokens identical, then crashed |
| `rosa-generalize-aggressive.jsonl` | valid (negative) | 123 renames per span; wrong-match injection 2.4× worse |
| `rosa-ksweep.jsonl` | **void** | 600 steps → retrieval at chance for every K (undertrained) |
| `rosa-ksweep-v2.jsonl` | **void** | reimplemented kernel with shared weights; drift, plus OOM at K=4096 |
| `rosa-verifier-offdiag.jsonl` | **degenerate** | `SPAN == WIN`, so the valid-match control collapsed to ppl 2.72 |
| `rosa-verifier-fixed-control.jsonl` | valid, 1 seed | fixed control; the verifier advantage did not reproduce |
| `rosa-verifier-3seeds.jsonl` | valid | 3 seeds; no-metadata ablation beats the metadata arm everywhere |
| `rosa-accuracy-levers.jsonl` | valid | top1 / top4 / length-gate comparison |
| `rosa-aux-and-multilayer.jsonl` | partial | failed on the aux-head reshape |
| `rosa-aux-and-multilayer.out2.jsonl` | valid | multi-layer vs aux vs aux+multi |
| `rosa-multi-content-control.jsonl` | **void** | OOM: three 408M-parameter aux heads alive at once |
| `rosa-codebook-ksweep-orig.jsonl` | **valid** | the K sweep that resolved the codebook question |
| `rosa-lora-vs-frozen.jsonl` | partial | v1 |
| `rosa-lora-vs-frozen.out2/.out3/.out4/.out5/.out6.jsonl` | **void** | successive plumbing failures (documented in `docs/experiments.md` #24) |
| `rosa-lora-vs-frozen.out7.jsonl` | valid but limited | sidecar hooks removed before eval: training-time influence only, `frozen+side` == `none` exactly |
| `rosa-lora-vs-frozen.out8.jsonl` | **valid** | sidecar live at eval: LoRA beats LoRA+sidecar on all four conditions |
| `rosa-content-vs-capacity.jsonl` | crashed | fp16/fp32 in the reused `lm_head`; the zero-parameter aux variant never ran |

## Fields you will see

- `in` / `out` — perplexity on positions inside / outside the 512-token attention window, each with
  `n` (position count). Compare only buckets with the same `n` across arms.
- `ppl` — perplexity on the second half of each sequence.
- `acc` — next-token accuracy (chance is stated in the same line as `chance_acc` when it matters).
- `retrieval_pos_acc` — fraction of queries where ROSA's chosen **source position** equals the true
  value's position. Positions, not tokens: this is the only cross-space-safe accuracy metric here.
- `matched_frac` / `fire_rate` — fraction of query positions where the module produced a candidate.
- `gate` / `gate_at_q` / `gate_valid` / `gate_invalid` — mean read-gate opening, either at query
  positions or split by whether the match was valid.
- `train_loss` / `train_loss_curve` — training loss, sampled at fixed step intervals.
  Prefer loss over `acc` on bistable tasks (see `docs/methodology.md` #1).
- `oracle_src_acc` / `oracle_idx_pos_acc` — sanity check that the ground-truth source position is
  recoverable at 1.000 before any training happens.
- `fatal` / `error` — the run died; the string is the exception, with a `trace` when available.
