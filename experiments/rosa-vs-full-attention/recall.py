"""Recall head-to-head: ROSA (compressed suffix index) vs full attention, same positions.

Constructions
  dup       [A][F][A]   SPAN=512, FILLER tokens between copies → the copy sits `dist` back
  dup-ctl   [F][A]      identical local context, earlier copy deleted (leave-the-copy-out)
  plain     a straight 1536-token window of code / prose, no construction at all

Arms
  rosa      suffix-automaton prediction, C, mode 1 (first-occurrence tie-break)
  full      Qwen2.5-0.5B, full causal attention over the whole sequence
  win512    the same model, same positions, attention hard-masked to a 512-token window
            (an additive causal mask: j <= i and i - j < 512)

Metric: top-1 next-token accuracy at every scored position, bucketed by the match length of
the verbatim run ending there. One mask per construction, computed from the data, shared by
every arm (methodology.md #6).
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

MODEL = "Qwen/Qwen2.5-0.5B"
SPAN = 512
PLAIN_LEN = 1536


def rosa_arm(ids):
    """pred[i] = the token ROSA would copy for position i+1; lens[i] = its match length."""
    src = common.rosa_sources(ids, mode=1)
    pred, lens, fired, src_true = [], [], [], []
    for i, s in enumerate(src):
        if s < 0:
            pred.append(-1); lens.append(0); fired.append(False)
        else:
            pred.append(ids[s])
            lens.append(common.match_len(ids, i, s))
            fired.append(True)
        src_true.append(s)
    return pred, lens, fired, src_true


def window_mask(T, w, dtype):
    """Additive causal mask that additionally forbids attention past `w` tokens."""
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    ok = (j <= i) & (i - j < w)
    m = torch.zeros(T, T, dtype=dtype)
    m.masked_fill_(~ok, torch.finfo(dtype).min)
    return m[None, None]


def logits_of(model, batch, window=None):
    x = torch.tensor(batch)
    kw = {}
    if window is not None:
        kw["attention_mask"] = window_mask(x.shape[1], window, model.dtype)
    with torch.inference_mode():
        return model(x, use_cache=False, **kw).logits


def score_row(logits, j, seq, idxs):
    """logits[j, i] predicts seq[i+1] -- one sequence's positions at a time."""
    out = []
    for i in idxs:
        lg = logits[j, i]
        p = torch.softmax(lg, -1)
        tgt = seq[i + 1]
        out.append({"pred": int(lg.argmax()), "p_target": float(p[tgt]),
                    "rank_target": int((lg > lg[tgt]).sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseq", type=int, default=6)
    ap.add_argument("--filler", type=int, default=512)
    ap.add_argument("--corpora", default="code,prose")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    dist = SPAN + a.filler
    rows = []

    for corpus in a.corpora.split(","):
        need = a.nseq * max(2 * SPAN + a.filler, PLAIN_LEN) + 256
        toks = common.corpus_tokens(corpus, tok, need)
        if len(toks) < need:
            raise SystemExit(f"{corpus}: only {len(toks)} tokens, need {need}")

        # ---------------- duplicate construction -----------------------------
        step = 2 * SPAN + a.filler
        dup, ctl = [], []
        for k in range(a.nseq):
            off = k * step
            A, F = toks[off:off + SPAN], toks[off + SPAN:off + SPAN + a.filler]
            dup.append(A + F + A)
            ctl.append(F + A)
        idx_dup = list(range(dist, dist + SPAN - 1))       # second copy, minus the last token
        idx_ctl = list(range(a.filler, a.filler + SPAN - 1))
        mask_dup = {i: i - dist + 1 for i in idx_dup}      # match length == position in span
        mask_ctl = {i: 0 for i in idx_ctl}

        for j, seq in enumerate(dup):
            pred, lens, fired, src = rosa_arm(seq)
            for i in idx_dup:
                rows.append({"arm": "rosa:dup", "corpus": corpus, "seq": j, "pos": i,
                             "match_len": mask_dup[i], "target": seq[i + 1], "pred": pred[i],
                             "fired": fired[i], "rosa_match_len": lens[i],
                             "src_is_true": src[i] == i - dist + 1,
                             "dist_used": None if src[i] < 0 else i - src[i]})
            pred, lens, fired, _ = rosa_arm(ctl[j])
            for i in idx_ctl:
                rows.append({"arm": "rosa:dup-ctl", "corpus": corpus, "seq": j, "pos": i,
                             "match_len": 0, "target": ctl[j][i + 1], "pred": pred[i],
                             "fired": fired[i], "rosa_match_len": lens[i],
                             "src_is_true": None, "dist_used": None})

        for name, seqs, idxs, mask in (("dup", dup, idx_dup, mask_dup),
                                       ("dup-ctl", ctl, idx_ctl, mask_ctl)):
            lg = logits_of(model, seqs)
            for j, seq in enumerate(seqs):
                for i, r in zip(idxs, score_row(lg, j, seq, idxs)):
                    rows.append({"arm": f"full:{name}", "corpus": corpus, "seq": j, "pos": i,
                                 "match_len": mask[i], "target": seq[i + 1], **r})
        # window arm: the same sequences, attention hard-limited to the last SPAN tokens
        lg = logits_of(model, dup, window=SPAN)
        for j, seq in enumerate(dup):
            for i, r in zip(idx_dup, score_row(lg, j, seq, idx_dup)):
                rows.append({"arm": "win512:dup", "corpus": corpus, "seq": j, "pos": i,
                             "match_len": mask_dup[i], "target": seq[i + 1], **r})

        # ---------------- plain window (no construction) ---------------------
        plain = [toks[k * PLAIN_LEN:(k + 1) * PLAIN_LEN] for k in range(a.nseq)]
        idx_plain = list(range(512, PLAIN_LEN - 1))
        mask_plain = {}
        for j, seq in enumerate(plain):
            pred, lens, fired, src = rosa_arm(seq)
            for i in idx_plain:
                mask_plain.setdefault(j, {})[i] = lens[i]
                rows.append({"arm": "rosa:plain", "corpus": corpus, "seq": j, "pos": i,
                             "match_len": lens[i], "target": seq[i + 1], "pred": pred[i],
                             "fired": fired[i], "rosa_match_len": lens[i],
                             "src_is_true": None, "dist_used": None})
        lg = logits_of(model, plain)
        for j, seq in enumerate(plain):
            for i, r in zip(idx_plain, score_row(lg, j, seq, idx_plain)):
                rows.append({"arm": "full:plain", "corpus": corpus, "seq": j, "pos": i,
                             "match_len": mask_plain[j][i], "target": seq[i + 1], **r})
        print(f"{corpus}: {len(rows)} rows", flush=True)

    common.emit(common.HERE / "results" / f"recall-{a.tag}.jsonl", rows)


if __name__ == "__main__":
    main()
