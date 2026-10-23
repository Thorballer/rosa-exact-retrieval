"""Definitive test: when a long verbatim match sits OUTSIDE the window, what is
the retrieval-specific gain, as a function of match length?

Why this design.  In natural corpora long matches are rare (32 of 1,373 positions
in the last run), so the long-match cell had no statistical power.  Here the
sequences are built so long matches are guaranteed and abundant:

  dup      : seq = chunk(1024) ++ the SAME chunk(1024).  Every position in the
             second half has an exact match at distance 1024 -- twice the window,
             so attention cannot reach it -- and the match length grows from 1 to
             1024 across the half.  Real text, guaranteed long matches.
  nomatch  : seq = chunkA ++ chunkB (different documents).  No useful long match;
             measures whether the sidecar hurts when copying is useless.

Arms: none | real | const (fixed position 0) | scrambled.  All bucketed by the
TRUE match length on one common mask.

Definitive claim shape: gain(real over const) rises with match length, positive
only where a real match exists, and ~1.0 on nomatch.
"""
import glob, json, os, random
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
TRAIN_T, EVAL_T, WIN, LAYER = 1024, 2048, 512, 12
TRAIN_STEPS, NSEQ = 300, 30
BUCKETS = [(1, 8), (8, 32), (32, 128), (128, 512), (512, 4096)]


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    open(OUT, "a").write(line + "\n")


def build_src(x):
    n = len(x); s = 2 * n + 1
    b = [None] * s; c = [-1] * s; d = [0] * s; e = [-1] * s
    b[0] = {}; g = 0; z = 1
    src = [-1] * n; ml = [0] * n
    for i, t in enumerate(x):
        r = z; z += 1; b[r] = {}; d[r] = d[g] + 1; p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r; p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z; z += 1; b[u] = b[q].copy(); d[u] = d[p] + 1; c[u] = c[q]; e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u; p = c[p]
                c[q] = c[r] = u
        v = g = r
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                src[i] = e[v] + 1; ml[i] = d[v]; break
            v = c[v]
        v = g
        while v != -1 and e[v] < i:
            e[v] = i; v = c[v]
    return src, ml


def scramble(row):
    out = list(row)
    for j, s in enumerate(row):
        if s >= 0:
            out[j] = (int(s) * 7919 + j * 104729) % max(1, j)
    return out


def constant(row):
    return [0 if s >= 0 else -1 for s in row]


class Sidecar(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h, src):
        hn = self.ln(h)
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        return torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))


def code_chunks(tok, n=400):
    pats = ["/usr/local/lib/python3*/dist-packages/torch",
            "/usr/local/lib/python3*/site-packages/torch"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(p, "**", "*.py"), recursive=True)
    files = sorted({f for f in files if os.path.getsize(f) > 20000})
    random.Random(0).shuffle(files)
    out = []
    for f in files[:150]:
        try:
            ids = tok(open(f, encoding="utf-8", errors="ignore").read())["input_ids"]
        except Exception:
            continue
        for s in range(0, len(ids) - TRAIN_T, TRAIN_T):
            out.append(ids[s:s + TRAIN_T])
        if len(out) >= n:
            break
    return out[:n]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log({"stage": "load", "model": MODEL, "device": DEV, "eval_T": EVAL_T, "WIN": WIN})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size
    chunks = code_chunks(tok)
    log({"chunks": len(chunks), "decoded_sample": tok.decode(chunks[0][:30])})
    train = chunks[:-60]

    holder = {"src": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h.float(), holder["src"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def make_mask(n):
        idx = torch.arange(n)
        keep = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < WIN)
        m = torch.zeros(1, 1, n, n, dtype=torch.float16, device=DEV)
        m.masked_fill_(~keep.to(DEV), float("-inf"))
        return m

    masks = {TRAIN_T: make_mask(TRAIN_T), EVAL_T: make_mask(EVAL_T)}

    def forward_seq(seq, row, n):
        holder["src"] = torch.tensor([row], dtype=torch.long)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=masks[n], use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    built = {}
    for tag, fn in (("real", list), ("const", constant), ("scrambled", scramble)):
        sc = Sidecar(d).to(DEV); sc.train()
        holder["active"] = sc
        handle = model.model.layers[LAYER].register_forward_hook(hook)
        opt = torch.optim.AdamW(sc.parameters(), lr=3e-3)
        rng = random.Random(1)
        for step in range(TRAIN_STEPS):
            seq = train[rng.randrange(len(train))]
            loss = forward_seq(seq, fn(build_src(seq)[0]), TRAIN_T).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0); opt.step()
            if step % 150 == 0:
                log({"sidecar": tag, "step": step, "loss": round(loss.item(), 4)})
        handle.remove(); sc.eval(); built[tag] = sc
    torch.save({k: v.state_dict() for k, v in built.items()}, "/kaggle/working/definitive.pt")

    held = chunks[-60:]
    evalsets = {
        "dup": [(c + c, build_src(c + c)) for c in held[:NSEQ]],
        "nomatch": [(held[i] + held[i + NSEQ], build_src(held[i] + held[i + NSEQ]))
                    for i in range(NSEQ)],
    }
    for cond, rows in evalsets.items():
        dist = [m for _, (_, ml) in rows for m in ml[TRAIN_T:]]
        log({"cond": cond, "match_len_frac_ge32": round(sum(1 for m in dist if m >= 32) / max(1, len(dist)), 3),
             "match_len_frac_ge128": round(sum(1 for m in dist if m >= 128) / max(1, len(dist)), 3),
             "n_positions": len(dist)})

    for cond, rows in evalsets.items():
        arms = ["none", "real", "const"] + (["scrambled"] if cond == "dup" else [])
        for arm in arms:
            fn = {"none": list, "real": list, "const": constant, "scrambled": scramble}[arm]
            h = None
            if arm != "none":
                holder["active"] = built[arm]
                h = model.model.layers[LAYER].register_forward_hook(hook)
            acc = {f"{lo}-{hi}": [0.0, 0] for lo, hi in BUCKETS}
            acc["all"] = [0.0, 0]
            with torch.no_grad():
                for seq, (true_row, ml) in rows:
                    for j, ls in enumerate(forward_seq(seq, fn(true_row), EVAL_T).tolist()):
                        if j < TRAIN_T:          # score only the second half
                            continue
                        acc["all"][0] += ls; acc["all"][1] += 1
                        for lo, hi in BUCKETS:
                            if lo <= ml[j] < hi:
                                k = f"{lo}-{hi}"; acc[k][0] += ls; acc[k][1] += 1
                                break
            if h:
                h.remove()
            log({"cond": cond, "arm": arm,
                 **{k: {"ppl": round(float(torch.tensor(v[0] / max(v[1], 1)).exp()), 2), "n": v[1]}
                    for k, v in acc.items() if v[1]}})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-800:]})
