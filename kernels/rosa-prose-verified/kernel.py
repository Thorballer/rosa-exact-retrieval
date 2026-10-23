"""Prose corpora, verified before use.  The point is the corpus, not the model.

Fetches Gutenberg plain text, logs enough about each book that garbage is obvious
(bytes, alphabetic fraction, decoded sample, mean/max verbatim match length), and
only then runs none / real / scrambled.

If the text is mangled, every downstream number is meaningless, so the checks run
before any GPU work.
"""
import json, re, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
T, WIN, LAYER, HOLDOUT, STEPS = 1024, 512, 12, 40, 300
BOOKS = ["https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
         "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
         "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
         "https://www.gutenberg.org/cache/epub/84/pg84.txt",
         "https://www.gutenberg.org/cache/epub/98/pg98.txt"]


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


class Sidecar(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.last = {}

    def forward(self, h, src):
        hn = self.ln(h)
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        out = torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))
        self.last = {"gate": g.detach()}
        return out


def fetch_books():
    texts = []
    for u in BOOKS:
        try:
            raw = urllib.request.urlopen(u, timeout=60).read()
            txt = raw.decode("utf-8", errors="replace")
            alpha = sum(ch.isalpha() or ch.isspace() for ch in txt[:100000]) / 100000
            words = len(re.findall(r"[A-Za-z]{2,}", txt[:50000]))
            log({"book": u.rsplit("/", 1)[-1], "bytes": len(raw), "alpha_frac": round(alpha, 3),
                 "words_in_first_50k": words, "sample": txt[200:320].replace("\n", " ")})
            if alpha > 0.75 and txt.count("the ") > 100:
                texts.append(txt)
            else:
                log({"rejected": u.rsplit("/", 1)[-1], "reason": "looks like markup/garbage"})
        except Exception as e:
            log({"book": u.rsplit("/", 1)[-1], "error": repr(e)[:150]})
    return texts


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    texts = fetch_books()
    log({"books_accepted": len(texts)})
    if len(texts) < 2:
        log({"abort": "corpus unusable"}); return

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size

    chunks = []
    for txt in texts:
        ids = tok(txt)["input_ids"]
        for s in range(0, len(ids) - T, T):
            chunks.append(ids[s:s + T])
    log({"chunks": len(chunks), "decoded_sample": tok.decode(chunks[0][:40])})

    tr_src = [build_src(c)[1] for c in chunks[:20]]
    flat = [m for row in tr_src for m in row]
    log({"match_len_probe": {"mean": round(sum(flat) / max(1, len(flat)), 2),
                             "max": max(flat) if flat else 0,
                             "frac_ge16": round(sum(1 for m in flat if m >= 16) / max(1, len(flat)), 4)}})

    idx = torch.arange(T)
    keep = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < WIN)
    mask = torch.zeros(1, 1, T, T, dtype=torch.float16, device=DEV)
    mask.masked_fill_(~keep.to(DEV), float("-inf"))
    holder = {"src": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h.float(), holder["src"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def forward_seq(seq, row):
        holder["src"] = torch.tensor([row], dtype=torch.long)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    train, test = chunks[:-HOLDOUT], chunks[-HOLDOUT:]
    evalset = [(s, build_src(s)) for s in test]
    built = {}
    for tag, fn in (("real", list), ("scrambled", scramble)):
        sc = Sidecar(d).to(DEV); sc.train()
        holder["active"] = sc
        handle = model.model.layers[LAYER].register_forward_hook(hook)
        opt = torch.optim.AdamW(sc.parameters(), lr=3e-3)
        import random as _r
        rng = _r.Random(1)
        for step in range(STEPS):
            seq = train[rng.randrange(len(train))]
            loss = forward_seq(seq, fn(build_src(seq)[0])).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0); opt.step()
            if step % 150 == 0:
                log({"sidecar": tag, "step": step, "loss": round(loss.item(), 4)})
        handle.remove()
        sc.eval(); built[tag] = sc

    for arm, sc in [("none", None)] + [(t, built[t]) for t in built]:
        h = None
        if sc is not None:
            holder["active"] = sc
            h = model.model.layers[LAYER].register_forward_hook(hook)
        fn = scramble if arm == "scrambled" else list
        acc = {"in": [0.0, 0], "out": [0.0, 0], "out_long": [0.0, 0], "out_short": [0.0, 0]}
        with torch.no_grad():
            for seq, (true_row, ml) in evalset:
                for j, ls in enumerate(forward_seq(seq, fn(true_row)).tolist()):
                    s = true_row[j]
                    if s < 0:
                        continue
                    if (j - s) <= WIN:
                        acc["in"][0] += ls; acc["in"][1] += 1
                    else:
                        acc["out"][0] += ls; acc["out"][1] += 1
                        k = "out_long" if ml[j] >= 16 else "out_short"
                        acc[k][0] += ls; acc[k][1] += 1
        if h:
            h.remove()
        log({"corpus": "prose_verified", "arm": arm,
             **{k: {"ppl": round(float(torch.tensor(v[0] / max(v[1], 1)).exp()), 3), "n": v[1]}
                for k, v in acc.items() if v[1]}})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-800:]})
