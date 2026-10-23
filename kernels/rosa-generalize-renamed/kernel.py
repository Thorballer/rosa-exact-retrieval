"""Can coarse symbols generalize where exact tokens cannot?

Construction.  Take a 512-token Python span, then make a copy in which one or more
identifiers are renamed CONSISTENTLY (so semantics and structure are identical but
the tokens differ).  Sequence = span ++ renamed_span (1024 tokens, window 512).
The renamed copy sits outside the window, and its exact token match no longer
exists -- but its structural pattern does.

Arms:
  none    : no retrieval
  exact   : index over raw token ids           (cannot match a renamed span)
  coarse  : index over a learned codebook of K codes, trained by next-token
            prediction so tokens with similar successor distributions merge
            (morphological / syntactic equivalence class instead of identity)

If coarse > exact on renamed spans, coarser symbols buy generalization that
exactness cannot.  If coarse ~= exact ~= none, the "fuzzy symbols" route is dead
and only embedding-level retrieval (kNN-LM) can cross the paraphrase gap.
"""
import glob, json, os, random, re
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, T, WIN, LAYER = 512, 1024, 512, 12
STEPS, NSEQ, K = 300, 30, 256
IDENT = re.compile(r"\b([a-z_][a-z0-9_]{2,})\b")


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


class Sidecar(nn.Module):
    """exact: index = token ids.  coarse: index = argmax of a learned projection."""

    def __init__(self, d, k=None, vocab=None):
        super().__init__()
        self.k = k
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))
        if k:
            self.proj = nn.Linear(d, k)
            self.code_emb = nn.Embedding(k, d)
            self.aux = nn.Linear(d, vocab)
        self.last = {}

    def codes(self, h, toks, emb):
        if not self.k:
            return toks, None
        logits = self.proj(self.ln(emb))
        soft = F.softmax(logits, dim=-1) @ self.code_emb.weight
        aux = F.cross_entropy(self.aux(soft)[:, :-1].reshape(-1, self.aux.out_features),
                              toks[:, 1:].reshape(-1))
        return logits.argmax(-1), aux

    def forward(self, h, src):
        hn = self.ln(h)
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        return torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))


def spans(tok, want=200):
    pats = ["/usr/local/lib/python3*/dist-packages/torch",
            "/usr/local/lib/python3*/site-packages/torch"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(p, "**", "*.py"), recursive=True)
    files = sorted({f for f in files if 20000 < os.path.getsize(f) < 400000})
    random.Random(0).shuffle(files)
    out = []
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        names = [n for n in set(IDENT.findall(txt)) if len(n) > 4]
        if not names:
            continue
        for _ in range(6):
            old = random.Random(len(out) + hash(f) % 1000).choice(names)
            new = "zq" + old[::-1][:8] + "x"            # a name that never occurs
            if new in txt:
                continue
            renamed = re.sub(rf"\b{re.escape(old)}\b", new, txt)
            a = tok(txt)["input_ids"]
            b = tok(renamed)["input_ids"]
            if len(a) < SPAN + 10 or len(b) < SPAN + 10:
                continue
            off = random.Random(len(out)).randrange(0, min(len(a), len(b)) - SPAN)
            out.append((a[off:off + SPAN], b[off:off + SPAN], old, new))
        if len(out) >= want:
            break
    return out[:want]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log({"stage": "load", "model": MODEL, "device": DEV, "K": K, "span": SPAN})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size
    V = model.config.vocab_size
    pairs = spans(tok)
    log({"pairs": len(pairs), "example_rename": [pairs[0][2], pairs[0][3]] if pairs else None})

    seqs, exact_ml, iou = [], [], []
    for a, b, _, _ in pairs:
        s = a + b
        src, ml = build_src(s)
        seqs.append((s, src, ml))
        exact_ml.append([m for m in ml[SPAN:]])
        iou.append(len(set(a) & set(b)) / max(1, len(set(a) | set(b))))
    flat = [m for row in exact_ml for m in row]
    log({"eval": {"seqs": len(seqs), "exact_match_mean": round(sum(flat) / max(1, len(flat)), 2),
                  "exact_match_max": max(flat) if flat else 0,
                  "frac_ge8": round(sum(1 for m in flat if m >= 8) / max(1, len(flat)), 3),
                  "token_set_iou_span_vs_renamed": round(sum(iou) / max(1, len(iou)), 3)}})
    if not seqs:
        log({"abort": "no pairs"}); return

    idx = torch.arange(T)
    keep = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < WIN)
    mask = torch.zeros(1, 1, T, T, dtype=torch.float16, device=DEV)
    mask.masked_fill_(~keep.to(DEV), float("-inf"))
    tridx = torch.arange(SPAN)
    keep_t = (tridx[None, :] <= tridx[:, None]) & ((tridx[:, None] - tridx[None, :]) < WIN)
    mask_t = torch.zeros(1, 1, SPAN, SPAN, dtype=torch.float16, device=DEV)
    mask_t.masked_fill_(~keep_t.to(DEV), float("-inf"))

    holder = {"src": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h.float(), holder["src"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def fwd(seq, row, n):
        holder["src"] = torch.tensor([row], dtype=torch.long)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=mask if n == T else mask_t,
                       use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    train = [p[0] for p in pairs[:-NSEQ]]
    built = {}
    for tag, k in (("exact", None), ("coarse", K)):
        sc = Sidecar(d, k, V).to(DEV); sc.train()
        holder["active"] = sc
        handle = model.model.layers[LAYER].register_forward_hook(hook)
        opt = torch.optim.AdamW(sc.parameters(), lr=3e-3)
        rng = random.Random(1)
        for step in range(STEPS):
            seq = train[rng.randrange(len(train))]
            ids = torch.tensor([seq], device=DEV)
            if k:
                embs = model.model.embed_tokens(ids)
                code_ids, aux = sc.codes(None, ids, embs)
                use_row = build_src(code_ids[0].tolist())[0]     # codes -> source positions
            else:
                aux = None
                use_row = build_src(seq)[0]
            loss = fwd(seq, use_row, SPAN).mean()
            if aux is not None:
                loss = loss + 0.1 * aux
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0); opt.step()
            if step % 150 == 0:
                log({"sidecar": tag, "step": step, "loss": round(loss.item(), 4)})
        handle.remove(); sc.eval(); built[tag] = sc

    for arm in ("none", "exact", "coarse"):
        h = None
        if arm != "none":
            holder["active"] = built[arm]
            h = model.model.layers[LAYER].register_forward_hook(hook)
        acc = {"all": [0.0, 0], "ml_lt8": [0.0, 0], "ml_ge8": [0.0, 0]}
        with torch.no_grad():
            for s, src, ml in seqs:
                if arm == "coarse":
                    ids = torch.tensor([s], device=DEV)
                    embs = model.model.embed_tokens(ids)
                    code_ids, _ = built["coarse"].codes(None, ids, embs)
                    use_row = build_src(code_ids[0].tolist())[0]   # codes -> source positions
                else:
                    use_row = src
                for j, ls in enumerate(fwd(s, use_row, T).tolist()):
                    if j < SPAN:
                        continue
                    acc["all"][0] += ls; acc["all"][1] += 1
                    k = "ml_ge8" if ml[j] >= 8 else "ml_lt8"
                    acc[k][0] += ls; acc[k][1] += 1
        if h:
            h.remove()
        log({"cond": "renamed", "arm": arm,
             **{k: {"ppl": round(float(torch.tensor(v[0] / max(v[1], 1)).exp()), 2), "n": v[1]}
                for k, v in acc.items() if v[1]}})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-900:]})
