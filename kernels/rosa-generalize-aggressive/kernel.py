"""Can coarse symbols transfer structure when exact tokens do not?

Surface change is now AGGRESSIVE: every identifier renamed consistently, comments
stripped, whitespace collapsed.  Long verbatim runs should collapse, so the exact
index has almost nothing, while statement structure and semantics are preserved
(consistent renaming).

Primary metric is retrieval POSITION accuracy against a true alignment map
(tolerance buckets), because a coarse index that never lands on the structurally
corresponding place cannot help PPL and would only add noise.

Arms: none | exact (token ids) | coarse (K learned codes, trained by next-token).
"""
import glob, json, os, random, re
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, T, WIN, LAYER = 512, 1024, 512, 12
STEPS, K, NP = 300, 256, 120
IDENT = re.compile(r"\b[a-z_][a-z0-9_]{1,}\b")


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


def surface_change(txt, seed):
    """strip comments, collapse whitespace, rename EVERY identifier consistently"""
    txt = re.sub(r"#.*", "", txt)
    names = sorted(set(IDENT.findall(txt)))
    rnd = random.Random(seed)
    rnd.shuffle(names)
    mapping = {n: "zq" + "".join(rnd.choice("abcdefghij") for _ in range(6)) + "x" for n in names}
    txt = IDENT.sub(lambda m: mapping[m.group(0)], txt)
    return re.sub(r"\s+", " ", txt).strip(), len(names)


def align(a, b, lookahead=12):
    """map each index in b to the index in a it corresponds to (greedy), or -1"""
    out = [-1] * len(b)
    i = 0
    for j, tok in enumerate(b):
        if i < len(a) and a[i] == tok:
            out[j] = i; i += 1; continue
        k = None
        for d in range(1, lookahead):
            if i + d < len(a) and a[i + d] == tok:
                k = i + d; break
        if k is not None:
            out[j] = k; i = k + 1
    return out


class Sidecar(nn.Module):
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

    def codes(self, ids, emb):
        x = self.ln(emb.float())                     # fp16 embeddings -> fp32 sidecar
        logits = self.proj(x)
        soft = F.softmax(logits, dim=-1) @ self.code_emb.weight
        aux = F.cross_entropy(self.aux(soft)[:, :-1].reshape(-1, self.aux.out_features),
                              ids[:, 1:].reshape(-1))
        return logits.argmax(-1), aux

    def forward(self, h, src):
        hn = self.ln(h.float())
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        return torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))


def make_pairs(tok, want=NP):
    pats = ["/usr/local/lib/python3*/dist-packages/torch",
            "/usr/local/lib/python3*/site-packages/torch"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(p, "**", "*.py"), recursive=True)
    files = sorted({f for f in files if 20000 < os.path.getsize(f) < 300000})
    random.Random(0).shuffle(files)
    pairs = []
    for fi, f in enumerate(files):
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        body = txt[:4000]
        new, nren = surface_change(body, fi)
        a = tok(body)["input_ids"]
        b = tok(new)["input_ids"]
        if len(a) < SPAN or len(b) < SPAN:
            continue
        pairs.append((a[:SPAN], b[:SPAN], nren))
        if len(pairs) >= want:
            break
    return pairs


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log({"stage": "load", "model": MODEL, "device": DEV, "K": K, "SPAN": SPAN})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d, V = model.config.hidden_size, model.config.vocab_size
    pairs = make_pairs(tok)
    log({"pairs": len(pairs), "mean_renames": round(sum(p[2] for p in pairs) / max(1, len(pairs)), 1)})
    if not pairs:
        log({"abort": "no pairs"}); return

    data = []
    ml_stats = []
    for a, b, _ in pairs:
        s = a + b
        src, ml = build_src(s)
        amap = align(a, b)
        data.append((s, src, ml, amap))
        ml_stats += [m for m in ml[SPAN:]]
    log({"exact_match_in_renamed_half": {
        "mean": round(sum(ml_stats) / max(1, len(ml_stats)), 2),
        "max": max(ml_stats) if ml_stats else 0,
        "frac_ge8": round(sum(1 for m in ml_stats if m >= 8) / max(1, len(ml_stats)), 3)}})

    idx = torch.arange(T); tr = torch.arange(SPAN)
    def causal(n):
        ii = torch.arange(n)
        k = (ii[None, :] <= ii[:, None]) & ((ii[:, None] - ii[None, :]) < WIN)
        m = torch.zeros(1, 1, n, n, dtype=torch.float16, device=DEV)
        m.masked_fill_(~k.to(DEV), float("-inf")); return m
    masks = {SPAN: causal(SPAN), T: causal(T)}
    holder = {"src": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h, holder["src"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def fwd(seq, row, n):
        holder["src"] = torch.tensor([row], dtype=torch.long)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=masks[n], use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    train = [p[0] for p in pairs[:-20]]
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
                code_ids, aux = sc.codes(ids, model.model.embed_tokens(ids))
                row = build_src(code_ids[0].tolist())[0]
            else:
                aux, row = None, build_src(seq)[0]
            loss = fwd(seq, row, SPAN).mean()
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
        ppl = [0.0, 0]; tol = {"within4": 0, "within16": 0, "within64": 0, "beyond": 0}
        with torch.no_grad():
            for s, src, ml, amap in data:
                if arm == "coarse":
                    ids = torch.tensor([s], device=DEV)
                    code_ids, _ = built["coarse"].codes(ids, model.model.embed_tokens(ids))
                    use_row = build_src(code_ids[0].tolist())[0]
                else:
                    use_row = src
                for j, ls in enumerate(fwd(s, use_row, T).tolist()):
                    if j < SPAN:
                        continue
                    ppl[0] += ls; ppl[1] += 1
                    tgt = amap[j - SPAN]
                    s_pos = use_row[j]
                    if arm != "none" and tgt is not None and tgt >= 0 and s_pos >= 0:
                        dlt = abs(s_pos - tgt)
                        tol["within4" if dlt <= 4 else "within16" if dlt <= 16
                            else "within64" if dlt <= 64 else "beyond"] += 1
        if h:
            h.remove()
        tot = sum(tol.values())
        log({"cond": "renamed_aggressive", "arm": arm,
             "ppl": round(float(torch.tensor(ppl[0] / max(ppl[1], 1)).exp()), 2),
             "pos_acc": ({k: round(v / tot, 3) for k, v in tol.items()} if tot else None)})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-900:]})
