"""Two untried levers: teach the value, and inject at more than one layer.

base      : current design - one hook at layer 12, value = vproj(h[src]), learned gate.
aux       : same, plus an auxiliary head that must predict the NEXT TOKEN from the
            retrieved value.  Right now the only gradient into the value pathway is
            the frozen LM's loss many layers downstream, which is a weak signal; this
            forces the retrieved vector to carry the answer.  (This is the trick that
            made the codebook train.)
multilayer: inject at layers 6, 12 and 18 (separate small sidecars).  Tests whether a
            single hook point is the limitation.  Costs params, not per-token state.
aux+multi : both.

Conditions: dup (valid long matches) | renamed (invalid) | renamed_rot (invalid,
off-diagonal).  Memory accounting logged.
"""
import gc, glob, json, os, random, re
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, T, WIN, ROT = 1024, 2048, 512, 340
STEPS, NP, LAYERS_USED = 400, 80, [6, 12, 18]
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
    return torch.tensor(src, dtype=torch.long), torch.tensor(ml, dtype=torch.long)


class Side(nn.Module):
    def __init__(self, d, vocab=None, aux=False):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.aux = aux
        self.aux_head = nn.Linear(d, vocab) if aux else None
        self.last = {}

    def forward(self, h, src):
        hn = self.ln(h.float())
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        mixed = torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))
        rec = {"gate": g.detach()}
        if self.aux:
            rec["aux_val"] = self.aux_head(mixed)      # was the raw vector, not vocab logits
        self.last = rec
        return mixed.to(h.dtype)


def surface(txt, seed):
    txt = re.sub(r"#.*", "", txt)
    names = sorted(set(IDENT.findall(txt)))
    rnd = random.Random(seed); rnd.shuffle(names)
    mp = {n: "zq" + "".join(rnd.choice("abcdefghij") for _ in range(6)) + "x" for n in names}
    return re.sub(r"\s+", " ", IDENT.sub(lambda m: mp[m.group(0)], txt)).strip()


def spans(tok, want=NP):
    pats = ["/usr/local/lib/python3*/dist-packages/torch",
            "/usr/local/lib/python3*/site-packages/torch"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(p, "**", "*.py"), recursive=True)
    files = sorted({f for f in files if 20000 < os.path.getsize(f) < 300000})
    random.Random(0).shuffle(files)
    out = []
    for fi, f in enumerate(files):
        try:
            body = open(f, encoding="utf-8", errors="ignore").read()[:6000]
        except Exception:
            continue
        a = tok(body)["input_ids"]; b = tok(surface(body, fi))["input_ids"]
        if len(a) < SPAN or len(b) < SPAN:
            continue
        out.append((a[:SPAN], b[:SPAN]))
        if len(out) >= want:
            break
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log({"stage": "load", "model": MODEL, "span": SPAN, "win": WIN, "layers": LAYERS_USED})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d, V = model.config.hidden_size, model.config.vocab_size
    cfg = model.config
    kv = 2 * cfg.num_key_value_heads * (d // cfg.num_attention_heads) * 2 * cfg.num_hidden_layers
    log({"memory_Per_token": {"kv_fp16": kv, "index_est": 60, "ratio": round(kv / 60, 1),
                             "note": "extra layers/sidecars add parameters, not per-token state"}})

    pairs = spans(tok)
    log({"pairs": len(pairs)})
    if len(pairs) < 10:
        log({"abort": "too few spans"}); return
    conds = {"dup": [a + a for a, _ in pairs],
             "renamed": [a + b for a, b in pairs],
             "renamed_rot": [a + (b[ROT:] + b[:ROT]) for a, b in pairs]}

    def causal(n):
        ii = torch.arange(n)
        k = (ii[None, :] <= ii[:, None]) & ((ii[:, None] - ii[None, :]) < WIN)
        m = torch.zeros(1, 1, n, n, dtype=torch.float16, device=DEV)
        m.masked_fill_(~k.to(DEV), float("-inf")); return m
    masks = {SPAN: causal(SPAN), T: causal(T)}
    holder = {"src": None, "mods": None, "aux_out": None}

    def hook_factory(idx):
        def hook(mod, args, output):
            h = args[0]
            side = holder["mods"][idx]
            add = side(h, holder["src"].to(h.device))
            if side.aux:
                holder["aux_out"] = side.last["aux_val"]
            if isinstance(output, tuple):
                return (output[0] + add,) + tuple(output[1:])
            return output + add
        return hook

    def fwd(seq, src, n, toks=None):
        holder["src"] = src.unsqueeze(0)
        holder["aux_out"] = None
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=masks[n], use_cache=False).logits
        ce = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")
        return ce

    valid_sp = [p[0] for p in pairs[:-20]]
    invalid_sp = [p[1] for p in pairs[:-20]]
    ARMS = {"base": (False, [12]), "aux": (True, [12]),
            "multi": (False, LAYERS_USED), "auxmulti": (True, LAYERS_USED)}
    built = {}
    for arm, (use_aux, layers) in ARMS.items():
        torch.manual_seed(0)
        mods = {i: Side(d, V, use_aux).to(DEV) for i in layers}
        for i in layers:
            mods[i].train()
        holder["mods"] = mods
        handles = [model.model.layers[i].register_forward_hook(hook_factory(i)) for i in layers]
        params = [p for i in layers for p in mods[i].parameters()]
        opt = torch.optim.AdamW(params, lr=3e-3)
        rng = random.Random(1)
        for step in range(STEPS):
            seq = valid_sp[rng.randrange(len(valid_sp))] if rng.random() < 0.5 \
                else invalid_sp[rng.randrange(len(invalid_sp))]
            src, ml = build_src(seq)
            ce = fwd(seq, src, SPAN).mean()
            loss = ce
            if use_aux and holder["aux_out"] is not None:
                av = holder["aux_out"]
                tgt = torch.tensor([seq[1:]], device=DEV)
                loss = loss + 0.1 * F.cross_entropy(av[:, :-1].float().reshape(-1, V), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % 200 == 0:
                log({"arm": arm, "step": step, "loss": round(loss.item(), 4)})
        for h in handles:
            h.remove()
        for i in layers:
            mods[i].eval()
        built[arm] = (mods, layers)
        gc.collect(); torch.cuda.empty_cache()

    for cond, seqs in conds.items():
        for arm in ("none",) + tuple(ARMS):
            handles = []
            if arm != "none":
                mods, layers = built[arm]
                holder["mods"] = mods
                handles = [model.model.layers[i].register_forward_hook(hook_factory(i)) for i in layers]
            tot = [0.0, 0]
            with torch.no_grad():
                for s in seqs[-12:]:
                    src, ml = build_src(s)
                    for j, ls in enumerate(fwd(s, src, T).tolist()):
                        if j >= SPAN:
                            tot[0] += ls; tot[1] += 1
            for h in handles:
                h.remove()
            log({"cond": cond, "arm": arm,
                 "ppl": round(float(torch.tensor(tot[0] / max(tot[1], 1)).exp()), 3)})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-900:]})
