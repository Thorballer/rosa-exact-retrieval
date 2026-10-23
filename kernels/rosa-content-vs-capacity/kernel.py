"""Is the gain CONTENT or CAPACITY?  And can the aux head be free?

Arms share one architecture - three injection points (layers 6/12/18), each with a
value projection + learned gate - and differ ONLY in what the source index says:

  none      : no sidecar
  real      : src = longest previous match position
  const     : src = 0 always           (no content, identical capacity)
  scrambled : real positions, shuffled (instrumented, content-free)

The aux head reuses the model's own lm_head instead of creating three 896x151936
projections, so aux costs ZERO extra parameters.  Each arm is trained and evaluated
immediately, then freed, which is what OOM'd the previous version.

Conditions: dup (valid long matches) | renamed | renamed_rot | in_dist.
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
    """exact suffix-automaton retrieval: position of the longest previous match"""
    n = len(x); s = 2 * n + 1
    b = [None] * s; c = [-1] * s; d = [0] * s; e = [-1] * s
    b[0] = {}; g = 0; z = 1
    src = [-1] * n
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
                src[i] = e[v] + 1; break
            v = c[v]
        v = g
        while v != -1 and e[v] < i:
            e[v] = i; v = c[v]
    return torch.tensor(src, dtype=torch.long)


class Side(nn.Module):
    def __init__(self, d, lm_head=None):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.lm_head = lm_head              # reuse the model's own head: zero new params
        self.last = {}

    def forward(self, h, src):
        hn = self.ln(h.float())
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        mixed = torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))
        self.last = {"aux": self.lm_head(mixed) if self.lm_head is not None else None}
        return mixed.to(h.dtype)


def scramble(src, mode, rng):
    if mode == "real":
        return src
    if mode == "const":
        return torch.zeros_like(src)
    idx = (src >= 0).nonzero(as_tuple=True)[0]
    perm = idx[torch.randperm(len(idx), generator=rng)]
    out = src.clone(); out[idx] = src[perm]
    return out


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
    log({"stage": "load", "ver": "v4-lmhead-aux", "model": MODEL, "layers": LAYERS_USED})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size
    cfg = model.config
    kv = 2 * cfg.num_key_value_heads * (d // cfg.num_attention_heads) * 2 * cfg.num_hidden_layers
    log({"memory_per_token": {"kv_fp16": kv, "index_est": 60, "ratio": round(kv / 60, 1)},
         "sidecar_params_per_layer": 2 * d + d * d + d + d + 1,
         "aux_params": 0, "note": "aux reuses lm_head"})

    pairs = spans(tok)
    log({"pairs": len(pairs)})
    if len(pairs) < 10:
        log({"abort": "too few spans"}); return
    conds = {"dup": [a + a for a, _ in pairs],
             "renamed": [a + b for a, b in pairs],
             "renamed_rot": [a + (b[ROT:] + b[:ROT]) for a, b in pairs],
             "in_dist": [a for a, _ in pairs]}
    train = [p[0] for p in pairs[:-20]]
    invalid = [p[1] for p in pairs[:-20]]

    def causal(n):
        ii = torch.arange(n)
        k = (ii[None, :] <= ii[:, None]) & ((ii[:, None] - ii[None, :]) < WIN)
        m = torch.zeros(1, 1, n, n, dtype=torch.float16, device=DEV)
        m.masked_fill_(~k.to(DEV), float("-inf")); return m
    masks = {SPAN: causal(SPAN), T: causal(T)}
    holder = {"src": None, "sides": None, "aux": {}}

    def make_hook(which):
        def hook(mod, args, output):
            h = args[0]
            sc = holder["sides"][which]
            add = sc(h, holder["src"].to(h.device))
            holder["aux"][which] = sc.last["aux"]
            if isinstance(output, tuple):
                return (output[0] + add,) + tuple(output[1:])
            return output + add
        return hook

    def fwd(seq, src, n):
        holder["src"] = src.unsqueeze(0)
        holder["aux"] = {}
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=masks[n], use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    for mode in ("none", "real", "const", "scrambled"):
        torch.manual_seed(0)
        g_rng = torch.Generator().manual_seed(7)
        hooks = []
        if mode != "none":
            sides = {i: Side(d, model.lm_head).to(DEV) for i in LAYERS_USED}
            for i in LAYERS_USED:
                sides[i].train()
            holder["sides"] = sides
            hooks = [model.model.layers[i].register_forward_hook(make_hook(i)) for i in LAYERS_USED]
            params = [p for i in LAYERS_USED for p in sides[i].parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=3e-3)
            rng = random.Random(1)
            for step in range(STEPS):
                seq = train[rng.randrange(len(train))] if rng.random() < 0.5 \
                    else invalid[rng.randrange(len(invalid))]
                src = scramble(build_src(seq), mode, g_rng)
                loss = fwd(seq, src, SPAN).mean()
                av = holder["aux"].get(LAYERS_USED[-1])
                if av is not None:
                    loss = loss + 0.1 * F.cross_entropy(
                        av[:, :-1].float().reshape(-1, model.config.vocab_size),
                        torch.tensor(seq[1:], device=DEV).reshape(-1))
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                if step % 200 == 0:
                    log({"mode": mode, "step": step, "loss": round(loss.item(), 4)})
            for i in LAYERS_USED:
                sides[i].eval()
            log({"mode": mode, "trainable_params": sum(p.numel() for p in params)})
        for cond, seqs in conds.items():
            tot = [0.0, 0]
            with torch.no_grad():
                for s in seqs[-12:]:
                    src = build_src(s)
                    if mode != "none":
                        src = scramble(src, mode, g_rng)
                    n = T if len(s) > SPAN else SPAN
                    for j, ls in enumerate(fwd(s, src, n).tolist()):
                        if j >= (SPAN if len(s) > SPAN else 0):
                            tot[0] += ls; tot[1] += 1
            log({"mode": mode, "cond": cond,
                 "ppl": round(float(torch.tensor(tot[0] / max(tot[1], 1)).exp()), 3)})
        for hh in hooks:
            hh.remove()
        gc.collect(); torch.cuda.empty_cache()
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-900:]})
