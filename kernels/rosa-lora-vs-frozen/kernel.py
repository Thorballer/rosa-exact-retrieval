"""Does the model use retrieval better when it is allowed to adapt?

Every real-model result so far used a FROZEN backbone, where a new memory channel
can only act as a domain adapter - the model has no chance to learn when or how to
consume it, and cannot learn to distrust it.  That is a plausible explanation for
why retrieval-specific gains kept coming out small and inconsistent.

Arms (identical data, identical budget):
  lora       : LoRA fine-tune only (rank 8 on q/k/v/o), no retrieval
  lora+side  : same LoRA + the exact retriever at layer 12 (metadata-free gate)
  frozen+side: the previous setting, for reference

Train: plain next-token on Python spans (SPAN=1024, no duplication - realistic).
Eval : dup (valid long matches) | renamed (invalid) | renamed_rot (invalid,
off-diagonal) | plus in-distribution held-out code.
"""
import gc, glob, json, os, random, re
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, T, WIN, ROT, LAYER = 1024, 2048, 512, 340, 12
STEPS, NP = 400, 80
LAYERS_USED = [6, 12, 18]
IDENT = re.compile(r"\b[a-z_][a-z0-9_]{1,}\b")


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    open(OUT, "a").write(line + "\n")


def build_src(x):
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
    def __init__(self, d, vocab=None):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.aux_head = nn.Linear(d, vocab) if vocab else None
        self.last = {}

    def forward(self, h, src):
        hn = self.ln(h.float())
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        mixed = torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))
        self.last = {"aux": self.aux_head(mixed) if self.aux_head is not None else None}
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


class LoRALinear(nn.Module):
    """minimal LoRA: base stays frozen, low-rank update is trained"""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        w = base.weight
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=w.device, dtype=torch.float32) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=w.device, dtype=torch.float32))
        self.scale = alpha / r

    def forward(self, x):
        out = self.base(x)
        upd = ((x.float() @ self.A.float().T) @ self.B.float().T).to(out.dtype)  # dtype-safe
        return out + upd * self.scale


def layers_of(m):
    inner = getattr(m, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError("cannot locate transformer layers on " + type(m).__name__)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log({"stage": "load", "model": MODEL, "steps": STEPS, "layers": LAYERS_USED})
    tok = AutoTokenizer.from_pretrained(MODEL)
    pairs = spans(tok)
    log({"pairs": len(pairs)})
    if len(pairs) < 10:
        log({"abort": "too few spans"}); return
    conds = {"dup": [a + a for a, _ in pairs],
             "renamed": [a + b for a, b in pairs],
             "renamed_rot": [a + (b[ROT:] + b[:ROT]) for a, b in pairs],
             "in_dist": [a for a, _ in pairs]}
    train = [p[0] for p in pairs[:-20]]
    valid_sp = train[:]
    invalid_sp = [p[1] for p in pairs[:-20]]

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

    for arm in ("lora", "lora+side", "frozen+side"):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
        sides = {i: Side(model.config.hidden_size, model.config.vocab_size).to(DEV) for i in LAYERS_USED}
        trainable = [p for i in LAYERS_USED for p in sides[i].parameters()]
        if arm.startswith("lora"):
            # freeze FIRST, then wrap: wrapping after freezing also froze A/B
            for p in model.parameters():
                p.requires_grad_(False)
            for lyr in layers_of(model):
                for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(lyr.self_attn, nm, LoRALinear(getattr(lyr.self_attn, nm)))
            trainable = [p for p in model.parameters() if p.requires_grad] + \
                        [p for i in LAYERS_USED for p in sides[i].parameters()]
            log({"arm": arm, "lora_params": sum(p.numel() for p in trainable if p.requires_grad)})
            assert trainable and any(p.requires_grad for p in trainable), "no trainable params"
        else:
            for p in model.parameters():
                p.requires_grad_(False)
        use_side = arm != "lora"
        hooks = []
        if use_side:
            holder["sides"] = sides
            hooks = [layers_of(model)[i].register_forward_hook(make_hook(i)) for i in LAYERS_USED]
        opt = torch.optim.AdamW(trainable, lr=1e-3 if arm.startswith("lora") else 3e-3)
        rng = random.Random(1)
        for step in range(STEPS):
            seq = valid_sp[rng.randrange(len(valid_sp))] if rng.random() < 0.5 \
                else invalid_sp[rng.randrange(len(invalid_sp))]
            holder["src"] = build_src(seq).unsqueeze(0)
            ids = torch.tensor([seq], device=DEV)
            logits = model(input_ids=ids, attention_mask=masks[SPAN], use_cache=False).logits
            loss = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
            if use_side and holder["aux"]:
                av = holder["aux"][LAYERS_USED[-1]]
                if av is not None:
                    loss = loss + 0.1 * F.cross_entropy(
                        av[:, :-1].float().reshape(-1, model.config.vocab_size), ids[0, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            if step % 100 == 0:
                log({"arm": arm, "step": step, "train_loss": round(loss.item(), 4)})
        model.eval()
        for cond, seqs in conds.items():
            tot = [0.0, 0]
            with torch.no_grad():
                for s in seqs[-12:]:
                    holder["src"] = build_src(s).unsqueeze(0)
                    ids = torch.tensor([s], device=DEV)
                    logits = model(input_ids=ids, attention_mask=masks[T if len(s) > SPAN else SPAN],
                                   use_cache=False).logits
                    ce = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")
                    start = SPAN if len(s) > SPAN else 0
                    v = ce[start:].tolist()
                    tot[0] += sum(v); tot[1] += len(v)
            log({"arm": arm, "cond": cond,
                 "ppl": round(float(torch.tensor(tot[0] / max(tot[1], 1)).exp()), 3)})
        for hh in hooks:
            hh.remove()
        del model, sides, opt
        gc.collect(); torch.cuda.empty_cache()
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-900:]})
