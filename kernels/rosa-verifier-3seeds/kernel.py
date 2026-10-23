"""Verifier, with the control fixed.  SPAN=1024 > WIN=512, so the duplicated
copy is genuinely out of reach (the earlier run set SPAN==WIN and its dup
baseline collapsed to PPL 2.72, which made that control meaningless).

The decisive table is three conditions x three arms:
  dup         : valid long matches at distance 1024   -> verifier MUST keep the gain
  renamed     : matches collapse to ~4 tokens         -> verifier MUST NOT damage
  renamed_rot : invalid + off-diagonal correspondence -> same
Arms: none | verifier (metadata, trained on mixed validity) | no_meta ablation.
"""
import glob, json, os, random, re
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, T, WIN, LAYER, ROT = 1024, 2048, 512, 12, 340
STEPS, NP = 300, 80
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


def surface(txt, seed):
    txt = re.sub(r"#.*", "", txt)
    names = sorted(set(IDENT.findall(txt)))
    rnd = random.Random(seed); rnd.shuffle(names)
    mp = {n: "zq" + "".join(rnd.choice("abcdefghij") for _ in range(6)) + "x" for n in names}
    return re.sub(r"\s+", " ", IDENT.sub(lambda m: mp[m.group(0)], txt)).strip()


class Verifier(nn.Module):
    def __init__(self, d, meta=True):
        super().__init__()
        self.meta = meta
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d + (3 if meta else 0), 1)
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.last = {}

    def forward(self, h, src, feats=None):
        hn = self.ln(h.float())
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        x = torch.cat([hn, feats], -1) if (self.meta and feats is not None) else hn
        g = torch.sigmoid(self.gate(x) + self.gate_bias)
        self.last = {"gate": g.detach()}
        return torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))


def feats(src, ml):
    n = len(src); f = torch.zeros(n, 3)
    for j in range(n):
        if src[j] >= 0:
            f[j, 0] = min(ml[j], 64) / 64.0
            f[j, 1] = min(j - src[j], 2048) / 2048.0
            f[j, 2] = 1.0
    return f


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
        a = tok(body)["input_ids"]
        b = tok(surface(body, fi))["input_ids"]
        if len(a) < SPAN or len(b) < SPAN:
            continue
        out.append((a[:SPAN], b[:SPAN]))
        if len(out) >= want:
            break
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log({"stage": "load", "model": MODEL, "span": SPAN, "win": WIN, "rot": ROT})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size
    pairs = spans(tok)
    log({"pairs": len(pairs)})
    if len(pairs) < 10:
        log({"abort": "too few spans"}); return

    conds = {
        "dup": [a + a for a, _ in pairs],
        "renamed": [a + b for a, b in pairs],
        "renamed_rot": [a + (b[ROT:] + b[:ROT]) for a, b in pairs],
    }
    stats = {}
    for name, seqs in conds.items():
        mls = [m for s in seqs[:10] for m in build_src(s)[1][SPAN:]]
        stats[name] = {"match_mean": round(sum(mls) / max(1, len(mls)), 2),
                       "frac_ge16": round(sum(1 for m in mls if m >= 16) / max(1, len(mls)), 3)}
    log({"conditions": stats})

    def causal(n):
        ii = torch.arange(n)
        k = (ii[None, :] <= ii[:, None]) & ((ii[:, None] - ii[None, :]) < WIN)
        m = torch.zeros(1, 1, n, n, dtype=torch.float16, device=DEV)
        m.masked_fill_(~k.to(DEV), float("-inf")); return m
    masks = {SPAN: causal(SPAN), T: causal(T)}
    holder = {"src": None, "feats": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h, holder["src"].to(h.device),
                               holder["feats"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def fwd(seq, row, ml, n):
        holder["src"] = torch.tensor([row], dtype=torch.long)
        holder["feats"] = feats(row, ml).unsqueeze(0)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=masks[n], use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    valid = [p[0] for p in pairs[:-20]]
    invalid = [p[1] for p in pairs[:-20]]
    SEEDS = [0, 1, 2]
    for seed in SEEDS:
        built = {}
        for tag, meta in (("verifier", True), ("no_meta", False)):
            torch.manual_seed(seed)
            v = Verifier(d, meta).to(DEV); v.train()
            holder["active"] = v
            handle = model.model.layers[LAYER].register_forward_hook(hook)
            opt = torch.optim.AdamW(v.parameters(), lr=3e-3)
            rng = random.Random(seed + 1)
            for step in range(STEPS):
                seq = valid[rng.randrange(len(valid))] if rng.random() < 0.5 \
                    else invalid[rng.randrange(len(invalid))]
                row, ml = build_src(seq)
                loss = fwd(seq, row, ml, SPAN).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(v.parameters(), 1.0); opt.step()
            handle.remove(); v.eval(); built[tag] = v

        for cond, seqs in conds.items():
            for arm in ("none", "verifier", "no_meta"):
                h = None
                if arm != "none":
                    holder["active"] = built[arm]
                    h = model.model.layers[LAYER].register_forward_hook(hook)
                tot = [0.0, 0]; gv_ = []; gi_ = []
                with torch.no_grad():
                    for s in seqs[-12:]:
                        row, ml = build_src(s)
                        for j, ls in enumerate(fwd(s, row, ml, T).tolist()):
                            if j >= SPAN:
                                tot[0] += ls; tot[1] += 1
                        if arm != "none":
                            g = built[arm].last["gate"][0, SPAN:]
                            lm = torch.tensor(ml[SPAN:])
                            if (lm >= 16).any():
                                gv_.append(g[lm >= 16].mean().item())
                            if (lm < 8).any():
                                gi_.append(g[lm < 8].mean().item())
                if h:
                    h.remove()
                rec = {"seed": seed, "cond": cond, "arm": arm,
                       "ppl": round(float(torch.tensor(tot[0] / max(tot[1], 1)).exp()), 3)}
                if gv_:
                    rec["gate_valid"] = round(sum(gv_) / len(gv_), 4)
                if gi_:
                    rec["gate_invalid"] = round(sum(gi_) / len(gi_), 4)
                log(rec)
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-900:]})
