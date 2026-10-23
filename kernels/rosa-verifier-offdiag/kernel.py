"""Three fixes tested together.

1. VERIFIER: gate input gains match metadata (length, distance, present?) and the
   sidecar is trained on a MIX where half the batches have invalid matches
   (surface-renamed spans).  Does it stop the 2.4x damage from wrong matches
   while keeping the gain on valid ones?
2. NO-METADATA ABLATION: same mixed training, gate sees only the hidden state.
   Isolates whether metadata is what makes the verifier work.
3. OFF-DIAGONAL STRUCTURAL TEST: the renamed copy is ROTATED, so its structural
   correspondence sits far from the diagonal (a_idx = (j + rot) % SPAN).  A match
   that is "within tolerance" can no longer be found by simply pointing one span
   back.

Conditions
  dup         : A ++ A            (valid long matches, distance SPAN)
  renamed     : A ++ rename(A)    (matches collapse to ~4 tokens: invalid)
  renamed_rot : A ++ rotate(rename(A))   (invalid AND off-diagonal correspondence)
"""
import glob, json, os, random, re
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, T, WIN, LAYER, ROT = 512, 1024, 512, 12, 170
STEPS, NP = 300, 100
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
    """gate sees the hidden state and (optionally) match metadata."""

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
        x = torch.cat([hn, feats], -1) if self.meta and feats is not None else hn
        g = torch.sigmoid(self.gate(x) + self.gate_bias)
        self.last = {"gate": g.detach()}
        return torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))


def feats_from(src, ml):
    n = len(src)
    f = torch.zeros(n, 3)
    for j in range(n):
        if src[j] >= 0:
            f[j, 0] = min(ml[j], 64) / 64.0
            f[j, 1] = min(j - src[j], 1024) / 1024.0
            f[j, 2] = 1.0
    return f


def make_pairs(tok, want=NP):
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
            body = open(f, encoding="utf-8", errors="ignore").read()[:4000]
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
    log({"stage": "load", "model": MODEL, "device": DEV, "rot": ROT})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size
    pairs = make_pairs(tok)
    log({"pairs": len(pairs)})

    dups, rens, rots = [], [], []
    for a, b in pairs:
        dups.append(a + a)
        rens.append(a + b)
        brot = b[ROT:] + b[:ROT]
        rots.append(a + brot)
    stats = {}
    for name, seqs in (("dup", dups), ("renamed", rens), ("renamed_rot", rots)):
        mls = [m for s in seqs[:20] for m in build_src(s)[1][SPAN:]]
        stats[name] = {"match_mean": round(sum(mls) / max(1, len(mls)), 2),
                       "frac_ge16": round(sum(1 for m in mls if m >= 16) / max(1, len(mls)), 3)}
    log({"conditions": stats})

    idx = torch.arange(T)
    keep = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < WIN)
    mask = torch.zeros(1, 1, T, T, dtype=torch.float16, device=DEV)
    mask.masked_fill_(~keep.to(DEV), float("-inf"))
    ti = torch.arange(SPAN)
    kt = (ti[None, :] <= ti[:, None]) & ((ti[:, None] - ti[None, :]) < WIN)
    mask_s = torch.zeros(1, 1, SPAN, SPAN, dtype=torch.float16, device=DEV)
    mask_s.masked_fill_(~kt.to(DEV), float("-inf"))

    holder = {"src": None, "feats": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h, holder["src"].to(h.device),
                               holder["feats"].to(h.device) if holder["feats"] is not None else None)
        add = add.to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def fwd(seq, row, ml, n):
        holder["src"] = torch.tensor([row], dtype=torch.long)
        holder["feats"] = feats_from(row, ml).unsqueeze(0)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=mask if n == T else mask_s,
                       use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    train_valid = [s[:SPAN] for s in dups[:60]]
    train_invalid = [s[SPAN:] for s in rens[:60]]      # the RENAMED half: matches invalid
    built = {}
    for tag, meta in (("verifier", True), ("no_meta", False)):
        v = Verifier(d, meta).to(DEV); v.train()
        holder["active"] = v
        handle = model.model.layers[LAYER].register_forward_hook(hook)
        opt = torch.optim.AdamW(v.parameters(), lr=3e-3)
        rng = random.Random(1)
        for step in range(STEPS):
            if rng.random() < 0.5:
                seq = train_valid[rng.randrange(len(train_valid))]
            else:
                seq = train_invalid[rng.randrange(len(train_invalid))]
            row, ml = build_src(seq)
            loss = fwd(seq, row, ml, SPAN).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(v.parameters(), 1.0); opt.step()
            if step % 100 == 0:
                log({"verifier": tag, "step": step, "loss": round(loss.item(), 4),
                     "gate": round(v.last["gate"].mean().item(), 4)})
        handle.remove(); v.eval(); built[tag] = v

    for cond, seqs in (("dup", dups), ("renamed", rens), ("renamed_rot", rots)):
        for arm in ("none", "verifier", "no_meta"):
            h = None
            if arm != "none":
                holder["active"] = built[arm]
                h = model.model.layers[LAYER].register_forward_hook(hook)
            tot = [0.0, 0]; gates = []
            with torch.no_grad():
                for s in seqs[-20:]:
                    row, ml = build_src(s)
                    for j, ls in enumerate(fwd(s, row, ml, T).tolist()):
                        if j >= SPAN:
                            tot[0] += ls; tot[1] += 1
                    if arm != "none":
                        gates.append(built[arm].last["gate"][0, SPAN:].mean().item())
            if h:
                h.remove()
            log({"cond": cond, "arm": arm,
                 "ppl": round(float(torch.tensor(tot[0] / max(tot[1], 1)).exp()), 2),
                 **({"gate_renamed_half": round(sum(gates) / len(gates), 4)} if gates else {})})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-900:]})
