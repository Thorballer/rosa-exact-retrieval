"""Accuracy improvements that don't cost memory.

The failure modes are (a) hard top-1 picks one continuation by a dumb rule, and
(b) it fires where matching is useless.  Three fixes, all O(1) memory per token:

  topK      : walk the suffix-link chain and collect up to K matches at DIFFERENT
              ORDERS (multi-order backoff, PPM-style), then mix them with a small
              learned softmax over candidates.  Extra storage: ZERO - candidates
              are computed from the existing index at query time.
  lengate   : fire only when the longest match is >= MINLEN tokens.  Hard rule, no
              parameters.  Targets the "harmful where matching is unreliable" case.
  topK+lengate

Arms: none | top1 (=current design) | top4 | top1+lengate | top4+lengate
Conditions: dup (valid long matches) | renamed (invalid) | renamed_rot (invalid,
off-diagonal).
Also logs bytes/token for the index vs a KV cache for the same model.
"""
import glob, json, os, random, re
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
SPAN, T, WIN, LAYER, ROT = 1024, 2048, 512, 12, 340
STEPS, NP, KCAND, MINLEN = 400, 80, 4, 8
IDENT = re.compile(r"\b[a-z_][a-z0-9_]{1,}\b")


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    open(OUT, "a").write(line + "\n")


def build_topk(x, K=KCAND):
    """per position: up to K (src, match_len) pairs, longest match first.
    Walks the suffix-link chain, so candidates are matches at decreasing orders."""
    n = len(x); s = 2 * n + 1
    b = [None] * s; c = [-1] * s; d = [0] * s; e = [-1] * s
    b[0] = {}; g = 0; z = 1
    srcs = [[-1] * K for _ in range(n)]
    mls = [[0] * K for _ in range(n)]
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
        v = g = r; k = 0
        while v != -1 and k < K:
            if d[v] > 0 and e[v] >= 0:
                srcs[i][k] = e[v] + 1; mls[i][k] = d[v]; k += 1
            v = c[v]
        v = g
        while v != -1 and e[v] < i:
            e[v] = i; v = c[v]
    S = torch.tensor(srcs, dtype=torch.long)
    M = torch.tensor(mls, dtype=torch.long)
    return S, M


class Mixer(nn.Module):
    """mixes up to K candidate value vectors with a learned softmax over candidates."""

    def __init__(self, d, lengate=False):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.scorer = nn.Sequential(nn.Linear(d + 3, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate = nn.Linear(d, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        self.lengate = lengate
        self.last = {}

    def forward(self, h, S, M):
        """h [B,N,D]; S,M [B,N,K]"""
        B, N, D = h.shape
        K = S.shape[-1]
        hn = self.ln(h.float())
        v = self.vproj(hn)                                         # [B,N,D]
        # gather each candidate along the SEQUENCE axis (dim 1), not the K axis
        cands = []
        for kk in range(K):
            idx1 = S[..., kk].clamp(min=0)[..., None].expand(-1, -1, D)   # [B,N,D]
            cands.append(v.gather(1, idx1))
        cand = torch.stack(cands, 2)                               # [B,N,K,D]
        valid = (S >= 0)
        if self.lengate:
            valid = valid & (M[:, :, :1] >= MINLEN)                # gate on best match len
        mf = M.float().clamp(max=64) / 64.0
        df = torch.zeros_like(mf)
        j = torch.arange(N, device=h.device)[None, :, None]
        df = ((j - S.clamp(min=0)).float().clamp(max=2048) / 2048.0)
        feats = torch.cat([cand, mf[..., None], df[..., None],
                           valid.float()[..., None]], -1)          # [B,N,K,D+3]
        logits = self.scorer(feats).squeeze(-1)                    # [B,N,K]
        logits = logits.masked_fill(~valid, -1e9)
        w = torch.softmax(logits, -1)
        mixed = (w[..., None] * torch.where(valid[..., None], cand,
                                            torch.zeros_like(cand))).sum(2)
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        any_valid = valid.any(-1, keepdim=True)
        out = torch.where(any_valid, g * mixed, torch.zeros_like(mixed))
        if not getattr(self, "_logged", False):
            self._logged = True
            print(json.dumps({"mixer_shapes": {"h": list(h.shape), "S": list(S.shape),
                                               "cand": list(cand.shape), "w": list(w.shape)}}), flush=True)
        self.last = {"gate": g.detach(), "w": w.detach(), "valid": valid.float().detach()}
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
    log({"stage": "load", "model": MODEL, "span": SPAN, "win": WIN, "K": KCAND, "minlen": MINLEN})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size
    cfg = model.config
    kv_bytes = 2 * cfg.num_key_value_heads * (d // cfg.num_attention_heads) * 2 * cfg.num_hidden_layers
    log({"memory_per_token_bytes": {"kv_cache_fp16": kv_bytes,
                                    "index_bound_est": 60,
                                    "ratio": round(kv_bytes / 60, 1),
                                    "note": "top-K candidates add zero storage"}})

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
    holder = {"S": None, "M": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h, holder["S"].to(h.device), holder["M"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def fwd(seq, S, M, n):
        holder["S"] = S.unsqueeze(0); holder["M"] = M.unsqueeze(0)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=masks[n], use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    valid_spans = [p[0] for p in pairs[:-20]]
    invalid_spans = [p[1] for p in pairs[:-20]]
    arms = (("top1", 1, False), ("top4", KCAND, False),
            ("top1_len", 1, True), ("top4_len", KCAND, True))
    built = {}
    for tag, k, lg in arms:
        torch.manual_seed(0)
        mx = Mixer(d, lg).to(DEV); mx.train()
        holder["active"] = mx
        handle = model.model.layers[LAYER].register_forward_hook(hook)
        opt = torch.optim.AdamW(mx.parameters(), lr=3e-3)
        rng = random.Random(1)
        for step in range(STEPS):
            seq = valid_spans[rng.randrange(len(valid_spans))] if rng.random() < 0.5 \
                else invalid_spans[rng.randrange(len(invalid_spans))]
            S, M = build_topk(seq, k)
            loss = fwd(seq, S, M, SPAN).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(mx.parameters(), 1.0); opt.step()
            if step % 200 == 0:
                log({"arm": tag, "step": step, "loss": round(loss.item(), 4)})
        handle.remove(); mx.eval(); built[tag] = (mx, k)
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()

    for cond, seqs in conds.items():
        for arm in (("none",) + tuple(a[0] for a in arms)):
            h = None
            if arm != "none":
                mx, k = built[arm]; holder["active"] = mx
                h = model.model.layers[LAYER].register_forward_hook(hook)
            tot = [0.0, 0]; fire = [0, 0]
            with torch.no_grad():
                for s in seqs[-12:]:
                    kk = 1 if arm in ("none", "top1", "top1_len") else KCAND
                    S, M = build_topk(s, kk)
                    for j, ls in enumerate(fwd(s, S, M, T).tolist()):
                        if j >= SPAN:
                            tot[0] += ls; tot[1] += 1
                    if arm != "none":
                        v = built[arm][0].last["valid"][0, SPAN:]
                        fire[0] += v.any(-1).float().sum().item(); fire[1] += v.shape[0]
            if h:
                h.remove()
            log({"cond": cond, "arm": arm,
                 "ppl": round(float(torch.tensor(tot[0] / max(tot[1], 1)).exp()), 3),
                 **({"fire_rate": round(fire[0] / max(fire[1], 1), 3)} if fire[1] else {})})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:300], "trace": traceback.format_exc()[-900:]})
