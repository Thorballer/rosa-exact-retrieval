"""Fix the codebook: quantize the token embedding instead of the hidden state.

  raw       : index = token id                       (upper bound, exact)
  hid       : index = codes of LN(h)                 (rosa-torch's design)
  tok:K     : index = codes of LN(E[x])              (this fix, K codes)

Sweeping K exposes the central tension: a codebook large enough to keep keys
distinct retrieves exactly; a small one merges tokens ('fuzzy') and the suffix
match starts returning the wrong earlier occurrence.
"""
import json, time
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NK, NV, NF = 64, 64, 64
V = NK + NV + NF
L, NP, WIN = 512, 8, 128
D, H, LAYERS = 96, 4, 2
STEPS, SEEDS, BS = 2500, 2, 8
N_TRAIN, N_EVAL = 6000, 1500
PAIR_MAX, Q_MIN = 300, 340
CONFIGS = [("raw", 0), ("hid", 256), ("tok", 64), ("tok", 256), ("tok", 1024)]


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
        v = g = r; a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = e[v] + 1; break
            v = c[v]
        src[i] = a
        v = g
        while v != -1 and e[v] < i:
            e[v] = i; v = c[v]
    return src


def make_set(n, seed):
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(NK + NV, V, (n, L), generator=g)
    keys = torch.stack([torch.randperm(NK, generator=g)[:NP] for _ in range(n)])
    vals = NK + torch.randint(0, NV, (n, NP), generator=g)
    slots = torch.linspace(8, PAIR_MAX - 16, NP).long()
    pair_pos = slots[None, :] + torch.randint(0, 12, (n, NP), generator=g)
    q_pos = torch.sort(torch.randperm(L - Q_MIN, generator=g)[:NP]).values[None, :] + Q_MIN
    q_pos = q_pos.repeat(n, 1)[:, torch.randperm(NP, generator=g)]
    for b in range(n):
        toks[b, pair_pos[b]] = keys[b]
        toks[b, pair_pos[b] + 1] = vals[b]
        toks[b, q_pos[b]] = keys[b]
    src = torch.tensor([build_src(toks[b].tolist()) for b in range(n)], dtype=torch.long)
    return dict(toks=toks, src=src, qpos=q_pos, tgt=vals, vpos=pair_pos + 1,
                dist=q_pos - pair_pos)


class QuantROSA(nn.Module):
    def __init__(self, d, k, mode, vocab):
        super().__init__()
        self.mode = mode
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Linear(d, k) if k else None
        self.code_emb = nn.Embedding(k, d) if k else None
        self.aux = nn.Linear(d, vocab) if k else None
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.last = {}

    def forward(self, h, x, emb, src_hint=None):
        if self.mode == "raw":
            codes, aux, n_codes = x, None, int(x.unique().numel())
        else:
            source = emb if self.mode == "tok" else h
            logits = self.proj(self.ln(source))
            codes = logits.argmax(-1)
            code_soft = F.softmax(logits, dim=-1) @ self.code_emb.weight
            aux = F.cross_entropy(self.aux(code_soft)[:, :-1].reshape(-1, V),
                                  x[:, 1:].reshape(-1))
            n_codes = int(codes.unique().numel())
        if src_hint is not None:
            src = src_hint.to(h.device)
        else:
            src = torch.tensor([build_src(codes[b].tolist()) for b in range(codes.shape[0])],
                               device=h.device)
        v = self.vproj(h)
        gv = v.gather(1, src.clamp(min=0)[..., None].expand(-1, -1, v.shape[-1]))
        g = torch.sigmoid(self.gate(h))
        out = h + torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))
        self.last = {"src": src, "gate": g.detach(), "aux": aux, "n_codes": n_codes}
        return out


class Block(nn.Module):
    def __init__(self, d, h, win, mode, k, vocab):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.trosa = QuantROSA(d, k, mode, vocab) if mode else None
        self.h, self.win, self.info = h, win, None

    def forward(self, x, toks, emb, src_hint=None):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.win >= n:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            idx = torch.arange(n, device=x.device)
            mask = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < self.win)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        if self.trosa is not None:
            x = self.trosa(self.ln2(x), toks, emb, src_hint)
            self.info = self.trosa.last
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self, win, mode, k):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block(D, H, win, mode, k, V) for _ in range(LAYERS)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, toks, src_hint=None):
        e = self.emb(toks)
        z = e
        for b in self.blocks:
            z = b(z, toks, e, src_hint)
        return self.head(self.ln(z)), self.blocks[-1].info


def score(model, data):
    model.eval()
    with torch.no_grad():
        out, info = model(data["toks"].to(DEV),
                          data["src"].to(DEV) if data.get("hint") else None)
        qp = data["qpos"].to(DEV)
        lg = out.gather(1, qp[..., None].expand(-1, -1, V))
        tgt = data["tgt"].to(DEV)
        res = {"loss": round(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1)).item(), 3),
               "acc": round((lg.argmax(-1) == tgt).float().mean().item(), 3)}
        if info is not None:
            res["oracle_src_acc"] = round(
                (data["src"].gather(1, qp.cpu()) == data["vpos"]).float().mean().item(), 3)
            res["idx_pos_acc"] = round(
                (info["src"].cpu().gather(1, qp.cpu()) == data["vpos"]).float().mean().item(), 3)
            res["gate_at_q"] = round(info["gate"].cpu().gather(1, qp[..., None].cpu()).mean().item(), 4)
            res["n_codes_used"] = info["n_codes"]
    model.train()
    return res


def run(mode, k, seed, TR, EV):
    torch.manual_seed(seed)
    model = Model(WIN, mode, k).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=STEPS, pct_start=0.1)
    g = torch.Generator().manual_seed(seed + 7)
    t0 = time.perf_counter(); curve = []
    for step in range(STEPS):
        idx = torch.randint(0, TR["toks"].shape[0], (BS,), generator=g)
        hint = TR["src"][idx].to(DEV) if mode == "raw" else None
        out, info = model(TR["toks"][idx].to(DEV), hint)
        qp = TR["qpos"][idx].to(DEV)
        lg = out.gather(1, qp[..., None].expand(-1, -1, V))
        loss = F.cross_entropy(lg.reshape(-1, V), TR["tgt"][idx].to(DEV).reshape(-1))
        if info is not None and info["aux"] is not None:
            loss = loss + 0.1 * info["aux"]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % (STEPS // 5) == 0 or step == STEPS - 1:
            curve.append(round(loss.item(), 3))
    TR["hint"] = EV["hint"] = (mode == "raw")
    r = score(model, EV)
    r.update(index=f"{mode}{'' if not k else ':' + str(k)}", seed=seed, steps=STEPS,
             params=sum(p.numel() for p in model.parameters()),
             train_loss_curve=curve, secs=round(time.perf_counter() - t0, 1))
    return r


if __name__ == "__main__":
    print(f"device={DEV}", flush=True)
    t0 = time.perf_counter()
    TR, EV = make_set(N_TRAIN, 1), make_set(N_EVAL, 2)
    log({"data_build_s": round(time.perf_counter() - t0, 1), "L": L, "window": WIN,
         "distinct_tokens": V, "chance_acc": round(1 / NV, 4),
         "keys_per_code_warning": f"{V} distinct tokens into K codes"})
    for mode, k in CONFIGS:
        for s in range(SEEDS):
            try:
                log(run(mode, k, s, TR, EV))
            except Exception as e:
                log({"index": f"{mode}:{k}", "seed": s, "error": repr(e)[:300]})
    log({"done": True})
