"""Can the learned read gate learn to VETO?  (identifiable version)

Previous harm control was unidentifiable: sequence looked identical whether the
verbatim successor was right or wrong, so "always copy" was the only learnable
policy.  Fix: a FLAG token sits just before the queries (inside the window).

  flag=COPY  -> label is the paired value          (copy is correct)
  flag=HARM  -> label is a different value         (copy is exactly wrong)

Train on a 50/50 mix, evaluate on pure-copy and pure-harm sets separately.
Success = high accuracy on COPY, chance accuracy on HARM, and a LOWER gate on
HARM than on COPY.
"""
import json, time
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NK, NV, NF = 64, 64, 64
FLAG_COPY, FLAG_HARM = NK + NV + NF, NK + NV + NF + 1
V = FLAG_HARM + 1
L, NP, WIN = 512, 8, 128
D, H, LAYERS = 96, 4, 2
STEPS, SEEDS, BS = 4000, 3, 16
N_TRAIN, N_EVAL = 6000, 1500
PAIR_MAX, Q_MIN = 300, 340
BUCKETS = [(0, 128), (128, 256), (256, 384), (384, 999)]


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
                src[i] = e[v] + 1
                break
            v = c[v]
        v = g
        while v != -1 and e[v] < i:
            e[v] = i; v = c[v]
    return src


class TokenROSA(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.last_gate = None

    def forward(self, h, src):
        v = self.vproj(h)
        idx = src.clamp(min=0)[..., None].expand(-1, -1, v.shape[-1])
        gv = v.gather(1, idx)
        g = torch.sigmoid(self.gate(h))
        self.last_gate = g.detach()
        return h + torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))


def make_set(n, seed, mode="mixed"):
    """mode: mixed (per-row random flag) | copy | harm"""
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(NK + NV, FLAG_COPY, (n, L), generator=g)
    keys = torch.stack([torch.randperm(NK, generator=g)[:NP] for _ in range(n)])
    vals = NK + torch.randint(0, NV, (n, NP), generator=g)
    slots = torch.linspace(8, PAIR_MAX - 16, NP).long()
    pair_pos = slots[None, :] + torch.randint(0, 12, (n, NP), generator=g)
    q_pos = torch.sort(torch.randperm(L - Q_MIN, generator=g)[:NP]).values[None, :] + Q_MIN
    q_pos = q_pos.repeat(n, 1)[:, torch.randperm(NP, generator=g)]
    if mode == "copy":
        is_copy = torch.ones(n, dtype=torch.bool)
    elif mode == "harm":
        is_copy = torch.zeros(n, dtype=torch.bool)
    else:
        is_copy = torch.rand(n, generator=g) < 0.5
    other = NK + (vals - NK + 1 + torch.randint(0, NV - 1, (n, NP), generator=g)) % NV
    labels = torch.where(is_copy[:, None], vals, other)
    for b in range(n):
        toks[b, pair_pos[b]] = keys[b]
        toks[b, pair_pos[b] + 1] = vals[b]
        toks[b, q_pos[b]] = keys[b]
        toks[b, Q_MIN - 1] = FLAG_COPY if is_copy[b] else FLAG_HARM
    src = torch.tensor([build_src(toks[b].tolist()) for b in range(n)], dtype=torch.long)
    return dict(toks=toks, src=src, qpos=q_pos, tgt=labels, vpos=pair_pos + 1,
                dist=q_pos - pair_pos, is_copy=is_copy)


class Block(nn.Module):
    def __init__(self, d, h, win, use_trosa):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.trosa = TokenROSA(d) if use_trosa else None
        self.h, self.win, self.g = h, win, None

    def forward(self, x, src=None):
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
            x = self.trosa(self.ln2(x), src)
            self.g = self.trosa.last_gate
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self, win, use_trosa):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block(D, H, win, use_trosa) for _ in range(LAYERS)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)

    def forward(self, toks, src=None):
        z = self.emb(toks)
        for b in self.blocks:
            z = b(z, src)
        return self.head(self.ln(z)), self.blocks[-1].g


def score(model, data, use_src):
    model.eval()
    with torch.no_grad():
        out, gate = model(data["toks"].to(DEV), data["src"].to(DEV) if use_src else None)
        qp = data["qpos"].to(DEV)
        lg = out.gather(1, qp[..., None].expand(-1, -1, V))
        tgt = data["tgt"].to(DEV)
        res = {"loss": round(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1)).item(), 3),
               "acc": round((lg.argmax(-1) == tgt).float().mean().item(), 3)}
        if gate is not None:
            res["gate_at_q"] = round(gate.gather(1, qp[..., None]).mean().item(), 4)
            res["src_pos_acc"] = round(
                (data["src"].gather(1, qp.cpu()) == data["vpos"]).float().mean().item(), 3)
    model.train()
    return res


def run(arm, seed, TR, EV_C, EV_H):
    torch.manual_seed(seed)
    use_trosa = arm.endswith("trosa")
    model = Model(L if arm.startswith("full") else WIN, use_trosa).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=STEPS, pct_start=0.1)
    g = torch.Generator().manual_seed(seed + 7)
    t0 = time.perf_counter(); curve = []
    for step in range(STEPS):
        idx = torch.randint(0, TR["toks"].shape[0], (BS,), generator=g)
        out, _ = model(TR["toks"][idx].to(DEV), TR["src"][idx].to(DEV) if use_trosa else None)
        qp = TR["qpos"][idx].to(DEV)
        lg = out.gather(1, qp[..., None].expand(-1, -1, V))
        loss = F.cross_entropy(lg.reshape(-1, V), TR["tgt"][idx].to(DEV).reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % (STEPS // 5) == 0 or step == STEPS - 1:
            curve.append(round(loss.item(), 3))
    r = {"arm": arm, "seed": seed, "task": "mixed_train", "steps": STEPS,
         "params": sum(p.numel() for p in model.parameters()),
         "train_loss_curve": curve, "secs": round(time.perf_counter() - t0, 1)}
    for name, ds in (("eval_copy", EV_C), ("eval_harm", EV_H)):
        r[name] = score(model, ds, use_trosa)
    return r


if __name__ == "__main__":
    print(f"device={DEV}", flush=True)
    t0 = time.perf_counter()
    TR = make_set(N_TRAIN, 1, "mixed")
    EV_C, EV_H = make_set(N_EVAL, 2, "copy"), make_set(N_EVAL, 3, "harm")
    log({"data_build_s": round(time.perf_counter() - t0, 1), "L": L, "window": WIN,
         "chance_acc": round(1 / NV, 4),
         "note": "flag token at Q_MIN-1 identifies whether copying is correct"})
    for arm in ("full", "window", "win+trosa", "full+trosa"):
        for s in range(SEEDS):
            try:
                log(run(arm, s, TR, EV_C, EV_H))
            except Exception as e:
                log({"arm": arm, "seed": s, "error": repr(e)[:300]})
    log({"done": True})
