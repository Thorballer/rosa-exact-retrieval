"""Symbolic ROSA vs attention, long context, graded metric.

Index = real token ids (exact suffix automaton, verified 400/400 vs brute force),
value + gate learned.  Task: L=512 filler + 8 key->value pairs at distances
40..470 + 8 query keys; predict each value.  window=128, so most pairs are out
of window reach.

arms: full | window | win+trosa | full+trosa    metric: loss + acc by distance
"""
import json, time
import torch, torch.nn as nn, torch.nn.functional as F
"""Symbolic ROSA: exact suffix index over real token ids, learned value + gate.

Fix for the measured failure: the learned codebook never produced codes where the
same token matched its earlier occurrence (retrieval_pos_acc ~0.6-2.4% = chance).
Here the index is the token stream itself, so matches are exact by construction.
"""
import random, torch, torch.nn as nn


def build_src(x):
    """src[i] = position of the token that followed the longest earlier occurrence
    of a suffix of x[0..i]; -1 if none.  O(n) amortized (suffix automaton)."""
    n = len(x)
    s = 2 * n + 1
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


def brute_src(x):
    n = len(x); out = []
    for i in range(n):
        best = (0, -1)
        for L in range(1, i + 2):
            if i - L + 1 < 0:
                break
            for j in range(i - 1, L - 2, -1):
                if x[j - L + 1:j + 1] == x[i - L + 1:i + 1]:
                    if L > best[0] or (L == best[0] and j > best[1]):
                        best = (L, j)
        out.append(best[1] + 1 if best[1] >= 0 else -1)
    return out


class TokenROSA(nn.Module):
    """Exact symbolic retrieval + learned value/gate, added as a residual."""

    def __init__(self, d):
        super().__init__()
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)

    def forward(self, h, src):
        v = self.vproj(h)
        idx = src.clamp(min=0)[..., None].expand(-1, -1, v.shape[-1])
        gv = v.gather(1, idx)
        g = torch.sigmoid(self.gate(h))
        return h + torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))




OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NK, NV, NF = 64, 64, 64
V = NK + NV + NF
L, NP, WIN = 512, 8, 128
D, H, LAYERS = 96, 4, 2
STEPS, SEEDS, BS = 9000, 3, 16
N_TRAIN, N_EVAL = 6000, 1500
PAIR_MAX, Q_MIN = 300, 340
BUCKETS = [(0, 128), (128, 256), (256, 384), (384, 999)]


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    open(OUT, "a").write(line + "\n")


def make_set(n, seed, task="copy"):
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(NK + NV, V, (n, L), generator=g)
    keys = torch.stack([torch.randperm(NK, generator=g)[:NP] for _ in range(n)])
    vals = NK + torch.randint(0, NV, (n, NP), generator=g)
    slots = torch.linspace(8, PAIR_MAX - 16, NP).long()
    pair_pos = slots[None, :] + torch.randint(0, 12, (n, NP), generator=g)
    q_pos = torch.sort(torch.randperm(L - Q_MIN, generator=g)[:NP]).values[None, :] + Q_MIN
    q_pos = q_pos.repeat(n, 1)[:, torch.randperm(NP, generator=g)]   # decouple distance from slot
    labels = vals.clone()
    if task == "harm":                    # verbatim successor is the WRONG answer
        labels = NK + (vals - NK + 1 + torch.randint(0, NV - 1, (n, NP), generator=g)) % NV
    for b in range(n):
        toks[b, pair_pos[b]] = keys[b]
        toks[b, pair_pos[b] + 1] = vals[b]
        toks[b, q_pos[b]] = keys[b]
    src = torch.tensor([build_src(toks[b].tolist()) for b in range(n)], dtype=torch.long)
    return dict(toks=toks, src=src, qpos=q_pos, tgt=labels, vpos=pair_pos + 1,
                dist=q_pos - pair_pos)


class Block(nn.Module):
    def __init__(self, d, h, win, use_trosa):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.trosa = TokenROSA(d) if use_trosa else None
        self.h, self.win = h, win

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
        return self.head(self.ln(z))


def score(model, data, use_src):
    model.eval()
    with torch.no_grad():
        out = model(data["toks"].to(DEV), data["src"].to(DEV) if use_src else None)
        qp = data["qpos"].to(DEV)
        lg = out.gather(1, qp[..., None].expand(-1, -1, V))
        tgt, d = data["tgt"].to(DEV), data["dist"]
        hit = (lg.argmax(-1) == tgt)
        res = {"loss": round(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1)).item(), 3),
               "acc": round(hit.float().mean().item(), 3), "by_dist": {}}
        for lo, hi in BUCKETS:
            m = ((d >= lo) & (d < hi)).to(DEV)
            if m.sum():
                res["by_dist"][f"{lo}-{hi if hi < 999 else 'inf'}"] = {
                    "acc": round((hit & m).sum().item() / m.sum().item(), 2), "n": int(m.sum())}
        if use_src:
            res["src_pos_acc"] = round(
                (data["src"].gather(1, qp.cpu()) == data["vpos"]).float().mean().item(), 3)
    model.train()
    return res


def run(arm, seed, TR, EV, task):
    torch.manual_seed(seed)
    use_trosa = arm.endswith("trosa")
    model = Model(L if arm.startswith("full") else WIN, use_trosa).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=STEPS, pct_start=0.1)
    g = torch.Generator().manual_seed(seed + 7)
    t0 = time.perf_counter(); curve = []
    for step in range(STEPS):
        idx = torch.randint(0, TR["toks"].shape[0], (BS,), generator=g)
        out = model(TR["toks"][idx].to(DEV), TR["src"][idx].to(DEV) if use_trosa else None)
        qp = TR["qpos"][idx].to(DEV)
        lg = out.gather(1, qp[..., None].expand(-1, -1, V))
        loss = F.cross_entropy(lg.reshape(-1, V), TR["tgt"][idx].to(DEV).reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % (STEPS // 5) == 0 or step == STEPS - 1:
            curve.append(round(loss.item(), 3))
    r = score(model, EV, use_trosa)
    r.update(arm=arm, seed=seed, task=task, steps=STEPS,
             params=sum(p.numel() for p in model.parameters()),
             train_loss_curve=curve, secs=round(time.perf_counter() - t0, 1))
    return r


if __name__ == "__main__":
    print(f"device={DEV}", flush=True)
    t0 = time.perf_counter()
    sets = {"copy": (make_set(N_TRAIN, 1), make_set(N_EVAL, 2)),
            "harm": (make_set(N_TRAIN, 1, "harm"), make_set(N_EVAL, 2, "harm"))}
    log({"data_build_s": round(time.perf_counter() - t0, 1), "L": L, "pairs": NP, "window": WIN,
         "chance_acc": round(1 / NV, 4), "index": "raw token ids, exact SAM"})
    for task, (TR, EV) in sets.items():
        log({"task": task, "oracle_src_pos_acc":
             round((EV["src"].gather(1, EV["qpos"]) == EV["vpos"]).float().mean().item(), 3),
             "dist_range": [int(EV["dist"].min()), int(EV["dist"].max())]})
        for arm in ("full", "window", "win+trosa", "full+trosa"):
            for s in range(SEEDS if task == "copy" else 1):
                try:
                    log(run(arm, s, TR, EV, task))
                except Exception as e:
                    log({"arm": arm, "task": task, "seed": s, "error": repr(e)[:300]})
    log({"done": True})
