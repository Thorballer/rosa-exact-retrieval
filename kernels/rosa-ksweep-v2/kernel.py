"""Codebook size vs retrieval accuracy, v2.

v1 was void: 600 steps left retrieval at chance for every K (0.001-0.025) while the
same sweep at 2500 steps previously gave 0.59/0.80/0.93.  So: 2500 steps, batch 4
with 2-step accumulation, and explicit memory release between configs (K=4096 OOM'd
on a T4 in v1 trying to allocate 6.25 GiB).

Metric: idx_pos_acc = fraction of queries where the code-index points at the true
value position.  Oracle over raw token ids = 1.0.
"""
import gc, json
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NK, NV, NF = 64, 64, 64
V = NK + NV + NF
L, NP, WIN, D, H, LAYERS = 512, 8, 128, 96, 4, 2
STEPS, BS, ACCUM, NTR, NEV = 2500, 4, 2, 4000, 800
KS = [256, 1024, 4096]
PAIR_MAX, Q_MIN = 300, 340


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
    return src


def make(n, seed):
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(NK + NV, V, (n, L), generator=g)
    keys = torch.stack([torch.randperm(NK, generator=g)[:NP] for _ in range(n)])
    vals = NK + torch.randint(0, NV, (n, NP), generator=g)
    slots = torch.linspace(8, PAIR_MAX - 16, NP).long()
    pp = slots[None, :] + torch.randint(0, 12, (n, NP), generator=g)
    qp = torch.sort(torch.randperm(L - Q_MIN, generator=g)[:NP]).values[None, :] + Q_MIN
    qp = qp.repeat(n, 1)[:, torch.randperm(NP, generator=g)]
    for b in range(n):
        toks[b, pp[b]] = keys[b]
        toks[b, pp[b] + 1] = vals[b]
        toks[b, qp[b]] = keys[b]
    src = torch.tensor([build_src(toks[b].tolist()) for b in range(n)], dtype=torch.long)
    return dict(toks=toks, src=src, qpos=qp, tgt=vals, vpos=pp + 1)


class Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.ln1, self.ln2, self.lnf = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.proj_code = nn.Linear(D, k)
        self.code_emb = nn.Embedding(k, D)
        self.aux = nn.Linear(D, V)
        self.vproj, self.gate = nn.Linear(D, D), nn.Linear(D, 1)
        self.head = nn.Linear(D, V)
        self.last = {}

    def forward(self, toks):
        z = self.emb(toks); b, n, d = z.shape
        for _ in range(LAYERS):
            q, kk, v = self.qkv(self.ln1(z)).view(b, n, 3, H, d // H).permute(2, 0, 3, 1, 4)
            idx = torch.arange(n, device=z.device)
            m = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < WIN)
            a = F.scaled_dot_product_attention(q, kk, v, attn_mask=m)
            z = z + self.proj(a.transpose(1, 2).reshape(b, n, d))
            logits = self.proj_code(self.ln2(z))
            codes = logits.argmax(-1)
            soft = F.softmax(logits, -1) @ self.code_emb.weight
            aux = F.cross_entropy(self.aux(soft)[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1))
            src = torch.tensor([build_src(codes[i].tolist()) for i in range(b)], device=z.device)
            gv = self.vproj(self.ln2(z)).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, d))
            g = torch.sigmoid(self.gate(self.ln2(z)))
            z = z + torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(z))
            z = z + self.mlp(self.ln2(z))
            self.last = {"src": src, "aux": aux}
        return self.head(self.lnf(z)), self.last


def score(model, data):
    model.eval()
    with torch.no_grad():
        out, info = model(data["toks"].to(DEV))
        qp = data["qpos"].to(DEV)
        pred = out.gather(1, qp[..., None].expand(-1, -1, V)).argmax(-1)
        acc = (pred == data["tgt"].to(DEV)).float().mean().item()
        idx = (info["src"].cpu().gather(1, data["qpos"]) == data["vpos"]).float().mean().item()
    model.train()
    return round(acc, 3), round(idx, 3)


def run(k, seed, TR, EV):
    torch.manual_seed(seed)
    model = Net(k).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed + 7)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(ACCUM):
            idx = torch.randint(0, TR["toks"].shape[0], (BS,), generator=g)
            out, info = model(TR["toks"][idx].to(DEV))
            qp = TR["qpos"][idx].to(DEV)
            loss = F.cross_entropy(out.gather(1, qp[..., None].expand(-1, -1, V)).reshape(-1, V),
                                   TR["tgt"][idx].to(DEV).reshape(-1)) + 0.1 * info["aux"]
            (loss / ACCUM).backward()
            tot += loss.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            log({"K": k, "seed": seed, "step": step, "loss": round(tot, 4)})
    acc, idx_acc = score(model, EV)
    res = dict(K=k, seed=seed, acc=acc, idx_pos_acc=idx_acc,
               params=sum(p.numel() for p in model.parameters()))
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    print(f"device={DEV} V={V}", flush=True)
    TR, EV = make(NTR, 1), make(NEV, 2)
    log({"oracle_idx_pos_acc": round((EV["src"].gather(1, EV["qpos"]) == EV["vpos"]).float().mean().item(), 3),
         "V": V, "chance_acc": round(1 / NV, 4), "steps": STEPS})
    for k in KS:
        try:
            log(run(k, 0, TR, EV))
        except Exception as e:
            log({"K": k, "error": repr(e)[:200]})
            gc.collect(); torch.cuda.empty_cache()
    log({"done": True})
