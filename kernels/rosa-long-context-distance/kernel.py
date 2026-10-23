"""Does long-range reach collapse?  Fixed long contexts, swept retrieval distance.

Task (needle-by-distance): random filler tokens, one key->value pair at distance d
before a query key at the end; predict the value.  No positional embeddings anywhere,
so nothing can cheat by extrapolating position encodings.

Train: L=384, d in [32, 320]
Eval : L=4096 fixed, d in {32, 256, 1024, 2048, 4000}  -> accuracy AND loss vs distance
       plus the in-distribution L=384 sanity point

arms: full (causal attention), window (w=256), win+rosa, full+rosa
"""
import json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rosa-torch"], check=False)

import torch, torch.nn as nn, torch.nn.functional as F
from rosa import ROSA

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FK, FV, FF = 64, 64, 256                     # keys 0-63, values 64-127, filler 128-383
V = FK + FV + FF
DISTANCES = [32, 256, 1024, 2048, 4000]
EVAL_LEN, TRAIN_LEN, WIN = 4096, 384, 256
STEPS, SEEDS, D_MODEL, HEADS, LAYERS = 1000, 3, 96, 4, 2


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def make(bs, L, dlo, dhi, gen):
    """one query per row; pair sits at distance d before the query key"""
    seq = torch.randint(FK + FV, V, (bs, L), generator=gen)
    keys = torch.randint(0, FK, (bs,), generator=gen)
    vals = FK + torch.randint(0, FV, (bs,), generator=gen)
    d = torch.randint(dlo, dhi + 1, (bs,), generator=gen).clamp(max=L - 2)
    rows = torch.arange(bs)
    qpos = torch.full((bs,), L - 1)
    ppos = qpos - d
    seq[rows, ppos] = keys
    seq[rows, ppos + 1] = vals
    seq[rows, qpos] = keys
    return seq, ppos + 1, vals          # true value position, target value


def builtin_mask(n, w, device):
    if w >= n:
        return None                     # is_causal handles it
    idx = torch.arange(n, device=device)
    return (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < w)


class Block(nn.Module):
    def __init__(self, d, h, w, use_rosa):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.rosa = ROSA(d_model=d, codebook_sizes=(8, 8), suffix_k=8, occurrences_r=2,
                         virtual_candidates=0, learned_residual_scale=0.1,
                         candidate_backend="auto") if use_rosa else None
        self.w, self.h, self.aux = w, h, {}

    def forward(self, x, qpos=None):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        m = builtin_mask(n, self.w, x.device)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=(m is None))
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        if self.rosa is not None:
            qp = qpos.to(x.device) if qpos is not None else None
            out = self.rosa(self.ln2(x), query_positions=qp)
            x = out.updated
            self.aux = {}
            if qp is not None:
                for key, field in (("ml", "hard_rosa_match_length"), ("si", "hard_rosa_source_index")):
                    t = getattr(out, field, None)
                    if t is not None:
                        self.aux["_" + key] = t.gather(1, qp).detach()
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self, d, h, layers, w, use_rosa):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, h, w, use_rosa) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V)
        self.aux_t = {}

    def forward(self, x, qpos=None):
        z = self.emb(x)
        self.aux_t = {}
        for b in self.blocks:
            z = b(z, qpos)
            self.aux_t.update(b.aux)
        return self.head(self.ln(z))


@torch.no_grad()
def evaluate(model, L, dist, gen, reps=3, bs=4):
    model.eval(); hit = tot = 0; loss_sum = 0.0; src_hit = src_n = 0
    for _ in range(reps):
        seq, vpos, tgt = make(bs, L, dist, dist, gen)
        qp = torch.full((bs, 1), L - 1)
        logits = model(seq.to(DEV), qp.to(DEV))
        lg = logits[:, -1, :]
        hit += (lg.argmax(-1) == tgt.to(DEV)).sum().item(); tot += bs
        loss_sum += F.cross_entropy(lg, tgt.to(DEV), reduction="sum").item()
        if "_si" in model.aux_t:
            src_hit += (model.aux_t["_si"].squeeze(1) == torch.tensor(vpos).to(DEV)).sum().item()
            src_n += bs
    model.train()
    return (hit / tot, loss_sum / tot, (src_hit / src_n if src_n else None))


def run(arm, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 100)
    w = EVAL_LEN if arm.startswith("full") else WIN
    model = Model(D_MODEL, HEADS, LAYERS, w, arm.endswith("rosa")).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=STEPS, pct_start=0.1)
    t0 = time.perf_counter(); curve = []
    for step in range(STEPS):
        seq, vpos, tgt = make(16, TRAIN_LEN, 32, 320, gen)
        qp = torch.full((16, 1), TRAIN_LEN - 1)
        lg = model(seq.to(DEV), qp.to(DEV))[:, -1, :]
        loss = F.cross_entropy(lg, tgt.to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % (STEPS // 5) == 0 or step == STEPS - 1:
            acc, ls, _ = evaluate(model, TRAIN_LEN, 200, gen, reps=1, bs=8)
            curve.append([round(ls, 3), round(acc, 3)])
    grid = {}
    for L, d in [(TRAIN_LEN, 200)] + [(EVAL_LEN, dd) for dd in DISTANCES]:
        acc, ls, src = evaluate(model, L, d, gen)
        grid[f"L{L}_d{d}"] = {"acc": round(acc, 3), "loss": round(ls, 3),
                              "src_pos_acc": (round(src, 3) if src is not None else None)}
    return dict(arm=arm, seed=seed, steps=STEPS, params=sum(p.numel() for p in model.parameters()),
                train_curve=curve, grid=grid, secs=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    print(f"device={DEV} | {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'}", flush=True)
    log({"design": {"train_len": TRAIN_LEN, "distances": DISTANCES, "eval_len": EVAL_LEN,
                    "window": WIN, "steps": STEPS, "seeds": SEEDS,
                    "chance_acc": round(1 / FV, 4), "no_pos_embeddings": True}})
    for arm in ("full", "window", "win+rosa", "full+rosa"):
        for s in range(SEEDS):
            try:
                log(run(arm, s))
            except Exception as e:
                log({"arm": arm, "seed": s, "error": repr(e)[:300]})
    log({"done": True})
