"""ROSA + learnable vectors vs pure attention on associative recall (Kaggle GPU).

Task: [k1 v1 k2 v2 ... kn vn] [perm(k)]  -- predict the value that followed each
query key in the first half.  NPAIR=16 -> 48 tokens; window=12 in windowed arms,
so the pair is always outside the window.

arms
  full      causal attention over the whole sequence
  window    causal attention, window 12
  win+rosa  window + learnable ROSA retrieval module
  full+rosa full attention + learnable ROSA  (does it help even with long context?)

tasks
  copy  second-half value == the paired value      (copying the verbatim successor is correct)
  harm  second-half value re-assigned              (copying is always wrong: tests the read gate)
"""
import json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rosa-torch"], check=False)

import torch, torch.nn as nn, torch.nn.functional as F
from rosa import ROSA, init_inference_state, forward_step

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
V_KEY, V_VAL, NPAIR = 16, 16, 16
N, WINDOW = 3 * NPAIR, 12
TRAIN_BS, BUDGET_S = 24, 2.4 * 3600
STEPS, SEEDS = 700, 3   # STEPS may be re-chosen after timing the ROSA arm
V = V_KEY + V_VAL


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def sample(bs, task, gen):
    seq = torch.empty(bs, N, dtype=torch.long)
    qpos = torch.arange(2 * NPAIR, N).expand(bs, -1)
    vpos = torch.empty(bs, NPAIR, dtype=torch.long)
    tgt = torch.empty(bs, NPAIR, dtype=torch.long)
    for b in range(bs):
        keys = torch.randperm(V_KEY, generator=gen)[:NPAIR]
        vals = torch.randint(0, V_VAL, (NPAIR,), generator=gen)
        pair = torch.stack([keys, V_KEY + vals], 1).reshape(-1)
        order = torch.randperm(NPAIR, generator=gen)
        vpos[b] = 2 * order + 1        # where the correct value lives in the first half
        seq[b] = torch.cat([pair, keys[order]])
        if task == "copy":
            tgt[b] = V_KEY + vals[order]
        else:
            off = 1 + torch.randint(0, V_VAL - 1, (NPAIR,), generator=gen)
            tgt[b] = V_KEY + (vals[order] + off) % V_VAL
    return seq, qpos, tgt, vpos


def window_mask(n, w, device):
    idx = torch.arange(n, device=device)
    near = (idx[:, None] - idx[None, :]).abs() < w
    causal = idx[None, :] <= idx[:, None]
    return torch.where(near & causal, torch.zeros(n, n, device=device),
                       torch.full((n, n), float("-inf"), device=device))


class Block(nn.Module):
    def __init__(self, d, h, w, use_rosa):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.rosa = (ROSA(d_model=d, codebook_sizes=(8, 8), suffix_k=8, occurrences_r=2,
                          virtual_candidates=0, learned_residual_scale=0.1,
                          candidate_backend="auto") if use_rosa else None)
        self.w, self.aux = w, {}

    def forward(self, x, qpos=None):
        m = window_mask(x.shape[1], self.w, x.device)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=m, need_weights=False)
        x = x + a
        if self.rosa is not None:
            qp = qpos.to(x.device) if qpos is not None else None
            try:
                out = self.rosa(self.ln2(x), query_positions=qp)
            except Exception:
                out = self.rosa(self.ln2(x))
            x = out.updated
            g = out.read_gate
            self.aux = {}
            if g is not None:
                gv = (g.gather(1, qp[..., None].expand(-1, -1, g.shape[-1])).mean().item()
                      if qp is not None else g.mean().item())
                self.aux = {"gate": round(gv, 4)}
            for key, field in (("ml", "hard_rosa_match_length"), ("si", "hard_rosa_source_index")):
                t = getattr(out, field, None)
                if t is not None and qp is not None:
                    self.aux["_" + key] = t.gather(1, qp).detach()

    # (aux dict is consumed by Model.forward)
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self, d, h, layers, w, use_rosa):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(N, d)
        self.blocks = nn.ModuleList([Block(d, h, w, use_rosa) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V)
        self.aux = {}

    def forward(self, x, qpos=None):
        z = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        self.aux, self.aux_t = {}, {}
        for b in self.blocks:
            z = b(z, qpos)
            for k, v in b.aux.items():
                (self.aux_t if k.startswith("_") else self.aux)[k] = v
        return self.head(self.ln(z))


@torch.no_grad()
def evaluate(model, task, gen, bs=128, reps=4):
    model.eval(); hit = tot = 0; gate = []
    ml_hit = hp_hit = n = 0
    for _ in range(reps):
        seq, qpos, tgt, vpos = sample(bs, task, gen)
        logits = model(seq.to(DEV), qpos.to(DEV))
        pred = logits.gather(1, qpos[..., None].to(DEV).expand(-1, -1, V)).argmax(-1)
        hit += (pred == tgt.to(DEV)).sum().item(); tot += tgt.numel()
        if "gate" in model.aux:
            gate.append(model.aux["gate"])
        t = tgt.to(DEV)
        if "_si" in model.aux_t:
            ml_hit += (model.aux_t["_ml"] > 0).sum().item()
            hp_hit += (model.aux_t["_si"] == vpos.to(DEV)).sum().item(); n += t.numel()
    model.train()
    return (hit / tot, (sum(gate) / len(gate) if gate else None),
            (ml_hit / n if n else None), (hp_hit / n if n else None))


@torch.no_grad()
def rosa_oracle(task, gen, bs=32):
    hit = tot = 0
    for _ in range(bs):
        seq, qpos, tgt, vpos = sample(1, task, gen)
        st = init_inference_state(1, max_length=N + 8, backend="auto")
        preds = [int(forward_step(st, torch.tensor([t])).item()) for t in seq[0].tolist()]
        st.close()
        for j, p in enumerate(qpos[0].tolist()):
            hit += int(preds[p] == int(tgt[0, j])); tot += 1
    return hit / tot


def run(arm, task, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1)
    d, h, layers = 96, 4, 2
    w = N if arm.startswith("full") else WINDOW
    use = arm.endswith("rosa")
    model = Model(d, h, layers, w, use).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=STEPS, pct_start=0.1)
    t0 = time.perf_counter(); curve = []
    for step in range(STEPS):
        seq, qpos, tgt, vpos = sample(TRAIN_BS, task, gen)
        logits = model(seq.to(DEV), qpos.to(DEV))
        ql = logits.gather(1, qpos[..., None].to(DEV).expand(-1, -1, V))
        loss = F.cross_entropy(ql.reshape(-1, V), tgt.to(DEV).reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % (STEPS // 5) == 0 or step == STEPS - 1:
            acc = evaluate(model, task, gen, bs=64, reps=2)[0]
            curve.append(round(acc, 3))
    acc, gate, matched, cand_ok = evaluate(model, task, gen, reps=8)
    return dict(arm=arm, task=task, seed=seed, steps=STEPS, params=nparam, d=d,
                acc=round(acc, 4), read_gate=(round(gate, 4) if gate else None),
                matched_frac=(round(matched, 4) if matched is not None else None),
                retrieval_pos_acc=(round(cand_ok, 4) if cand_ok is not None else None),
                curve=curve, secs=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    print(f"device={DEV} torch={torch.__version__} | "
          f"{torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'}", flush=True)
    torch.manual_seed(0); g = torch.Generator().manual_seed(1)
    m = Model(96, 4, 2, WINDOW, True).to(DEV)
    seq, qpos, tgt, vpos = sample(8, "copy", g)
    lg = m(seq.to(DEV), qpos.to(DEV))
    ql = lg.gather(1, qpos[..., None].to(DEV).expand(-1, -1, V))
    F.cross_entropy(ql.reshape(-1, V), tgt.to(DEV).reshape(-1)).backward()
    gn = [p.grad.norm().item() for b in m.blocks if b.rosa is not None for p in b.rosa.parameters()
          if p.grad is not None]
    log({"smoke": "rosa_grad_norms", "n_with_grad": len(gn), "max": round(max(gn), 6) if gn else 0})

    # time 8 ROSA training steps (bs=TRAIN_BS), then size the run to the budget
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(8):
        seq, qpos, tgt, vpos = sample(TRAIN_BS, "copy", g)
        lg = m(seq.to(DEV), qpos.to(DEV))
        ql = lg.gather(1, qpos[..., None].to(DEV).expand(-1, -1, V))
        F.cross_entropy(ql.reshape(-1, V), tgt.to(DEV).reshape(-1)).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    t_step = (time.perf_counter() - t0) / 8
    plan = [("copy", "window", 0)]
    plan += [("copy", a, s_) for a in ("full", "win+rosa", "full+rosa") for s_ in range(3)]
    plan += [("harm", "full+rosa", 0)]
    n_rosa = sum(1 for _, a, _ in plan if a.endswith("rosa"))
    STEPS = 800
    log({"timing": {"rosa_step_s": round(t_step, 3), "train_bs": TRAIN_BS,
                    "n_rosa_runs": n_rosa, "chosen_steps": STEPS,
                    "est_hours": round(n_rosa * STEPS * t_step * 1.6 / 3600, 2)}})

    t_start = time.time()
    for task, arm, s_ in plan:
        try:
            log(run(arm, task, s_))
        except Exception as e:
            log({"arm": arm, "task": task, "seed": s_, "error": repr(e)[:300]})
        if time.time() - t_start > BUDGET_S:
            log({"stopped": "time budget"}); sys.exit(0)
    log({"done": True, "hours": round((time.time() - t_start) / 3600, 2)})
