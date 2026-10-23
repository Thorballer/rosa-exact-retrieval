"""Real model + retrieval control, v2 (buckets fixed).

Four arms, all bucketed by the TRUE source so every arm is measured on the same
positions; only the injected source differs:

  none            : baseline (512-token window, no sidecar)
  real+real       : sidecar trained on real matches, fed real matches
  scr+scr         : sidecar trained on scrambled sources, fed scrambled sources
                    (same architecture/steps/injection rate, retrieval destroyed)
  real+scr        : sidecar trained on real matches, fed scrambled sources
                    (isolates dependence on the match itself)

If scrambling (either in training or at eval) preserves most of the gain, the
gain is a learned-residual-pathway effect, not retrieval.
"""
import glob, json, os, random
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
T, WIN, STEPS = 1024, 512, 400
LAYER, HOLDOUT = 12, 40


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


def scramble(row):
    out = list(row)
    for j, s in enumerate(row):
        if s >= 0:
            out[j] = (int(s) * 7919 + j * 104729) % max(1, j)
    return out


class Sidecar(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.vproj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.last = {}

    def forward(self, h, src):
        hn = self.ln(h)
        gv = self.vproj(hn).gather(1, src.clamp(min=0)[..., None].expand(-1, -1, hn.shape[-1]))
        g = torch.sigmoid(self.gate(hn) + self.gate_bias)
        out = torch.where((src >= 0)[..., None], g * gv, torch.zeros_like(h))
        self.last = {"gate": g.detach(), "used": (src >= 0).float().detach()}
        return out


def load_corpus(tok):
    pats = ["/usr/local/lib/python3*/dist-packages/torch",
            "/usr/local/lib/python3*/site-packages/torch",
            "/opt/conda/lib/python3*/site-packages/torch"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(p, "**", "*.py"), recursive=True)
    files = sorted({f for f in files if os.path.getsize(f) > 20000})
    random.Random(0).shuffle(files)
    files = files[:200]
    chunks = []
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        ids = tok(txt)["input_ids"]
        for s in range(0, len(ids) - T, T):
            chunks.append(ids[s:s + T])
    return chunks[:-HOLDOUT], chunks[-HOLDOUT:]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log({"stage": "load", "model": MODEL, "device": DEV, "STEPS": STEPS})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size
    train, test = load_corpus(tok)
    log({"hidden": d, "layers": model.config.num_hidden_layers, "WIN": WIN,
         "train_chunks": len(train), "test_chunks": len(test)})

    idx = torch.arange(T)
    keep = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < WIN)
    mask = torch.zeros(1, 1, T, T, dtype=torch.float16, device=DEV)
    mask.masked_fill_(~keep.to(DEV), float("-inf"))

    real, ctrl = Sidecar(d).to(DEV), Sidecar(d).to(DEV)
    holder = {"src": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h.float(), holder["src"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def forward_seq(seq, used_row):
        holder["src"] = torch.tensor([used_row], dtype=torch.long)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    sidecar_ok = True
    holder["active"] = real
    handle = model.model.layers[LAYER].register_forward_hook(hook)
    row0 = build_src(train[0])
    try:
        with torch.enable_grad():
            l = forward_seq(train[0], row0)
            l.mean().backward()
        log({"smoke": {"loss": round(l.mean().item(), 3),
                       "sidecar_grad": round(float(real.vproj.weight.grad.abs().mean()), 8),
                       "llm_params_with_grad": sum(1 for p in model.parameters() if p.grad is not None)}})
    except Exception as e:
        sidecar_ok = False
        handle.remove()
        log({"smoke_failed": repr(e)[:300]})

    if sidecar_ok:
        for tag, sc, scrambled in (("real", real, False), ("scrambled", ctrl, True)):
            sc.train(); rng = random.Random(1)
            opt = torch.optim.AdamW(sc.parameters(), lr=3e-3)
            holder["active"] = sc
            for step in range(STEPS):
                seq = train[rng.randrange(len(train))]
                row = build_src(seq)
                if scrambled:
                    row = scramble(row)
                loss = forward_seq(seq, row).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0); opt.step()
                if step % 100 == 0:
                    log({"sidecar": tag, "step": step, "train_loss": round(loss.item(), 4),
                         "gate": round(sc.last["gate"].mean().item(), 4)})
        handle.remove()
        torch.save({"real": real.state_dict(), "ctrl": ctrl.state_dict()},
                   "/kaggle/working/sidecars.pt")

    arms = ([("none", None, False), ("real+real", real, False),
             ("scr+scr", ctrl, True), ("real+scr", real, True)] if sidecar_ok
            else [("none", None, False)])
    for arm, sc, scramble_eval in arms:
        h = None
        if sc is not None:
            sc.eval(); holder["active"] = sc
            h = model.model.layers[LAYER].register_forward_hook(hook)
        acc = {"in": [0.0, 0], "out": [0.0, 0]}; used = n = 0
        with torch.no_grad():
            for seq in test:
                true_row = build_src(seq)
                used_row = scramble(true_row) if scramble_eval else true_row
                loss = forward_seq(seq, used_row).tolist()
                for j, ls in enumerate(loss):
                    s = true_row[j]           # bucket by the TRUE source for every arm
                    if s >= 0:
                        key = "in" if (j - s) <= WIN else "out"
                        acc[key][0] += ls; acc[key][1] += 1
                    used += (s >= 0); n += 1
        if h:
            h.remove()
        row = {k: {"ppl": round(float(torch.tensor(v[0] / max(v[1], 1)).exp()), 3),
                   "n": v[1]} for k, v in acc.items()}
        row.update(arm=arm, used_frac=round(used / max(n, 1), 3))
        log(row)
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:400], "trace": traceback.format_exc()[-1200:]})
        raise
