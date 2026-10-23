"""Real model: does exact retrieval help a FROZEN LM beyond its attention window?

  model   : Qwen2.5-0.5B, frozen (no parameter requires grad)
  context : sliding-window causal mask, WIN=512. Older tokens are invisible.
  sidecar : token-indexed suffix automaton + learned value + gate, injected as a
            residual at layer LAYER.  THE ONLY TRAINED PARAMETERS.
  corpus  : Python source shipped with the image (verbatim-repetitive: the
            mechanism's best case, which is the point)

Eval buckets by whether the verbatim match lies inside or beyond the window:
  ppl_in  : match within the window (attention can reach it)
  ppl_out : match beyond the window (attention cannot)
A real gain shows up in ppl_out without damaging ppl_in.
"""
import glob, json, os, random, time
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
T, WIN, STEPS = 1024, 512, 600
LAYER = 12
HOLDOUT = 40


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
    log({"stage": "load", "model": MODEL, "device": DEV,
         "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size
    train, test = load_corpus(tok)
    log({"hidden": d, "layers": model.config.num_hidden_layers, "T": T, "WIN": WIN,
         "train_chunks": len(train), "test_chunks": len(test)})

    idx = torch.arange(T)
    keep = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < WIN)
    mask = torch.zeros(1, 1, T, T, dtype=torch.float16, device=DEV)
    mask.masked_fill_(~keep.to(DEV), float("-inf"))

    side = Sidecar(d).to(DEV)
    holder = {"src": None}

    def hook(mod, args, output):
        h = args[0]
        sc = globals().get("_ACTIVE", None) or side
        add = sc(h.float(), holder["src"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def scramble(row, n):
        out = list(row)
        for j in range(n):
            if row[j] >= 0:
                out[j] = (int(row[j]) * 7919 + j * 104729) % max(1, j)
        return out

    def losses(seq, scrambled=False):
        row = build_src(seq)
        if scrambled:
            row = scramble(row, len(seq))
        holder["src"] = torch.tensor([row], dtype=torch.long)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        lg = logits[0, :-1].float()
        return F.cross_entropy(lg, ids[0, 1:], reduction="none"), holder["src"][0]

    # ---- smoke: does a gradient reach the sidecar and nowhere else?
    t0 = time.perf_counter()
    handle = model.model.layers[LAYER].register_forward_hook(hook)
    sidecar_ok = True
    try:
        with torch.enable_grad():
            l, _ = losses(train[0])
            l.mean().backward()
        log({"smoke": {"loss": round(l.mean().item(), 3),
                       "sidecar_grad": round(float(side.vproj.weight.grad.abs().mean()), 8),
                       "llm_params_with_grad": sum(1 for p in model.parameters() if p.grad is not None),
                       "hook_output_type": "tuple-or-tensor (type-preserving)",
                       "secs": round(time.perf_counter() - t0, 1)}})
    except Exception as e:
        sidecar_ok = False
        handle.remove()
        log({"smoke_failed": repr(e)[:300]})
    side_scrambled = Sidecar(d).to(DEV)

    # ---- train a real sidecar and a scrambled-source control sidecar
    if sidecar_ok:
        for tag, sc, scrambled in (("real", side, False), ("scrambled", side_scrambled, True)):
            sc.train(); rng = random.Random(1)
            opt = torch.optim.AdamW(sc.parameters(), lr=3e-3)
            for step in range(STEPS):
                loss, _ = losses(train[rng.randrange(len(train))], scrambled)
                loss = loss.mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0); opt.step()
                if step % 100 == 0:
                    log({"sidecar": tag, "step": step, "train_loss": round(loss.item(), 4),
                         "gate": round(sc.last["gate"].mean().item(), 4)})
        handle.remove()

    # ---- evaluate both arms with identical data
    results = {}
    for arm in (("no_sidecar", "sidecar", "scrambled") if sidecar_ok else ("no_sidecar",)):
        active = side if arm == "sidecar" else side_scrambled if arm == "scrambled" else None
        if active is not None:
            active.eval()
            globals()["_ACTIVE"] = active
        h = model.model.layers[LAYER].register_forward_hook(hook) if active is not None else None
        acc = {"in": [0.0, 0], "out": [0.0, 0]}; used = n = 0
        with torch.no_grad():
            for seq in test:
                loss, src = losses(seq, arm == "scrambled")
                lv = loss.tolist(); sv = src.tolist()
                for j, ls in enumerate(lv):          # loss[j] predicts token j+1
                    s = sv[j]                        # src[j] is that prediction's source
                    if s >= 0:
                        key = "in" if (j - s) <= WIN else "out"
                        acc[key][0] += ls; acc[key][1] += 1
                    used += (s >= 0); n += 1
        if h:
            h.remove()
        results[arm] = {k: {"ppl": round(float(torch.tensor(v[0] / max(v[1], 1)).exp()), 3),
                            "n": v[1]} for k, v in acc.items()}
        results[arm]["used_frac"] = round(used / max(n, 1), 3)
        log({"arm": arm, **results[arm]})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:400], "trace": traceback.format_exc()[-1200:]})
        raise
