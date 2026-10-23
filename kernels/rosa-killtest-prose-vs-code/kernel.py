"""Can the retrieval effect be killed?  Three tests in one run.

Corpora : code (torch .py) and PROSE (Gutenberg novels).  The mechanism should
          matter on code and largely vanish on prose -- if it doesn't, the effect
          was never about verbatim retrieval.
Arms    : none | real+real | scr+scr (scrambled control) | const+const (fixed
          source, i.e. a document-level vector with no position information)
Dose    : out-of-window ppl split by match length (short <16 tokens, long >=16).
          A real retrieval effect must grow with match length; a generic pathway
          effect will not care.
Buckets are computed ONCE from the true source and reused across all arms.
"""
import glob, json, os, random, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "/kaggle/working/results.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B"
T, WIN, LAYER, HOLDOUT = 1024, 512, 12, 40
STEPS = {"code": 400, "prose": 300}
BOOKS = ["https://www.gutenberg.org/files/2701/2701-0.txt",
         "https://www.gutenberg.org/files/1342/1342-0.txt",
         "https://www.gutenberg.org/files/1661/1661-0.txt",
         "https://www.gutenberg.org/files/84/84-0.txt"]


def log(d):
    line = json.dumps(d)
    print(line, flush=True)
    open(OUT, "a").write(line + "\n")


def build_src(x):
    """-> (src[i] = position of the predicted token at an earlier occurrence, match length)"""
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


def scramble(row):
    out = list(row)
    for j, s in enumerate(row):
        if s >= 0:
            out[j] = (int(s) * 7919 + j * 104729) % max(1, j)
    return out


def constant(row):
    return [0 if s >= 0 else -1 for s in row]


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
        self.last = {"gate": g.detach()}
        return out


def chunks_from(tok, texts):
    out = []
    for txt in texts:
        ids = tok(txt)["input_ids"]
        for s in range(0, len(ids) - T, T):
            out.append(ids[s:s + T])
    return out


def code_texts():
    pats = ["/usr/local/lib/python3*/dist-packages/torch",
            "/usr/local/lib/python3*/site-packages/torch",
            "/opt/conda/lib/python3*/site-packages/torch"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(p, "**", "*.py"), recursive=True)
    files = sorted({f for f in files if os.path.getsize(f) > 20000})
    random.Random(0).shuffle(files)
    out = []
    for f in files[:200]:
        try:
            out.append(open(f, encoding="utf-8", errors="ignore").read())
        except Exception:
            pass
    return out


def prose_texts():
    out = []
    for u in BOOKS:
        try:
            raw = urllib.request.urlopen(u, timeout=60).read().decode("utf-8", errors="ignore")
            out.append(raw)
        except Exception as e:
            log({"prose_fetch_failed": u, "err": repr(e)[:120]})
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log({"stage": "load", "model": MODEL, "device": DEV})
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float16).to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    d = model.config.hidden_size

    idx = torch.arange(T)
    keep = (idx[None, :] <= idx[:, None]) & ((idx[:, None] - idx[None, :]) < WIN)
    mask = torch.zeros(1, 1, T, T, dtype=torch.float16, device=DEV)
    mask.masked_fill_(~keep.to(DEV), float("-inf"))

    holder = {"src": None, "active": None}

    def hook(mod, args, output):
        h = args[0]
        add = holder["active"](h.float(), holder["src"].to(h.device)).to(h.dtype)
        if isinstance(output, tuple):
            return (output[0] + add,) + tuple(output[1:])
        return output + add

    def forward_seq(seq, row):
        holder["src"] = torch.tensor([row], dtype=torch.long)
        ids = torch.tensor([seq], device=DEV)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    corpora = {"code": chunks_from(tok, code_texts()), "prose": chunks_from(tok, prose_texts())}
    for name, ch in corpora.items():
        log({"corpus": name, "chunks": len(ch)})

    for name, ch in corpora.items():
        if len(ch) < 100:
            log({"corpus": name, "skipped": "too few chunks"}); continue
        train, test = ch[:-HOLDOUT], ch[-HOLDOUT:]
        evalset = [(s, build_src(s)) for s in test]      # ONE mask, reused by every arm
        arms = [("real", lambda r: list(r)), ("scrambled", scramble)]
        if name == "code":
            arms.append(("const", constant))
        built = {}
        for tag, fn in arms:
            sc = Sidecar(d).to(DEV); sc.train()
            holder["active"] = sc
            handle = model.model.layers[LAYER].register_forward_hook(hook)
            opt = torch.optim.AdamW(sc.parameters(), lr=3e-3)
            rng = random.Random(1)
            for step in range(STEPS[name]):
                seq = train[rng.randrange(len(train))]
                row = fn(build_src(seq)[0])
                loss = forward_seq(seq, row).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0); opt.step()
                if step % 150 == 0:
                    log({"corpus": name, "sidecar": tag, "step": step,
                         "loss": round(loss.item(), 4),
                         "gate": round(sc.last["gate"].mean().item(), 4)})
            handle.remove()
            sc.eval(); built[tag] = sc
        torch.save({k: v.state_dict() for k, v in built.items()}, f"/kaggle/working/{name}_sidecars.pt")

        for arm, sc in [("none", None)] + [(f"{t}+{t}", built[t]) for t, _ in arms]:
            h = None
            if sc is not None:
                holder["active"] = sc
                h = model.model.layers[LAYER].register_forward_hook(hook)
            tag = arm.split("+")[1] if "+" in arm else None
            used_row_fn = dict(arms)[tag] if tag else (lambda r: list(r))
            acc = {"in": [0.0, 0], "out": [0.0, 0], "out_short": [0.0, 0], "out_long": [0.0, 0]}
            with torch.no_grad():
                for seq, (true_row, mlen) in evalset:
                    row = used_row_fn(true_row)
                    for j, ls in enumerate(forward_seq(seq, row).tolist()):
                        s = true_row[j]
                        if s < 0:
                            continue
                        if (j - s) <= WIN:
                            acc["in"][0] += ls; acc["in"][1] += 1
                        else:
                            acc["out"][0] += ls; acc["out"][1] += 1
                            k = "out_long" if mlen[j] >= 16 else "out_short"
                            acc[k][0] += ls; acc[k][1] += 1
            if h:
                h.remove()
            res = {k: {"ppl": round(float(torch.tensor(v[0] / max(v[1], 1)).exp()), 3), "n": v[1]}
                   for k, v in acc.items() if v[1]}
            log({"corpus": name, "arm": arm, **res})
    log({"done": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log({"fatal": repr(e)[:400], "trace": traceback.format_exc()[-1000:]})
        raise
