import torch, kernel as K

K.OUT = "/tmp/gate.jsonl"
K.N_TRAIN, K.N_EVAL, K.STEPS, K.BS = 32, 16, 5, 4
TR = K.make_set(K.N_TRAIN, 1, "mixed")
EV_C, EV_H = K.make_set(K.N_EVAL, 2, "copy"), K.make_set(K.N_EVAL, 3, "harm")
print("flags copy/harm:", int((TR["is_copy"]).sum()), int((~TR["is_copy"]).sum()))
print("flag token written:", int((TR["toks"][:, K.Q_MIN - 1] == K.FLAG_COPY).sum()),
      int((TR["toks"][:, K.Q_MIN - 1] == K.FLAG_HARM).sum()))
for name, ds in (("copy", EV_C), ("harm", EV_H)):
    srcok = (ds["src"].gather(1, ds["qpos"]) == ds["vpos"]).float().mean().item()
    verb = (ds["tgt"] == ds["toks"].gather(1, ds["vpos"])).float().mean().item()
    print(f"{name}: oracle src_pos {srcok:.3f} | label==verbatim successor {verb:.3f}")
for arm in ("full", "win+trosa"):
    r = K.run(arm, 0, TR, EV_C, EV_H)
    print(arm, "curve", r["train_loss_curve"])
    print("   copy:", r["eval_copy"], "\n   harm:", r["eval_harm"])
