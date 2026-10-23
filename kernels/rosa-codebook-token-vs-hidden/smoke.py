import torch, kernel as K

K.OUT = "/tmp/cb.jsonl"
K.N_TRAIN, K.N_EVAL, K.STEPS, K.BS = 24, 16, 5, 2
TR, EV = K.make_set(K.N_TRAIN, 1), K.make_set(K.N_EVAL, 2)
print("oracle src_pos:", round((EV["src"].gather(1, EV["qpos"]) == EV["vpos"]).float().mean().item(), 3))
for mode, k in [("raw", 0), ("tok", 64), ("hid", 256)]:
    try:
        r = K.run(mode, k, 0, TR, EV)
        print(f"{r['index']:9s} acc {r['acc']:.3f} idx_pos {r['idx_pos_acc']:.3f} "
              f"oracle {r['oracle_src_acc']} gate {r['gate_at_q']} codes_used {r['n_codes_used']} "
              f"params {r['params']} secs {r['secs']}")
    except Exception as e:
        print(mode, k, "ERROR", repr(e)[:200])
