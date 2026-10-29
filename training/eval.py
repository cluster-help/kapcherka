"""Точность модели на npz (x, y, fam[, kind]) по семействам.

  python eval.py lite_best.pt tiny test.npz
"""
import sys, collections, numpy as np, torch
import train as tu
model, arch, npz = sys.argv[1], sys.argv[2], sys.argv[3]
ch, nh = tu.ARCH[arch]
m = tu.CRNN(tu.NCLASS, nh=nh, ch=ch).cuda().eval(); m.load_state_dict(torch.load(model, map_location="cuda"))
d = np.load(npz)
p = []
with torch.no_grad():
    for i in range(0, len(d["x"]), 256):
        p += tu.decode(m(torch.from_numpy(d["x"][i:i + 256]).cuda().float().unsqueeze(1) / 127.5 - 1).float())
acc = collections.defaultdict(lambda: [0, 0]); err = []
kinds = d["kind"] if "kind" in d.files else ["all"] * len(d["y"])
for pr, y, f, k in zip(p, d["y"], d["fam"], kinds):
    for key in (str(f), str(f) + ":" + str(k)):
        acc[key][0] += pr == y; acc[key][1] += 1
    if pr != y: err.append((str(f), str(k), str(y), pr))
for k in sorted(acc): print("%-12s %6.2f%%  (%d/%d)" % (k, 100 * acc[k][0] / acc[k][1], acc[k][0], acc[k][1]))
tot = sum(v[0] for k, v in acc.items() if ":" not in k), sum(v[1] for k, v in acc.items() if ":" not in k)
print("ВСЕГО %.2f%% (%d/%d)" % (100 * tot[0] / tot[1], *tot))
print("ошибки:", err[:40])
