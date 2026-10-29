# -*- coding: utf-8 -*-
"""Прогон модели по npz: ответ + уверенность (произведение max-вероятностей по кадрам жадного пути)."""
import sys, json, argparse
import numpy as np, torch
import train as tu

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("npz"); ap.add_argument("out")
ap.add_argument("--arch", default="big")
a = ap.parse_args()
ch, nh = tu.ARCH[a.arch]
m = tu.CRNN(tu.NCLASS, nh=nh, ch=ch).cuda().eval()
m.load_state_dict(torch.load(a.model, map_location="cuda"))
x = np.load(a.npz)["x"]
X = torch.from_numpy(x).cuda()
res = []
with torch.no_grad():
    for i in range(0, len(X), 1024):
        lo = m(X[i:i + 1024].float().unsqueeze(1) / 127.5 - 1).float().softmax(2)   # T,B,C
        mp, idx = lo.max(2)
        conf = mp.log().sum(0).exp().tolist()
        for s, c in zip(tu.decode(lo.log()), conf):
            res.append((s, c))
with open(a.out, "w", encoding="utf-8") as fo:
    for k, (s, c) in enumerate(res):
        fo.write(json.dumps({"idx": k, "pred": s, "conf": round(c, 5)}, ensure_ascii=False) + "\n")
print("pred", len(res))
