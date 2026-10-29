# -*- coding: utf-8 -*-
"""Обучение модели Капчерки: CRNN + CTC, одна модель на все виды капчи.

Вход: серое 160x32, (x/255-0.5)/0.5 — так же, как в kapcherka.preprocess.
Набор целиком в памяти GPU, аугментация на GPU, выборка с весами по семействам (FAM_W).
Формат данных — npz: x (N,32,160) uint8, y (N,) строки ответов, fam (N,) метка семейства.

  python train.py --arch big --train base.npz --extra hard.npz:4 --test test.npz --steps 6000 --out big
  python train.py --arch tiny --train base.npz --test test.npz --teacher big_best.pt --steps 16000 --out lite
"""
import os, time, math, argparse, collections, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHARSET = "0123456789абвгдежзийклмнопрстуфхцчшщъыьэюя"
C2I = {c: i + 1 for i, c in enumerate(CHARSET)}
I2C = {i + 1: c for i, c in enumerate(CHARSET)}
NCLASS = len(CHARSET) + 1
IMG_H, IMG_W = 32, 160
FED_CHARSET = "0123456789абгекнрстуфхя"
FAM_W = {"dig": 0.18, "mir": 0.18, "cyr": 0.2, "23os": 0.14, "kchr": 0.3}


class BiLSTM(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.rnn = nn.LSTM(i, h, bidirectional=True)
        self.fc = nn.Linear(h * 2, o)

    def forward(self, x):
        r, _ = self.rnn(x)
        T, B, H = r.size()
        return self.fc(r.reshape(T * B, H)).view(T, B, -1)


ARCH = {"big": ((64, 128, 256, 256, 512, 512, 512), 256),
        "mid": ((32, 64, 128, 128, 192, 192, 192), 128),
        "slim": ((32, 64, 96, 96, 128, 128, 128), 128),
        "tiny": ((16, 32, 64, 64, 96, 96, 96), 96)}


class CRNN(nn.Module):
    def __init__(self, nclass, nh=256, ch=(64, 128, 256, 256, 512, 512, 512)):
        super().__init__()
        c1, c2, c3, c4, c5, c6, c7 = ch

        def cbr(i, o, k=3, s=1, p=1, bn=False):
            L = [nn.Conv2d(i, o, k, s, p)]
            if bn: L.append(nn.BatchNorm2d(o))
            L.append(nn.ReLU(True))
            return L
        self.cnn = nn.Sequential(
            *cbr(1, c1), nn.MaxPool2d(2, 2),
            *cbr(c1, c2), nn.MaxPool2d(2, 2),
            *cbr(c2, c3, bn=True), *cbr(c3, c4), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            *cbr(c4, c5, bn=True), *cbr(c5, c6), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            *cbr(c6, c7, k=2, s=1, p=0, bn=True),
        )
        self.rnn = nn.Sequential(BiLSTM(c7, nh, nh), BiLSTM(nh, nh, nclass))

    def forward(self, x):
        c = self.cnn(x)
        c = c.squeeze(2).permute(2, 0, 1)
        return self.rnn(c)


def init_from(model, path, init_cs):
    sd = torch.load(path, map_location="cpu")
    if "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
    w, b = sd["rnn.1.fc.weight"], sd["rnn.1.fc.bias"]
    own = model.state_dict()
    for k, v in sd.items():
        if not k.startswith("rnn.1.fc.") and k in own and own[k].shape == v.shape:
            own[k] = v
    if w.shape[0] == NCLASS and init_cs == CHARSET:
        own["rnn.1.fc.weight"], own["rnn.1.fc.bias"] = w, b
    else:
        nw, nb = own["rnn.1.fc.weight"], own["rnn.1.fc.bias"]
        nw[0], nb[0] = w[0], b[0]
        for c, i in C2I.items():
            if c in init_cs:
                j = init_cs.index(c) + 1
                nw[i], nb[i] = w[j], b[j]
    model.load_state_dict(own)
    print("init from", path, flush=True)


def enc(labels, maxlen=8):
    t = np.zeros((len(labels), maxlen), np.int64)
    ln = np.zeros(len(labels), np.int64)
    for k, s in enumerate(labels):
        s = s[:maxlen]
        ln[k] = len(s)
        t[k, :len(s)] = [C2I[c] for c in s]
    return t, ln


def load_npz(path):
    d = np.load(path)
    ok = np.array([len(s) > 0 and all(c in C2I for c in s) for s in d["y"]])
    return d["x"][ok], d["y"][ok], d["fam"][ok]


def augment(x):
    """x: B,1,H,W float в [-1,1] на GPU."""
    B = x.size(0)
    dev = x.device
    ang = (torch.rand(B, device=dev) - 0.5) * 2 * math.radians(4)
    sc = 1 + (torch.rand(B, device=dev) - 0.5) * 0.12
    tx = (torch.rand(B, device=dev) - 0.5) * 2 * 0.05
    ty = (torch.rand(B, device=dev) - 0.5) * 2 * 0.10
    sh = (torch.rand(B, device=dev) - 0.5) * 0.2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin * (IMG_H / IMG_W) + sh, tx], 1),
                         torch.stack([sin * (IMG_W / IMG_H), cos, ty], 1)], 1)
    keep = torch.rand(B, device=dev) < 0.25          # четверть без геометрии
    eye = torch.tensor([[1, 0, 0], [0, 1, 0]], device=dev, dtype=theta.dtype)
    theta[keep] = eye
    g = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, g, padding_mode="border", align_corners=False)
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.5
    br = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4
    x = x * c + br
    nz = torch.rand(B, 1, 1, 1, device=dev) * 0.12
    x = x + torch.randn_like(x) * nz
    inv = torch.rand(B, device=dev) < 0.05
    x[inv] = -x[inv]
    return x.clamp(-1.5, 1.5)


def decode(logits):
    out = []
    for seq in logits.argmax(2).permute(1, 0).tolist():
        prev, s = 0, []
        for i in seq:
            if i != prev and i != 0: s.append(I2C.get(i, ""))
            prev = i
        out.append("".join(s))
    return out


@torch.no_grad()
def evaluate(model, X, Y, FAM, bs=1024):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].float().unsqueeze(1) / 127.5 - 1
        preds += decode(model(xb).float())
    model.train()
    acc = collections.defaultdict(lambda: [0, 0])
    for p, y, f in zip(preds, Y, FAM):
        acc[f][0] += p == y
        acc[f][1] += 1
    return {f: a[0] / a[1] for f, a in acc.items()}, {f: a[1] for f, a in acc.items()}, preds


def export_onnx(model, path):
    model = model.float().cpu().eval()
    dummy = torch.randn(1, 1, IMG_H, IMG_W)
    kw = {}
    try:
        torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"],
                          opset_version=13, dynamo=False)
    except TypeError:
        torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"], opset_version=13)
    import onnx
    m = onnx.load(path)
    onnx.save_model(m, path, save_as_external_data=False)
    for f in os.listdir(os.path.dirname(os.path.abspath(path))):
        if f.startswith(os.path.basename(path)) and f.endswith(".data"):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(path)), f))
    open(os.path.splitext(path)[0] + "_charset.txt", "w", encoding="utf-8").write(CHARSET)
    print("onnx ->", path, os.path.getsize(path), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--extra", action="append", default=[], help="file.npz:повтор (трудные примеры)")
    ap.add_argument("--test", required=True)
    ap.add_argument("--exclude", action="append", default=[], help="npz, чьи картинки убрать из обучения (проверочная выборка)")
    ap.add_argument("--init", default="")
    ap.add_argument("--arch", default="big", choices=list(ARCH))
    ap.add_argument("--teacher", default="", help="веса учителя (дистилляция по кадрам CTC)")
    ap.add_argument("--teacher-arch", default="big")
    ap.add_argument("--kd", type=float, default=1.0, help="вес KL к учителю")
    ap.add_argument("--kd-t", type=float, default=2.0, help="температура")
    ap.add_argument("--init-charset", default=FED_CHARSET)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-minutes", type=float, default=25)
    ap.add_argument("--out", default="uni_v1")
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    parts = [load_npz(p) for p in a.train]
    x = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    fam = np.concatenate([p[2] for p in parts])
    w = np.ones(len(fam), np.float64)                 # сырой вес примера; доля семейства — FAM_W
    xs, ys, ws, fams = [x], [y], [w], [fam]
    for spec in a.extra:
        p, rep = (spec.split(":") + ["3"])[:2]
        ex, ey, ef = load_npz(p)
        xs.append(ex); ys.append(ey); ws.append(np.full(len(ex), float(rep))); fams.append(ef)
        print("extra", p, len(ex), dict(collections.Counter(ef.tolist())), "x", rep, flush=True)
    x = np.concatenate(xs); y = np.concatenate(ys); w = np.concatenate(ws); fam = np.concatenate(fams)
    tx, ty, tf = load_npz(a.test)
    th = {hashlib.md5(r.tobytes()).digest() for r in (np.concatenate([tx] + [load_npz(s)[0] for s in a.exclude]) if a.exclude else tx)}
    keep = np.array([hashlib.md5(r.tobytes()).digest() not in th for r in x])
    print("train", len(x), "убрано совпадений с тестом", int((~keep).sum()), flush=True)
    x, y, w, fam = x[keep], y[keep], w[keep], fam[keep]
    for f_ in set(fam.tolist()):                      # нормировка: сумма весов семейства = FAM_W
        m_ = fam == f_
        w[m_] = w[m_] / w[m_].sum() * FAM_W.get(f_, 0.1)
    print("доли внутри семейств:", {f_: round(float(w[fam == f_].sum()), 3) for f_ in set(fam.tolist())}, flush=True)
    print("train", len(x), dict(collections.Counter(fam.tolist())), flush=True)
    X = torch.from_numpy(x).to(dev)
    tgt, tln = enc(list(y))
    TGT = torch.from_numpy(tgt).to(dev); TLN = torch.from_numpy(tln).to(dev)
    W = torch.from_numpy(w / w.sum()).to(dev)

    TX = torch.from_numpy(tx).to(dev)
    print("test", len(tx), dict(collections.Counter(tf.tolist())), flush=True)

    ch, nh = ARCH[a.arch]
    model = CRNN(NCLASS, nh=nh, ch=ch).to(dev)
    print("arch", a.arch, "params", sum(p.numel() for p in model.parameters()), flush=True)
    if a.init:
        init_from(model, a.init, a.init_charset)
        model.to(dev)
    teacher = None
    if a.teacher:
        tch, tnh = ARCH[a.teacher_arch]
        teacher = CRNN(NCLASS, nh=tnh, ch=tch).to(dev).eval()
        teacher.load_state_dict(torch.load(a.teacher, map_location=dev))
        for p_ in teacher.parameters(): p_.requires_grad_(False)
        print("teacher", a.teacher, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.08)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time(); best = -1; run = 0.0
    acc0, n0, _ = evaluate(model, TX, ty, tf)
    print("step 0", {k: round(v * 100, 2) for k, v in acc0.items()}, flush=True)
    for step in range(1, a.steps + 1):
        idx = torch.multinomial(W, a.bs, replacement=True)
        xb = X[idx].float().unsqueeze(1) / 127.5 - 1
        xb = augment(xb)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(xb)
        lp = out.float().log_softmax(2)
        T = lp.size(0)
        loss = ctc(lp, TGT[idx], torch.full((a.bs,), T, dtype=torch.long, device=dev), TLN[idx])
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tout = teacher(xb)
            tt = a.kd_t
            kl = F.kl_div((out.float() / tt).log_softmax(2), (tout.float() / tt).softmax(2), reduction="sum") * tt * tt / (T * a.bs)
            loss = loss + a.kd * kl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        last = step == a.steps or (time.time() - t0) / 60 > a.max_minutes
        if step % a.eval_every == 0 or last:
            acc, n, _ = evaluate(model, TX, ty, tf)
            score = sum(acc.values()) / len(acc)
            print("step %d loss %.4f %.0fs %s mean %.2f" % (step, run, time.time() - t0,
                  {k: round(v * 100, 2) for k, v in sorted(acc.items())}, score * 100), flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), a.out + "_best.pt")
        if last:
            break
    model.load_state_dict(torch.load(a.out + "_best.pt"))
    acc, n, preds = evaluate(model, TX, ty, tf)
    print("BEST", {k: (round(v * 100, 2), n[k]) for k, v in sorted(acc.items())}, flush=True)
    bad = [(f, yy, p) for p, yy, f in zip(preds, ty, tf) if p != yy]
    with open(a.out + "_test_errors.tsv", "w", encoding="utf-8") as fo:
        for f, yy, p in bad: fo.write("%s\t%s\t%s\n" % (f, yy, p))
    export_onnx(model, a.out + ".onnx")


if __name__ == "__main__":
    main()
