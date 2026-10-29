# -*- coding: utf-8 -*-
"""Синтетика капчи судов Карачаево-Черкесии: 105x33, серый фон, 5 цифр цвета (0,102,153)
разного размера и высоты (обрезаются краем), налезают друг на друга, 4-10 тонких цветных линий."""
import os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
# Шрифт, похожий на Arial (Arial, Liberation Sans, Arimo); путь — в KCHR_FONT
FONT = os.environ.get("KCHR_FONT", os.path.join(HERE, "arial.ttf"))
SS = 4                      # рисуем крупно, потом уменьшаем (сглаживание как у оригинала)
W, H = 105, 33
_fc = {}


def font(sz):
    if sz not in _fc:
        _fc[sz] = ImageFont.truetype(FONT, sz)
    return _fc[sz]


def rand_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def line(d):
    if random.random() < 0.8:      # почти горизонтальные через всю ширину
        x0, x1 = random.randint(-10, 40), random.randint(60, 115)
        y0 = random.uniform(0, H); y1 = y0 + random.uniform(-8, 8)
    else:
        x0, x1 = random.randint(-5, 50), random.randint(10, 110)
        y0, y1 = random.uniform(-5, H + 5), random.uniform(-5, H + 5)
    d.line([(x0 * SS, y0 * SS), (x1 * SS, y1 * SS)], fill=rand_color(), width=SS)


def gen(text=None):
    text = text or "".join(random.choice("0123456789") for _ in range(5))
    im = Image.new("RGB", (W * SS, H * SS), (243, 243, 243))
    d = ImageDraw.Draw(im)
    for _ in range(random.randint(0, 3)):
        line(d)
    x = random.uniform(-2, 14)
    col = (0, 102, 153)
    for ch in text:
        sz = int(random.triangular(16, 38, 30))
        f = font(sz * SS)
        bb = f.getbbox(ch)
        gw = bb[2] - bb[0]
        gh = (bb[3] - bb[1]) / SS
        cy = H / 2 + random.gauss(0, 3.5)            # центр цифры около середины, крупные режутся краем
        y = cy - gh / 2
        d.text((x * SS - bb[0], y * SS - bb[1]), ch, font=f, fill=col)
        x += gw / SS * random.uniform(0.55, 0.95)
    for _ in range(random.randint(4, 9)):
        line(d)
    return im.resize((W, H), Image.LANCZOS), text


if __name__ == "__main__":
    out = sys.argv[1]
    if out.endswith(".png"):
        random.seed(1)
        ims = [gen()[0] for _ in range(16)]
        sc = 3
        m = Image.new("RGB", (W * sc * 4 + 30, H * sc * 4 + 30), "white")
        for i, im in enumerate(ims):
            m.paste(im.resize((W * sc, H * sc), Image.NEAREST), ((i % 4) * (W * sc + 10), (i // 4) * (H * sc + 10)))
        m.save(out)
    else:
        n = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        random.seed(seed)
        xs, ys = [], []
        for _ in range(n):
            im, t = gen()
            xs.append(np.asarray(im.convert("L").resize((160, 32)), np.uint8)); ys.append(t)
        np.savez_compressed(out, x=np.stack(xs), y=np.array(ys, dtype="U8"), fam=np.array(["kchr"] * n, dtype="U4"))
