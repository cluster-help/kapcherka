"""Командная строка: kapcherka файл [файл ...] | kapcherka bench | kapcherka serve."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import __version__
from .solver import Solver


def _cmd_solve(a) -> int:
    s = Solver(a.model, threads=a.threads)
    rc = 0
    for path in a.files:
        try:
            r = s.solve(path)
        except Exception as e:  # noqa: BLE001
            print(f"{path}\tошибка: {e}", file=sys.stderr)
            rc = 1
            continue
        if a.json:
            print(json.dumps({"file": path, "text": r.text, "confidence": round(r.confidence, 4)}, ensure_ascii=False))
        else:
            print(f"{path}\t{r.text}\t{r.confidence:.3f}" if len(a.files) > 1 or a.confidence else r.text)
    return rc


def _cmd_bench(a) -> int:
    import numpy as np

    s = Solver(a.model, threads=a.threads)
    img = (np.random.rand(30, 100) * 255).astype("uint8")
    for _ in range(20):
        s.solve(img)
    ts = []
    for _ in range(a.n):
        t = time.perf_counter()
        s.solve(img)
        ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"капч: {a.n}, потоков: {a.threads}")
    print(f"медиана {ts[len(ts) // 2] * 1000:.2f} мс, p90 {ts[int(len(ts) * 0.9)] * 1000:.2f} мс, "
          f"{a.n / sum(ts):.0f} капч/с на поток")
    return 0


def _cmd_serve(a) -> int:
    from .server import serve

    serve(a.host, a.port, a.model, a.threads)
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help="путь к своей модели .onnx (рядом charset.txt)")
    common.add_argument("--threads", type=int, default=1, help="потоки onnxruntime на вызов (по умолчанию 1)")
    ap = argparse.ArgumentParser(prog="kapcherka", description="Распознавание капчи сайтов судов РФ")
    ap.add_argument("--version", action="version", version=f"kapcherka {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("solve", parents=[common], help="распознать картинки")
    p.add_argument("files", nargs="+")
    p.add_argument("--json", action="store_true", help="вывод строками JSON")
    p.add_argument("--confidence", action="store_true", help="печатать уверенность")
    p.set_defaults(func=_cmd_solve)

    p = sub.add_parser("bench", parents=[common], help="замер скорости")
    p.add_argument("-n", type=int, default=500)
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("serve", parents=[common], help="HTTP-сервис")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in ("solve", "bench", "serve", "-h", "--help", "--version"):
        argv = ["solve", *argv]                      # kapcherka file.png == kapcherka solve file.png
    a = ap.parse_args(argv)
    if not getattr(a, "func", None):
        ap.print_help()
        return 2
    return a.func(a)
