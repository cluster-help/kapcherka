"""Распознавание капчи: картинка -> строка.

Одна модель на все поддерживаемые виды капчи, тип определять не нужно.
Вход приводится к серому 160x32, выход — CTC по алфавиту из цифр и строчной кириллицы.
"""
from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = MODELS_DIR / "kapcherka-lite.onnx"
IMG_W, IMG_H = 160, 32

ImageInput = Union[bytes, bytearray, str, os.PathLike, Image.Image, np.ndarray]


@dataclass(frozen=True)
class Result:
    text: str
    confidence: float

    def __str__(self) -> str:
        return self.text


def _load_image(src: ImageInput) -> Image.Image:
    if isinstance(src, Image.Image):
        return src
    if isinstance(src, np.ndarray):
        return Image.fromarray(src)
    if isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(src)))
    else:
        img = Image.open(src)
    img.load()
    return img


def preprocess(src: ImageInput) -> np.ndarray:
    """Картинка -> тензор 1x1x32x160 float32 в диапазоне [-1, 1]."""
    g = _load_image(src).convert("L").resize((IMG_W, IMG_H))
    a = np.asarray(g, dtype=np.float32) / 255.0
    return ((a - 0.5) / 0.5)[None, None]


class Solver:
    """Решатель капчи на onnxruntime (CPU).

    Экземпляр потокобезопасен: один объект можно звать из многих потоков.
    threads — потоки onnxruntime на один вызов; для сервиса с параллельными
    запросами выгоднее 1, для одиночных вызовов можно больше.
    """

    def __init__(self, model_path: Union[str, os.PathLike, None] = None, threads: int = 1):
        import onnxruntime as ort

        path = Path(model_path) if model_path else DEFAULT_MODEL
        charset_file = path.with_name("charset.txt")
        if not charset_file.exists():
            charset_file = path.with_name(path.stem + "_charset.txt")
        self.charset = charset_file.read_text(encoding="utf-8").strip()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self.model_path = path

    def _decode(self, logits: np.ndarray) -> Result:
        # logits: T x 1 x C
        z = logits[:, 0, :].astype(np.float64)
        z = z - z.max(axis=1, keepdims=True)
        prob = np.exp(z)
        prob /= prob.sum(axis=1, keepdims=True)
        best = prob.argmax(axis=1)
        conf = float(np.exp(np.log(prob.max(axis=1)).sum()))
        out, prev = [], 0
        for i in best:
            i = int(i)
            if i != prev and i != 0:
                out.append(self.charset[i - 1])
            prev = i
        return Result("".join(out), conf)

    def solve(self, image: ImageInput) -> Result:
        """Распознать одну капчу. Возвращает Result(text, confidence)."""
        logits = self._sess.run(None, {self._input: preprocess(image)})[0]
        return self._decode(logits)

    def __call__(self, image: ImageInput) -> str:
        return self.solve(image).text


_default: Solver | None = None
_lock = threading.Lock()


def default_solver() -> Solver:
    global _default
    if _default is None:
        with _lock:
            if _default is None:
                _default = Solver()
    return _default


def solve(image: ImageInput) -> str:
    """Распознать капчу встроенной моделью: картинка (bytes, путь, PIL, numpy) -> строка."""
    return default_solver().solve(image).text
