"""CPU-reference colour space convertor (RGB→HSV→Grayscale) for Experiment 3."""

import math
from typing import Tuple


def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    cmax, cmin = max(r, g, b), min(r, g, b)
    delta = cmax - cmin
    v = cmax
    s = (delta / cmax) if cmax > 1e-9 else 0.0
    if delta < 1e-9:
        h = 0.0
    elif cmax == r:
        h = (60.0 * ((g - b) / delta) % 360) / 360.0
    elif cmax == g:
        h = (60.0 * ((b - r) / delta + 2)) / 360.0
    else:
        h = (60.0 * ((r - g) / delta + 4)) / 360.0
    return h, s, v


def rgb_to_grayscale(r: float, g: float, b: float) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    if s < 1e-9:
        return v, v, v
    h6 = h * 6.0
    i = int(h6) % 6
    f = h6 - int(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
