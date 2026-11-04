"""Conversiones de espacios de color y métricas de diferencia cromática."""
import numpy as np
from skimage import color as skcolor


def rgb_to_lab(rgb) -> np.ndarray:
    """RGB (0-255, forma (...,3)) -> CIELAB."""
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    return skcolor.rgb2lab(np.clip(arr, 0.0, 1.0))


def rgb_to_hsv(rgb) -> np.ndarray:
    """RGB (0-255) -> HSV con H, S, V en [0, 1]."""
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    return skcolor.rgb2hsv(np.clip(arr, 0.0, 1.0))


def hsv_efectivo(hsv) -> np.ndarray:
    """Atenúa la saturación de colores muy oscuros (el tono no es fiable cerca del negro)."""
    hsv = np.array(hsv, dtype=np.float64, copy=True)
    hsv[..., 1] = hsv[..., 1] * np.clip(hsv[..., 2] / 0.25, 0.0, 1.0)
    return hsv


def delta_e(lab1, lab2) -> np.ndarray:
    """Diferencia cromática CIEDE2000 (admite broadcasting)."""
    return skcolor.deltaE_ciede2000(np.asarray(lab1, dtype=np.float64),
                                    np.asarray(lab2, dtype=np.float64))


def similitud(delta, tolerancia):
    """C = 100 * max(0, 1 - deltaE / T). Puntuación de similitud, no una probabilidad."""
    return 100.0 * np.maximum(0.0, 1.0 - np.asarray(delta) / np.asarray(tolerancia))


def rgb_to_hex(rgb) -> str:
    r, g, b = [int(round(float(v))) for v in rgb]
    r, g, b = [max(0, min(255, v)) for v in (r, g, b)]
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
