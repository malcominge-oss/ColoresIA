"""Paleta de colores de referencia (configurable)."""
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import config
from classification.color_metrics import hex_to_rgb, rgb_to_hsv, rgb_to_lab, hsv_efectivo


@dataclass(frozen=True)
class PaletteColor:
    nombre: str
    hex: str
    rgb: Tuple[int, int, int]
    hsv: Tuple[float, float, float]
    lab: Tuple[float, float, float]
    tolerancia: float            # factor multiplicativo sobre la tolerancia global


class Palette:
    def __init__(self, definiciones):
        colores: List[PaletteColor] = []
        for nombre, hx, tol in definiciones:
            rgb = hex_to_rgb(hx)
            hsv = rgb_to_hsv(np.array(rgb))
            lab = rgb_to_lab(np.array(rgb))
            colores.append(PaletteColor(nombre, hx.upper(), rgb,
                                        tuple(float(v) for v in hsv),
                                        tuple(float(v) for v in lab), float(tol)))
        self.colores = colores
        self.nombres = [c.nombre for c in colores]
        self.rgb = np.array([c.rgb for c in colores], dtype=np.float64)
        self.hsv = np.array([c.hsv for c in colores], dtype=np.float64)
        self.hsv_ef = hsv_efectivo(self.hsv)
        self.lab = np.array([c.lab for c in colores], dtype=np.float64)
        self.tol = np.array([c.tolerancia for c in colores], dtype=np.float64)

    def __len__(self):
        return len(self.colores)

    def to_list(self):
        return [{"nombre": c.nombre, "hex": c.hex, "rgb": list(c.rgb),
                 "hsv": [round(v, 4) for v in c.hsv],
                 "lab": [round(v, 3) for v in c.lab],
                 "tolerancia": c.tolerancia} for c in self.colores]


def load_palette() -> Palette:
    """Usa models/palette.json si existe: [{"nombre":..., "hex":..., "tolerancia":1.0}, ...]."""
    if config.CUSTOM_PALETTE_PATH.exists():
        datos = json.loads(config.CUSTOM_PALETTE_PATH.read_text(encoding="utf-8"))
        return Palette([(d["nombre"], d["hex"], d.get("tolerancia", 1.0)) for d in datos])
    return Palette(config.PALETTE_COLORS)
