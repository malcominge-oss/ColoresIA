"""Normalización de la imagen: reducción, máscara de píxeles válidos y coordenadas."""
from dataclasses import dataclass

import numpy as np
from PIL import Image

import config


class ImagenSinColores(ValueError):
    """No quedan píxeles válidos que analizar (lienzo vacío, todo fondo o transparente)."""


@dataclass
class PreparedImage:
    rgb: np.ndarray            # (N, 3) uint8, solo píxeles válidos
    xy: np.ndarray             # (N, 2) float32 en [0, 1]
    ancho: int
    alto: int
    total_pixels: int
    transparentes: int
    fondo: int
    origen: str
    ignorar_fondo: bool

    @property
    def n_valid(self) -> int:
        return int(self.rgb.shape[0])


def prepare(rgba: np.ndarray, origen: str = "archivo", ignorar_fondo=None,
            max_side: int = config.MAX_IMAGE_SIDE) -> PreparedImage:
    """
    origen: 'dibujo' (canvas) o 'archivo'. Ambos usan el mismo pipeline.
    ignorar_fondo: None -> True para dibujo, False para archivo.
    """
    if ignorar_fondo is None:
        ignorar_fondo = origen == "dibujo"
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ImagenSinColores("Formato de arreglo de imagen inesperado.")

    alto, ancho = rgba.shape[:2]
    if max(alto, ancho) > max_side:
        escala = max_side / max(alto, ancho)
        nuevo = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
        metodo = Image.NEAREST if origen == "dibujo" else Image.BOX
        rgba = np.array(Image.fromarray(rgba, "RGBA").resize(nuevo, metodo), dtype=np.uint8)
        alto, ancho = rgba.shape[:2]

    alpha_ok = rgba[..., 3] >= config.ALPHA_MIN
    rgb = rgba[..., :3]
    es_fondo = np.all(rgb >= config.BACKGROUND_MIN_CHANNEL, axis=2) & alpha_ok
    valido = alpha_ok & (~es_fondo if ignorar_fondo else True)

    n_valid = int(valido.sum())
    if n_valid == 0:
        raise ImagenSinColores("No hay colores para analizar: la imagen está vacía o solo tiene fondo.")

    ys, xs = np.nonzero(valido)
    xy = np.stack([xs / max(ancho - 1, 1), ys / max(alto - 1, 1)], axis=1).astype(np.float32)
    return PreparedImage(
        rgb=rgb[valido].astype(np.uint8), xy=xy, ancho=ancho, alto=alto,
        total_pixels=ancho * alto,
        transparentes=int((~alpha_ok).sum()),
        fondo=int(es_fondo.sum()) if ignorar_fondo else 0,
        origen=origen, ignorar_fondo=bool(ignorar_fondo),
    )
