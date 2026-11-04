"""Conjuntos de datos etiquetados: sintético (con verdad por píxel) y desde carpeta."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config
from classification.color_classifier import DESCONOCIDO
from processing.feature_extraction import FeatureSet, compute_distances, extract_features
from processing.image_loader import load_image_bytes
from processing.preprocessing import prepare


@dataclass
class Sample:
    nombre: str
    fs: FeatureSet
    proporciones: Dict[int, float]            # índice de paleta (o -1) -> fracción real
    truth_map: Optional[np.ndarray] = None    # (N,) etiqueta real por píxel válido


def _sample_from_rgba(nombre, rgba, palette, truth_map=None, proporciones=None, origen="archivo"):
    prep = prepare(rgba, origen=origen, ignorar_fondo=False)
    fs = extract_features(prep)
    compute_distances(fs, palette)
    if truth_map is not None:
        vals, cnt = np.unique(truth_map, return_counts=True)
        proporciones = {int(v): float(c) / truth_map.size for v, c in zip(vals, cnt)}
    return Sample(nombre, fs, proporciones or {}, truth_map)


def generate_synthetic(palette, n: int = 60, seed: int = 7, size: int = 48,
                       sigma_region: float = 14.0, sigma_pixel: float = 7.0) -> List[Sample]:
    """
    Imágenes sintéticas con 1-4 regiones de colores de la paleta, con variación por región
    y ruido por píxel. La verdad (qué color es cada píxel) se conoce exactamente.
    """
    rng = np.random.default_rng(seed)
    muestras: List[Sample] = []
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        k = int(rng.integers(1, 5))
        clases = rng.choice(len(palette), size=k, replace=False)
        puntos = rng.uniform(0, size, size=(k, 2))
        dist = (xx[..., None] - puntos[:, 0]) ** 2 + (yy[..., None] - puntos[:, 1]) ** 2
        region = dist.argmin(axis=2)
        img = np.zeros((size, size, 3))
        verdad = np.zeros((size, size), dtype=np.int32)
        for r, cls in enumerate(clases):
            base = palette.rgb[cls] + rng.normal(0, sigma_region, 3)
            m = region == r
            img[m] = base
            verdad[m] = cls
        img += rng.normal(0, sigma_pixel, img.shape)
        rgba = np.dstack([np.clip(img, 0, 255).astype(np.uint8),
                          np.full((size, size), 255, np.uint8)])
        # prepare() conserva el orden fila-mayor de los píxeles válidos
        muestras.append(_sample_from_rgba(f"sintetica_{i:03d}", rgba, palette,
                                          truth_map=verdad.reshape(-1)))
    return muestras


def load_folder(carpeta: Path, palette) -> List[Sample]:
    """
    carpeta/etiquetas.json: {"foto1.png": {"Rojo": 0.6, "Azul": 0.4}, ...}
    La fracción que falte hasta 1.0 se considera 'Desconocido'.
    """
    carpeta = Path(carpeta)
    etiquetas_path = carpeta / "etiquetas.json"
    if not etiquetas_path.exists():
        raise FileNotFoundError(f"Falta {etiquetas_path}")
    etiquetas = json.loads(etiquetas_path.read_text(encoding="utf-8"))
    nombre_a_idx = {n: i for i, n in enumerate(palette.nombres)}
    muestras = []
    for archivo, props in etiquetas.items():
        ruta = carpeta / archivo
        if not ruta.exists():
            continue
        rgba = load_image_bytes(ruta.read_bytes())
        p = {}
        for nombre, frac in props.items():
            if nombre not in nombre_a_idx:
                raise ValueError(f"Color desconocido en etiquetas: {nombre!r}")
            p[nombre_a_idx[nombre]] = float(frac)
        resto = 1.0 - sum(p.values())
        if resto > 1e-6:
            p[DESCONOCIDO] = resto
        muestras.append(_sample_from_rgba(archivo, rgba, palette, proporciones=p))
    if not muestras:
        raise ValueError("No se encontraron imágenes etiquetadas en la carpeta.")
    return muestras


def split(muestras: List[Sample], ratio: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    orden = rng.permutation(len(muestras))
    corte = max(1, int(len(muestras) * ratio))
    return [muestras[i] for i in orden[:corte]], [muestras[i] for i in orden[corte:]]
