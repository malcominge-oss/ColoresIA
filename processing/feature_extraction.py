"""Red de extracción de rasgos: RGB, HSV, CIELAB, histogramas y distribución espacial."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import config
from classification.color_metrics import (delta_e, hsv_efectivo, rgb_to_hsv,
                                           rgb_to_lab)
from processing.color_clustering import cluster_pixels
from processing.preprocessing import PreparedImage


@dataclass
class Cluster:
    idx: int
    n: int
    fraccion: float
    rgb: np.ndarray
    hsv: np.ndarray
    hsv_ef: np.ndarray
    lab: np.ndarray
    hist: np.ndarray            # histograma de tono (ponderado por saturación)
    lab_sample: np.ndarray      # muestra de píxeles en LAB (distribución del grupo)
    centroide: tuple
    dispersion: float


@dataclass
class FeatureSet:
    clusters: List[Cluster]
    pixel_cluster: np.ndarray           # (N,) grupo de cada píxel válido
    n_valid: int
    vector_global: np.ndarray
    distances: Optional[np.ndarray] = field(default=None)   # (4, K, C)


def _hist_tono(hsv: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(hsv[:, 0], bins=config.HUE_BINS, range=(0, 1), weights=hsv[:, 1])
    s = h.sum()
    return h / s if s > 0 else h


def extract_features(prep: PreparedImage, kmax: int = config.KMEANS_MAX_K, seed: int = 0) -> FeatureSet:
    rgb = prep.rgb
    lab = rgb_to_lab(rgb)
    hsv = rgb_to_hsv(rgb)
    etiquetas = cluster_pixels(lab, kmax, seed)
    rng = np.random.default_rng(seed)
    n_total = rgb.shape[0]

    clusters: List[Cluster] = []
    for g in range(int(etiquetas.max()) + 1):
        idx = np.nonzero(etiquetas == g)[0]
        if idx.size == 0:
            continue
        rgb_m = rgb[idx].astype(np.float64).mean(axis=0)
        hsv_m = rgb_to_hsv(rgb_m)
        muestra = idx if idx.size <= config.HIST_SAMPLE_PIXELS else rng.choice(
            idx, config.HIST_SAMPLE_PIXELS, replace=False)
        xy = prep.xy[idx]
        clusters.append(Cluster(
            idx=len(clusters), n=int(idx.size), fraccion=idx.size / n_total,
            rgb=rgb_m, hsv=hsv_m, hsv_ef=hsv_efectivo(hsv_m),
            lab=lab[idx].mean(axis=0), hist=_hist_tono(hsv[idx]),
            lab_sample=lab[muestra],
            centroide=(float(xy[:, 0].mean()), float(xy[:, 1].mean())),
            dispersion=float(np.sqrt(xy.var(axis=0).sum())),
        ))
    pixel_cluster = _compactar(etiquetas)   # coincide con Cluster.idx (sin grupos vacíos)

    vector = np.concatenate([
        rgb.mean(axis=0) / 255.0, hsv.mean(axis=0), lab.mean(axis=0) / 100.0,
        _hist_tono(hsv),
        np.histogram(hsv[:, 1], bins=4, range=(0, 1))[0] / n_total,
        np.histogram(hsv[:, 2], bins=4, range=(0, 1))[0] / n_total,
        [len(clusters)],
    ])
    return FeatureSet(clusters=clusters, pixel_cluster=pixel_cluster,
                      n_valid=n_total, vector_global=vector)


def _compactar(etiquetas: np.ndarray) -> np.ndarray:
    _, inv = np.unique(etiquetas, return_inverse=True)
    return inv.astype(np.int32)


def compute_distances(fs: FeatureSet, palette) -> np.ndarray:
    """
    Distancias crudas (0-100 aprox.) entre cada grupo y cada color de la paleta.
    Forma (4, K, C) en el orden config.FEATURES = (rgb, hsv, lab, hist).
    No dependen de los parámetros, así que se calculan una sola vez por imagen.
    """
    if fs.distances is not None:
        return fs.distances
    cl_rgb = np.array([c.rgb for c in fs.clusters])
    cl_hsv = np.array([c.hsv_ef for c in fs.clusters])
    cl_lab = np.array([c.lab for c in fs.clusters])

    d_rgb = 100.0 * np.linalg.norm(cl_rgb[:, None, :] - palette.rgb[None, :, :], axis=2) / 441.673

    dh = np.abs(cl_hsv[:, None, 0] - palette.hsv_ef[None, :, 0])
    dh = np.minimum(dh, 1.0 - dh) * 360.0
    dh = np.minimum(dh / 60.0, 1.0)
    fac_sat = np.sqrt(cl_hsv[:, None, 1] * palette.hsv_ef[None, :, 1])
    ds = cl_hsv[:, None, 1] - palette.hsv_ef[None, :, 1]
    dv = cl_hsv[:, None, 2] - palette.hsv_ef[None, :, 2]
    d_hsv = 100.0 * np.sqrt((dh * fac_sat) ** 2 + ds ** 2 + dv ** 2) / np.sqrt(3.0)

    d_lab = delta_e(cl_lab[:, None, :], palette.lab[None, :, :])

    d_hist = np.stack([
        delta_e(c.lab_sample[:, None, :], palette.lab[None, :, :]).mean(axis=0)
        for c in fs.clusters
    ])
    fs.distances = np.stack([d_rgb, d_hsv, d_lab, d_hist])
    return fs.distances
