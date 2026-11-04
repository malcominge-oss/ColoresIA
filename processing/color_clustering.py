"""Agrupación de píxeles similares (K-Means en CIELAB + fusión de grupos casi idénticos)."""
import numpy as np
from sklearn.cluster import KMeans

import config
from classification.color_metrics import delta_e


def cluster_pixels(lab: np.ndarray, kmax: int = config.KMEANS_MAX_K, seed: int = 0) -> np.ndarray:
    """Devuelve etiquetas (N,) con grupos consecutivos 0..k-1."""
    n = lab.shape[0]
    rng = np.random.default_rng(seed)
    if n > config.KMEANS_SAMPLE:
        muestra = lab[rng.choice(n, config.KMEANS_SAMPLE, replace=False)]
    else:
        muestra = lab
    distintos = len(np.unique(np.round(muestra / 2.0).astype(np.int32), axis=0))
    k = int(max(1, min(kmax, distintos)))
    if k == 1:
        return np.zeros(n, dtype=np.int32)

    km = KMeans(n_clusters=k, n_init=3, random_state=seed)
    km.fit(muestra)
    etiquetas = km.predict(lab)
    centros = km.cluster_centers_

    # Fusionar grupos cuyos centros son perceptualmente casi iguales.
    padre = list(range(k))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(k):
        for j in range(i + 1, k):
            if float(delta_e(centros[i], centros[j])) < config.CLUSTER_MERGE_DELTA_E:
                padre[raiz(j)] = raiz(i)
    mapa = np.array([raiz(i) for i in range(k)])
    etiquetas = mapa[etiquetas]
    _, compactadas = np.unique(etiquetas, return_inverse=True)
    return compactadas.astype(np.int32)
