"""Clasificador de colores: usa búsqueda (BFS / DFS / heurística) sobre los rasgos extraídos."""
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import config
from processing.feature_extraction import FeatureSet, compute_distances
from search.bfs import bfs
from search.dfs import dfs
from search.heuristic_search import heuristic_search
from search.search_graph import SearchProblem

METODOS = {"bfs": bfs, "dfs": dfs, "heuristica": heuristic_search}
ALIAS = {"heuristic": "heuristica", "heurística": "heuristica", "a*": "heuristica",
         "best-first": "heuristica", "astar": "heuristica"}
DESCONOCIDO = -1


def normalizar_metodo(nombre: str) -> str:
    n = (nombre or "heuristica").strip().lower()
    n = ALIAS.get(n, n)
    if n not in METODOS:
        raise ValueError(f"Método de búsqueda no válido: {nombre!r}. Usa bfs, dfs o heuristica.")
    return n


@dataclass
class Params:
    """Individuo del algoritmo genético: [w_rgb, w_hsv, w_lab, w_hist, tolerancia]."""
    pesos: tuple = config.DEFAULT_WEIGHTS
    tolerancia: float = config.DEFAULT_TOLERANCE

    def normalizado(self) -> "Params":
        w = np.clip(np.asarray(self.pesos, dtype=np.float64), 0.0, None)
        s = w.sum()
        w = w / s if s > 0 else np.full(len(config.FEATURES), 1.0 / len(config.FEATURES))
        lo, hi = config.TOLERANCE_RANGE
        return Params(tuple(float(x) for x in w), float(np.clip(self.tolerancia, lo, hi)))

    def to_vector(self) -> np.ndarray:
        return np.array([*self.pesos, self.tolerancia], dtype=np.float64)

    @classmethod
    def from_vector(cls, v) -> "Params":
        return cls(tuple(float(x) for x in v[:4]), float(v[4])).normalizado()

    def to_dict(self) -> Dict:
        p = self.normalizado()
        return {"pesos": dict(zip(config.FEATURES, (round(x, 5) for x in p.pesos))),
                "tolerancia": round(p.tolerancia, 4)}

    @classmethod
    def from_dict(cls, d: Dict) -> "Params":
        return cls(tuple(d["pesos"][f] for f in config.FEATURES), d["tolerancia"]).normalizado()


def load_params() -> Params:
    """Parámetros optimizados si existen; si no, los valores por defecto."""
    try:
        datos = json.loads(config.BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        return Params.from_dict(datos["parametros"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return Params().normalizado()


def params_info() -> Optional[Dict]:
    try:
        return json.loads(config.BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


@dataclass
class Classification:
    metodo: str
    asignaciones: List[Optional[int]]      # por grupo: índice de paleta o None
    costes: List[Optional[float]]
    labels_px: np.ndarray                  # (N,) índice de paleta o -1
    nodos_explorados: int
    nodos_generados: int
    tiempo_busqueda_ms: float
    delta_e_medio: float
    trazas: List[Dict] = field(default_factory=list)


def classify_features(fs: FeatureSet, palette, params: Params, metodo: str = "heuristica") -> Classification:
    metodo = normalizar_metodo(metodo)
    buscar = METODOS[metodo]
    p = params.normalizado()
    D = compute_distances(fs, palette)                        # (4, K, C)
    tau_c = p.tolerancia * palette.tol                        # (C,)
    N = np.asarray(p.pesos)[:, None, None] * D / tau_c[None, None, :]

    t0 = time.perf_counter()
    asign, costes, trazas = [], [], []
    exp = gen = 0
    for k, cl in enumerate(fs.clusters):
        res = buscar(SearchProblem(N[:, k, :], config.SEARCH_COST_LIMIT))
        asign.append(res.candidato)
        costes.append(res.coste)
        exp += res.nodos_explorados
        gen += res.nodos_generados
        trazas.append({"grupo": k, "candidato": res.candidato, "coste": res.coste,
                       "nodos_explorados": res.nodos_explorados,
                       "nodos_generados": res.nodos_generados, "camino": res.camino})
    t_ms = (time.perf_counter() - t0) * 1000.0

    mapa = np.array([DESCONOCIDO if a is None else a for a in asign], dtype=np.int32)
    labels_px = mapa[fs.pixel_cluster]

    num = den = 0.0
    for k, cl in enumerate(fs.clusters):
        if asign[k] is not None:
            num += cl.n * D[2, k, asign[k]]
            den += cl.n
    return Classification(metodo, asign, costes, labels_px, exp, gen, t_ms,
                          num / den if den else 0.0, trazas)


def resumen_por_color(fs: FeatureSet, palette, params: Params, cl: Classification) -> Dict:
    """Presencia, similitud, ΔE, coste y píxeles agregados por color (y desconocido)."""
    p = params.normalizado()
    D = compute_distances(fs, palette)
    tau_c = p.tolerancia * palette.tol
    filas: Dict[int, Dict] = {}
    unk = 0
    for k, c in enumerate(fs.clusters):
        a = cl.asignaciones[k]
        if a is None:
            unk += c.n
            continue
        f = filas.setdefault(a, {"pixeles": 0, "de": 0.0, "sim": 0.0, "coste": 0.0,
                                 "rgb": np.zeros(3)})
        de = float(D[2, k, a])
        f["pixeles"] += c.n
        f["de"] += c.n * de
        f["sim"] += c.n * float(100.0 * max(0.0, 1.0 - de / tau_c[a]))
        f["coste"] += c.n * float(cl.costes[k])
        f["rgb"] += c.n * c.rgb
    for f in filas.values():
        n = f["pixeles"]
        f["de"] /= n
        f["sim"] /= n
        f["coste"] /= n
        f["rgb"] = f["rgb"] / n
    return {"colores": filas, "desconocido": unk}
