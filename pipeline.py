"""Pipeline único de detección: lo usan el canvas, los archivos subidos, el entrenamiento y las pruebas."""
import time
from typing import Optional

import numpy as np

from classification.color_classifier import (Params, classify_features, load_params,
                                              params_info, resumen_por_color,
                                              normalizar_metodo, METODOS)
from classification.color_palette import Palette, load_palette
from processing.feature_extraction import compute_distances, extract_features
from processing.preprocessing import prepare
from results.result_formatter import format_result

_PALETTE: Optional[Palette] = None


def get_palette() -> Palette:
    global _PALETTE
    if _PALETTE is None:
        _PALETTE = load_palette()
    return _PALETTE


def analizar(rgba: np.ndarray, origen: str = "archivo", ignorar_fondo=None,
             metodo: str = "heuristica", comparar: bool = True,
             params: Optional[Params] = None) -> dict:
    metodo = normalizar_metodo(metodo)
    palette = get_palette()
    params = (params or load_params()).normalizado()

    t0 = time.perf_counter()
    prep = prepare(rgba, origen=origen, ignorar_fondo=ignorar_fondo)
    t1 = time.perf_counter()
    fs = extract_features(prep)
    compute_distances(fs, palette)
    t2 = time.perf_counter()
    cl = classify_features(fs, palette, params, metodo)
    resumen = resumen_por_color(fs, palette, params, cl)
    t3 = time.perf_counter()

    comparacion = None
    if comparar:
        comparacion = []
        ref = None
        for m in METODOS:
            c = cl if m == metodo else classify_features(fs, palette, params, m)
            nombres = sorted({palette.nombres[a] for a in c.asignaciones if a is not None})
            costes = [x for x in c.costes if x is not None]
            peso = [fs.clusters[i].n for i, x in enumerate(c.costes) if x is not None]
            coste = float(np.average(costes, weights=peso)) if costes else None
            comparacion.append({
                "metodo": m, "nodos_explorados": c.nodos_explorados,
                "nodos_generados": c.nodos_generados,
                "tiempo_busqueda_ms": round(c.tiempo_busqueda_ms, 3),
                "coste_total": None if coste is None else round(coste, 4),
                "colores_detectados": nombres,
                "asignaciones": [None if a is None else palette.nombres[a] for a in c.asignaciones],
            })
        ref = comparacion[-1]["asignaciones"] if comparacion[-1]["metodo"] == "heuristica" else None
        for fila in comparacion:
            fila["mismo_resultado_que_heuristica"] = (fila["asignaciones"] == ref) if ref else None

    info = params_info()
    ga_info = None
    if info:
        ga_info = {"generacion": info.get("generacion_mejor"), "fitness": info.get("fitness_mejor"),
                   "precision_validacion": info.get("validacion", {}).get("optimizado", {}).get("precision"),
                   "fecha": info.get("fecha")}

    tiempos = {"preprocesado": (t1 - t0) * 1000, "extraccion_rasgos": (t2 - t1) * 1000,
               "busqueda_y_clasificacion": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}
    return format_result(prep, fs, palette, params, cl, resumen, tiempos, comparacion, ga_info)
