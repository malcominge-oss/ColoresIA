"""Evaluación de un conjunto de parámetros sobre muestras etiquetadas."""
import time
from typing import Dict, List

import numpy as np

import config
from classification.color_classifier import DESCONOCIDO, Params, classify_features


def _precision_muestra(cl, muestra) -> float:
    if muestra.truth_map is not None:
        return float((cl.labels_px == muestra.truth_map).mean())
    pred_vals, pred_cnt = np.unique(cl.labels_px, return_counts=True)
    pred = {int(v): c / cl.labels_px.size for v, c in zip(pred_vals, pred_cnt)}
    return float(sum(min(pred.get(k, 0.0), v) for k, v in muestra.proporciones.items()))


def evaluate_samples(muestras: List, palette, params: Params, metodo: str = "heuristica",
                     detallado: bool = False) -> Dict:
    C = len(palette)
    acc, de_num, de_den, nodos, cl_total = [], 0.0, 0, 0, 0
    conf = np.zeros((C + 1, C + 1), dtype=np.int64)       # filas: real, columnas: predicho
    t0 = time.perf_counter()
    for m in muestras:
        cl = classify_features(m.fs, palette, params, metodo)
        acc.append(_precision_muestra(cl, m))
        de_num += cl.delta_e_medio * m.fs.n_valid
        de_den += m.fs.n_valid
        nodos += cl.nodos_explorados
        cl_total += len(m.fs.clusters)
        if detallado and m.truth_map is not None:
            real = np.where(m.truth_map < 0, C, m.truth_map)
            pred = np.where(cl.labels_px < 0, C, cl.labels_px)
            np.add.at(conf, (real, pred), 1)
    tiempo_ms = (time.perf_counter() - t0) * 1000.0
    out = {
        "precision": float(np.mean(acc)) if acc else 0.0,
        "delta_e_medio": de_num / de_den if de_den else 0.0,
        "nodos_por_grupo": nodos / max(cl_total, 1),
        "coste_computacional": nodos / max(cl_total, 1) / ((1 << len(config.FEATURES)) * C),
        "tiempo_ms": tiempo_ms,
        "n_muestras": len(muestras),
    }
    if detallado and conf.sum() > 0:
        nombres = list(palette.nombres) + ["Desconocido"]
        tp = np.diag(conf).astype(float)
        prec = np.divide(tp, conf.sum(axis=0), out=np.zeros(C + 1), where=conf.sum(axis=0) > 0)
        rec = np.divide(tp, conf.sum(axis=1), out=np.zeros(C + 1), where=conf.sum(axis=1) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C + 1), where=(prec + rec) > 0)
        presentes = conf.sum(axis=1) > 0
        out["matriz_confusion"] = {"clases": nombres, "matriz": conf.tolist()}
        out["por_clase"] = {n: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                                "f1": round(float(f1[i]), 4), "soporte": int(conf[i].sum())}
                            for i, n in enumerate(nombres) if conf[i].sum() > 0 or conf[:, i].sum() > 0}
        out["f1_macro"] = float(f1[presentes].mean()) if presentes.any() else 0.0
    return out
