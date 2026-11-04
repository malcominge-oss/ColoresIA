"""Construye la respuesta JSON consistente que consume el HTML."""
from typing import Dict, List, Optional

import numpy as np

import config
from classification.color_metrics import rgb_to_hex, rgb_to_hsv, rgb_to_lab


def _resumen_clasificacion(fs, palette, params, cl, resumen) -> Dict:
    n = fs.n_valid
    colores: List[Dict] = []
    for idx, f in sorted(resumen["colores"].items(), key=lambda kv: -kv[1]["pixeles"]):
        pc = palette.colores[idx]
        rgb_det = f["rgb"]
        hsv_det = rgb_to_hsv(rgb_det)
        lab_det = rgb_to_lab(rgb_det)
        colores.append({
            "nombre": pc.nombre,
            "hex": pc.hex,
            "rgb": list(pc.rgb),
            "hsv": {"h": round(pc.hsv[0] * 360, 1), "s": round(pc.hsv[1], 3), "v": round(pc.hsv[2], 3)},
            "lab": {"L": round(pc.lab[0], 2), "a": round(pc.lab[1], 2), "b": round(pc.lab[2], 2)},
            "hex_detectado": rgb_to_hex(rgb_det),
            "rgb_detectado": [int(round(v)) for v in rgb_det],
            "hsv_detectado": {"h": round(float(hsv_det[0]) * 360, 1), "s": round(float(hsv_det[1]), 3),
                              "v": round(float(hsv_det[2]), 3)},
            "lab_detectado": {"L": round(float(lab_det[0]), 2), "a": round(float(lab_det[1]), 2),
                              "b": round(float(lab_det[2]), 2)},
            "pixeles": int(f["pixeles"]),
            "presencia": round(100.0 * f["pixeles"] / n, 2),
            "similitud": round(f["sim"], 2),
            "delta_e": round(f["de"], 3),
            "coste": round(f["coste"], 4),
        })
    unk = resumen["desconocido"]
    return {
        "colores": colores,
        "desconocido": {"nombre": "Desconocido", "pixeles": int(unk),
                        "presencia": round(100.0 * unk / n, 2)},
        "suma_presencia": round(sum(c["presencia"] for c in colores) + 100.0 * unk / n, 2),
    }


def format_result(prep, fs, palette, params, cl, resumen, tiempos: Dict,
                  comparacion: Optional[List[Dict]] = None, ga_info: Optional[Dict] = None) -> Dict:
    base = _resumen_clasificacion(fs, palette, params, cl, resumen)
    coste_total = float(sum(c["coste"] * c["pixeles"] for c in base["colores"]) / max(
        sum(c["pixeles"] for c in base["colores"]), 1)) if base["colores"] else None
    return {
        "ok": True,
        "metodo": cl.metodo,
        **base,
        "tecnico": {
            "nodo_inicial": "Imagen original sin procesar",
            "nodo_objetivo": "Clasificación final de colores",
            "metodo_busqueda": cl.metodo,
            "nodos_explorados": cl.nodos_explorados,
            "nodos_generados": cl.nodos_generados,
            "rasgos_extraidos": list(config.FEATURES) + ["histograma_tono", "distribucion_espacial"],
            "coste_total": None if coste_total is None else round(coste_total, 4),
            "tiempos_ms": {k: round(v, 2) for k, v in tiempos.items()},
            "parametros": params.to_dict(),
            "optimizado_con_ga": ga_info is not None,
            "algoritmo_genetico": ga_info,
            "imagen": {"origen": prep.origen, "ancho": prep.ancho, "alto": prep.alto,
                       "pixeles_totales": prep.total_pixels, "pixeles_validos": prep.n_valid,
                       "pixeles_transparentes": prep.transparentes, "pixeles_fondo_ignorados": prep.fondo,
                       "fondo_ignorado": prep.ignorar_fondo},
            "grupos": [{
                "grupo": c.idx, "pixeles": c.n, "fraccion": round(c.fraccion, 4),
                "hex": rgb_to_hex(c.rgb),
                "color_asignado": (palette.nombres[cl.asignaciones[c.idx]]
                                   if cl.asignaciones[c.idx] is not None else "Desconocido"),
                "coste": None if cl.costes[c.idx] is None else round(cl.costes[c.idx], 4),
                "centroide": [round(c.centroide[0], 3), round(c.centroide[1], 3)],
                "dispersion": round(c.dispersion, 3),
                "camino_busqueda": cl.trazas[c.idx]["camino"],
                "nodos_explorados": cl.trazas[c.idx]["nodos_explorados"],
            } for c in fs.clusters],
        },
        "comparacion": comparacion,
    }
