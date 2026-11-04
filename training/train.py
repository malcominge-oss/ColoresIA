"""
Entrenamiento genético (separado de la detección).

Uso por consola:
    python -m training.train                       # dataset sintético
    python -m training.train --dataset datasets/mis_imagenes --generaciones 60
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import config
from classification.color_classifier import Params
from genetic.fitness import fitness_from_metrics
from genetic.genetic_algorithm import GeneticAlgorithm
from pipeline import get_palette
from training.dataset import generate_synthetic, load_folder, split
from training.validation import evaluate_samples


def train(dataset: str = "sintetico", n_imagenes: int = 60, metodo: str = "heuristica",
          callback: Optional[Callable[[Dict], None]] = None, guardar: bool = True,
          semilla_datos: int = 7, sigma_region: float = 30.0, **ga_cfg) -> Dict:
    palette = get_palette()
    t0 = time.perf_counter()
    if dataset == "sintetico":
        muestras = generate_synthetic(palette, n=n_imagenes, seed=semilla_datos, sigma_region=sigma_region)
        origen = f"sintético ({n_imagenes} imágenes, semilla {semilla_datos})"
    else:
        ruta = Path(dataset)
        if not ruta.is_absolute():
            ruta = config.BASE_DIR / ruta
        ruta = ruta.resolve()
        if config.DATASETS_DIR.resolve() not in ruta.parents and ruta != config.DATASETS_DIR.resolve():
            raise ValueError("El dataset debe estar dentro de la carpeta 'datasets/'.")
        muestras = load_folder(ruta, palette)
        origen = f"carpeta {ruta.name} ({len(muestras)} imágenes)"
    if len(muestras) < 4:
        raise ValueError("Se necesitan al menos 4 imágenes etiquetadas para entrenar y validar.")

    train_set, val_set = split(muestras, 0.7, seed=0)
    base = Params().normalizado()
    antes_train = evaluate_samples(train_set, palette, base, metodo, detallado=True)
    antes_val = evaluate_samples(val_set, palette, base, metodo, detallado=True)

    ga = GeneticAlgorithm(palette, **ga_cfg)
    res = ga.run(train_set, metodo, callback)

    opt = Params.from_vector(res["mejor_individuo"])
    despues_train = evaluate_samples(train_set, palette, opt, metodo, detallado=True)
    despues_val = evaluate_samples(val_set, palette, opt, metodo, detallado=True)
    fit_antes_val = fitness_from_metrics(antes_val)
    fit_despues_val = fitness_from_metrics(despues_val)

    # Anti-sobreajuste: solo se adoptan los parámetros si no empeoran en validación.
    adoptado = fit_despues_val >= fit_antes_val - 1e-9
    informe = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "dataset": origen,
        "metodo_busqueda": metodo,
        "parametros": (opt if adoptado else base).to_dict(),
        "parametros_candidatos_ga": opt.to_dict(),
        "adoptados": adoptado,
        "generacion_mejor": res["generacion_mejor"],
        "generaciones_ejecutadas": res["generaciones_ejecutadas"],
        "motivo_parada": res["motivo_parada"],
        "fitness_mejor": round(res["fitness"], 5),
        "configuracion_ga": res["configuracion"],
        "entrenamiento": {"antes": antes_train, "optimizado": despues_train},
        "validacion": {"antes": antes_val, "optimizado": despues_val,
                       "fitness_antes": round(fit_antes_val, 5),
                       "fitness_optimizado": round(fit_despues_val, 5)},
        "historial": res["historial"],
        "tiempo_total_s": round(time.perf_counter() - t0, 2),
    }
    if guardar:
        config.MODELS_DIR.mkdir(exist_ok=True)
        config.BEST_PARAMS_PATH.write_text(json.dumps(informe, ensure_ascii=False, indent=2, default=float),
                                           encoding="utf-8")
    return informe


def _resumen(inf: Dict) -> str:
    v = inf["validacion"]
    return (f"Generaciones: {inf['generaciones_ejecutadas']} (mejor en {inf['generacion_mejor']}, "
            f"parada: {inf['motivo_parada']})\n"
            f"Parámetros candidatos: {inf['parametros_candidatos_ga']}\n"
            f"Validación - precisión antes: {v['antes']['precision']:.4f} | "
            f"después: {v['optimizado']['precision']:.4f}\n"
            f"Validación - fitness antes: {v['fitness_antes']} | después: {v['fitness_optimizado']}\n"
            f"Parámetros adoptados: {'sí' if inf['adoptados'] else 'no (no mejoraron en validación)'}\n"
            f"Tiempo total: {inf['tiempo_total_s']} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrenamiento genético del clasificador de colores")
    ap.add_argument("--dataset", default="sintetico")
    ap.add_argument("--n-imagenes", type=int, default=60)
    ap.add_argument("--metodo", default="heuristica")
    ap.add_argument("--generaciones", type=int)
    ap.add_argument("--poblacion", type=int)
    ap.add_argument("--objetivo-precision", type=float)
    ap.add_argument("--paciencia", type=int)
    ap.add_argument("--sigma-region", type=float, default=30.0)
    a = ap.parse_args()
    inf = train(a.dataset, a.n_imagenes, a.metodo, sigma_region=a.sigma_region,
                callback=lambda r: print(f"gen {r['generacion']:3d}  fitness {r['mejor_fitness']:.4f}  "
                                         f"precisión {r['precision_mejor']:.4f}"),
                generaciones=a.generaciones, poblacion=a.poblacion,
                objetivo_precision=a.objetivo_precision, paciencia=a.paciencia)
    print(_resumen(inf))
