"""Algoritmo genético que optimiza pesos de rasgos y tolerancia del clasificador."""
import time
from typing import Callable, Dict, List, Optional

import numpy as np

import config
from classification.color_classifier import Params
from genetic.fitness import evaluate_individual
from genetic.operators import (crossover_arithmetic, mutate, normalize,
                               random_individual, tournament)


class GeneticAlgorithm:
    def __init__(self, palette, **cfg):
        self.palette = palette
        self.cfg = {**config.GA_DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
        self.rng = np.random.default_rng(self.cfg["semilla"])

    def run(self, muestras_train: List, metodo: str = "heuristica",
            callback: Optional[Callable[[Dict], None]] = None,
            limite_tiempo_s: Optional[float] = None) -> Dict:
        c = self.cfg
        pop_n = int(c["poblacion"])
        base = Params().normalizado().to_vector()           # el punto de partida entra en la población
        poblacion = [base] + [random_individual(self.rng) for _ in range(pop_n - 1)]

        historial = []
        mejor_global, mejor_fit, mejor_m, gen_mejor = None, -np.inf, None, 0
        sin_mejora = 0
        motivo = "generaciones_maximas"
        t_ini = time.perf_counter()

        for gen in range(1, int(c["generaciones"]) + 1):
            evals = [evaluate_individual(ind, muestras_train, self.palette, metodo) for ind in poblacion]
            fits = np.array([e[0] for e in evals])
            orden = np.argsort(-fits)
            mejor_i = int(orden[0])

            if fits[mejor_i] > mejor_fit + 1e-4:
                sin_mejora = 0
            else:
                sin_mejora += 1
            if fits[mejor_i] > mejor_fit:
                mejor_fit, mejor_global = float(fits[mejor_i]), poblacion[mejor_i].copy()
                mejor_m, gen_mejor = evals[mejor_i][1], gen

            registro = {"generacion": gen, "mejor_fitness": round(float(fits[mejor_i]), 5),
                        "fitness_medio": round(float(fits.mean()), 5),
                        "precision_mejor": round(evals[mejor_i][1]["precision"], 5),
                        "mejor_individuo": [round(float(x), 4) for x in poblacion[mejor_i]]}
            historial.append(registro)
            if callback:
                callback(registro)

            if mejor_m["precision"] >= c["objetivo_precision"]:
                motivo = "precision_objetivo_alcanzada"
                break
            if sin_mejora >= int(c["paciencia"]):
                motivo = "sin_mejora"
                break
            if limite_tiempo_s and time.perf_counter() - t_ini > limite_tiempo_s:
                motivo = "limite_de_tiempo"
                break

            # Nueva generación: elitismo + torneo + cruce aritmético + mutación
            nueva = [poblacion[i].copy() for i in orden[:int(c["elitismo"])]]
            while len(nueva) < pop_n:
                p1 = tournament(poblacion, fits, int(c["torneo"]), self.rng)
                p2 = tournament(poblacion, fits, int(c["torneo"]), self.rng)
                if self.rng.random() < c["prob_cruce"]:
                    h1, h2 = crossover_arithmetic(p1, p2, self.rng)
                else:
                    h1, h2 = p1.copy(), p2.copy()
                nueva.append(mutate(h1, self.rng, c["prob_mutacion"], c["sigma_mutacion"]))
                if len(nueva) < pop_n:
                    nueva.append(mutate(h2, self.rng, c["prob_mutacion"], c["sigma_mutacion"]))
            poblacion = nueva

        return {"mejor_individuo": normalize(mejor_global), "fitness": mejor_fit,
                "metricas_train": mejor_m, "generacion_mejor": gen_mejor,
                "generaciones_ejecutadas": len(historial), "motivo_parada": motivo,
                "historial": historial, "tiempo_s": time.perf_counter() - t_ini,
                "configuracion": c}
