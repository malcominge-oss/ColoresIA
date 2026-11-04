"""
Espacio de estados de la clasificación de UN grupo de píxeles.

Estado  = (mask, c): mask indica qué rasgos ya se acumularon (bit i = config.FEATURES[i]),
          c es el color candidato de la paleta que se está evaluando.
Inicial = nodo raíz virtual "imagen/grupo sin procesar" (sus hijos son (0, c) para todo c).
Objetivo= mask completa (los 4 rasgos acumulados) con coste g <= límite de tolerancia.
Transición (mask,c) -> (mask|f, c): acumular el rasgo f; cuesta step_costs[f, c]
          = peso_f * distancia_f(grupo, c) / tolerancia_c   (siempre >= 0).

Como los costes son >= 0, el coste acumulado g solo crece: un nodo con g > límite
nunca llegará a un objetivo aceptable, así que podarlo es exacto (no pierde soluciones).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import config

Node = Tuple[int, int]   # (mask, candidato)


@dataclass
class SearchResult:
    metodo: str
    candidato: Optional[int]          # None = sin coincidencia suficiente (desconocido)
    coste: Optional[float]
    nodos_explorados: int
    nodos_generados: int
    max_frontera: int
    camino: List[str] = field(default_factory=list)   # rasgos acumulados en orden


class SearchProblem:
    def __init__(self, step_costs: np.ndarray, limite: float = config.SEARCH_COST_LIMIT):
        self.costs = np.asarray(step_costs, dtype=np.float64)    # (F, C)
        self.F, self.C = self.costs.shape
        self.goal_mask = (1 << self.F) - 1
        self.limite = float(limite)
        # h admisible y consistente: para cada rasgo pendiente, el mínimo coste posible entre candidatos
        self.min_cost = self.costs.min(axis=1)

    def is_goal(self, mask: int) -> bool:
        return mask == self.goal_mask

    def heuristic(self, mask: int) -> float:
        return float(sum(self.min_cost[f] for f in range(self.F) if not (mask >> f) & 1))

    def camino(self, parents: dict, node: Node) -> List[str]:
        pasos = []
        while parents.get(node) is not None:
            node, f = parents[node]
            pasos.append(config.FEATURES[f])
        return list(reversed(pasos))
