"""Búsqueda en amplitud (BFS): explora por niveles de rasgos acumulados."""
from collections import deque

from search.search_graph import SearchProblem, SearchResult


def bfs(problem: SearchProblem) -> SearchResult:
    cola = deque()
    vistos = set()
    padres = {}
    explorados = generados = max_frontera = 0
    mejor = None                       # (coste, candidato, nodo)

    for c in range(problem.C):
        nodo = (0, c)
        vistos.add(nodo)
        padres[nodo] = None
        cola.append((0.0, nodo))
        generados += 1

    while cola:
        max_frontera = max(max_frontera, len(cola))
        g, nodo = cola.popleft()
        cota = problem.limite if mejor is None else min(problem.limite, mejor[0])
        if g > cota or (mejor is not None and g >= mejor[0]):
            continue                    # poda exacta (límite de tolerancia / mejor conocido)
        explorados += 1
        mask, c = nodo
        if problem.is_goal(mask):
            if mejor is None or (g, c) < (mejor[0], mejor[1]):
                mejor = (g, c, nodo)
            continue
        for f in range(problem.F):
            if (mask >> f) & 1:
                continue
            hijo = (mask | (1 << f), c)
            if hijo in vistos:
                continue
            vistos.add(hijo)
            padres[hijo] = (nodo, f)
            cola.append((g + problem.costs[f, c], hijo))
            generados += 1

    if mejor is None:
        return SearchResult("bfs", None, None, explorados, generados, max_frontera)
    return SearchResult("bfs", mejor[1], float(mejor[0]), explorados, generados, max_frontera,
                        problem.camino(padres, mejor[2]))
