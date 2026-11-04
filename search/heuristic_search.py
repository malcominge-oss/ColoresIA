"""
Búsqueda heurística Best-First / A*: f(n) = g(n) + h(n).

g(n): coste acumulado de los rasgos ya evaluados para el candidato.
h(n): suma, sobre los rasgos pendientes, del menor coste que cualquier candidato podría tener.
      Es admisible (nunca sobrestima) y consistente, por lo que el primer objetivo extraído
      de la frontera es óptimo y la búsqueda se detiene sin explorar el resto.
"""
import heapq
from itertools import count

from search.search_graph import SearchProblem, SearchResult

_EPS = 1e-12


def heuristic_search(problem: SearchProblem) -> SearchResult:
    contador = count()
    frontera = []
    cerrados = set()
    padres = {}
    mejor_g = {}
    explorados = generados = max_frontera = 0

    h0 = problem.heuristic(0)
    for c in range(problem.C):
        nodo = (0, c)
        padres[nodo] = None
        mejor_g[nodo] = 0.0
        if h0 <= problem.limite + _EPS:
            heapq.heappush(frontera, (h0, 0, next(contador), 0.0, nodo))
        generados += 1

    while frontera:
        max_frontera = max(max_frontera, len(frontera))
        f_val, _, _, g, nodo = heapq.heappop(frontera)
        if nodo in cerrados:
            continue
        if f_val > problem.limite + _EPS:
            break                       # nada restante puede cumplir la tolerancia
        cerrados.add(nodo)
        explorados += 1
        mask, c = nodo
        if problem.is_goal(mask):
            return SearchResult("heuristica", c, float(g), explorados, generados,
                                max_frontera, problem.camino(padres, nodo))
        for f in range(problem.F):
            if (mask >> f) & 1:
                continue
            hijo = (mask | (1 << f), c)
            if hijo in cerrados:
                continue
            g2 = g + problem.costs[f, c]
            f2 = g2 + problem.heuristic(hijo[0])
            if f2 > problem.limite + _EPS:
                continue
            if g2 < mejor_g.get(hijo, float("inf")) - _EPS:
                mejor_g[hijo] = g2
                padres[hijo] = (nodo, f)
                # a igual f se prefiere el nodo más profundo (más rasgos acumulados)
                heapq.heappush(frontera, (f2, -bin(hijo[0]).count("1"), next(contador), g2, hijo))
                generados += 1

    return SearchResult("heuristica", None, None, explorados, generados, max_frontera)
