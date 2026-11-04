"""Búsqueda en profundidad (DFS): sigue una rama de rasgos hasta el final y retrocede."""
from search.search_graph import SearchProblem, SearchResult


def dfs(problem: SearchProblem) -> SearchResult:
    pila = []
    vistos = set()
    padres = {}
    explorados = generados = max_frontera = 0
    mejor = None

    for c in reversed(range(problem.C)):          # para visitar primero el candidato 0
        nodo = (0, c)
        vistos.add(nodo)
        padres[nodo] = None
        pila.append((0.0, nodo))
        generados += 1

    while pila:
        max_frontera = max(max_frontera, len(pila))
        g, nodo = pila.pop()
        cota = problem.limite if mejor is None else min(problem.limite, mejor[0])
        if g > cota or (mejor is not None and g >= mejor[0]):
            continue                    # retroceso: esta rama ya no puede mejorar
        explorados += 1
        mask, c = nodo
        if problem.is_goal(mask):
            if mejor is None or (g, c) < (mejor[0], mejor[1]):
                mejor = (g, c, nodo)
            continue
        hijos = []
        for f in range(problem.F):
            if (mask >> f) & 1:
                continue
            hijo = (mask | (1 << f), c)
            if hijo in vistos:
                continue
            vistos.add(hijo)
            padres[hijo] = (nodo, f)
            hijos.append((g + problem.costs[f, c], hijo))
            generados += 1
        pila.extend(reversed(hijos))

    if mejor is None:
        return SearchResult("dfs", None, None, explorados, generados, max_frontera)
    return SearchResult("dfs", mejor[1], float(mejor[0]), explorados, generados, max_frontera,
                        problem.camino(padres, mejor[2]))
