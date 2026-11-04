"""Operadores genéticos. Individuo = [w_rgb, w_hsv, w_lab, w_hist, tolerancia]."""
import numpy as np

import config

N_W = len(config.FEATURES)


def normalize(ind: np.ndarray) -> np.ndarray:
    ind = np.array(ind, dtype=np.float64, copy=True)
    w = np.clip(ind[:N_W], 0.0, None)
    s = w.sum()
    ind[:N_W] = w / s if s > 0 else 1.0 / N_W
    ind[N_W] = np.clip(ind[N_W], *config.TOLERANCE_RANGE)
    return ind


def random_individual(rng: np.random.Generator) -> np.ndarray:
    w = rng.dirichlet(np.ones(N_W))
    tau = rng.uniform(*config.TOLERANCE_RANGE)
    return normalize(np.append(w, tau))


def tournament(poblacion, fitness, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(poblacion), size=min(k, len(poblacion)), replace=False)
    return poblacion[idx[np.argmax(np.asarray(fitness)[idx])]]


def crossover_arithmetic(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator):
    """Cruce aritmético gen a gen; después se normalizan los pesos."""
    a = rng.random(len(p1))
    return normalize(a * p1 + (1 - a) * p2), normalize((1 - a) * p1 + a * p2)


def mutate(ind: np.ndarray, rng: np.random.Generator, prob: float, sigma: float) -> np.ndarray:
    ind = np.array(ind, dtype=np.float64, copy=True)
    for g in range(N_W):
        if rng.random() < prob:
            ind[g] += rng.normal(0, sigma)
    if rng.random() < prob:
        lo, hi = config.TOLERANCE_RANGE
        ind[N_W] += rng.normal(0, sigma * (hi - lo))
    return normalize(ind)
