"""Fitness = alpha*A - beta*E - gamma*C (A: precisión, E: error cromático, C: coste computacional)."""
import numpy as np

import config
from classification.color_classifier import Params
from training.validation import evaluate_samples


def fitness_from_metrics(m: dict, alpha=None, beta=None, gamma=None) -> float:
    w = config.FITNESS_WEIGHTS
    alpha = w["alpha"] if alpha is None else alpha
    beta = w["beta"] if beta is None else beta
    gamma = w["gamma"] if gamma is None else gamma
    E = min(m["delta_e_medio"] / 50.0, 1.0)         # error cromático normalizado
    C = min(m["coste_computacional"], 1.0)          # nodos explorados normalizados
    return alpha * m["precision"] - beta * E - gamma * C


def evaluate_individual(ind: np.ndarray, muestras, palette, metodo="heuristica"):
    params = Params.from_vector(ind)
    m = evaluate_samples(muestras, palette, params, metodo)
    return fitness_from_metrics(m), m
