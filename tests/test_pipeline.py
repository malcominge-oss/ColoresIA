"""Pruebas del pipeline. Ejecutar:  pytest -q   (o  python tests/test_pipeline.py)"""
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.color_classifier import METODOS, Params, classify_features
from genetic.fitness import fitness_from_metrics
from genetic.genetic_algorithm import GeneticAlgorithm
from pipeline import analizar, get_palette
from processing.feature_extraction import compute_distances, extract_features
from processing.image_loader import ImagenInvalida, load_image_bytes
from processing.preprocessing import ImagenSinColores, prepare
from search.search_graph import SearchProblem
from training.dataset import generate_synthetic, split
from training.validation import evaluate_samples


def img(w=60, h=60, fondo=(255, 255, 255, 255)):
    a = np.zeros((h, w, 4), np.uint8)
    a[:] = fondo
    return a


def nombres(r):
    return {c["nombre"]: c for c in r["colores"]}


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


def test_un_solo_color():
    a = img(fondo=(255, 0, 0, 255))
    r = analizar(a, "archivo")
    assert list(nombres(r)) == ["Rojo"] and r["colores"][0]["presencia"] == 100.0


def test_dos_colores_mitad_y_mitad():
    a = img(fondo=(255, 0, 0, 255))
    a[:, 30:] = (0, 0, 255, 255)
    n = nombres(analizar(a, "archivo"))
    assert set(n) == {"Rojo", "Azul"}
    assert abs(n["Rojo"]["presencia"] - 50) < 1 and abs(n["Azul"]["presencia"] - 50) < 1


def test_varios_colores_suma_100():
    a = img(fondo=(255, 255, 0, 255))
    a[:30, :30] = (0, 200, 0, 255)
    a[30:, 30:] = (139, 0, 255, 255)
    a[:20, 40:] = (255, 128, 0, 255)
    r = analizar(a, "archivo")
    assert abs(r["suma_presencia"] - 100) < 0.1
    assert {"Amarillo", "Verde", "Violeta", "Naranja"} <= set(nombres(r))


def test_colores_similares_se_distinguen():
    a = img(fondo=(255, 0, 0, 255))
    a[:, 30:] = (255, 128, 0, 255)
    assert set(nombres(analizar(a, "archivo"))) == {"Rojo", "Naranja"}


def test_claros_y_oscuros():
    a = img(fondo=(0, 0, 0, 255))
    a[:, 20:40] = (128, 128, 128, 255)
    a[:, 40:] = (255, 255, 255, 255)
    assert set(nombres(analizar(a, "archivo"))) == {"Negro", "Gris", "Blanco"}


def test_dibujo_ignora_fondo_blanco_y_transparente():
    for fondo in [(255, 255, 255, 255), (0, 0, 0, 0)]:
        a = img(fondo=fondo)
        a[10:40, 10:40] = (0, 200, 0, 255)
        r = analizar(a, "dibujo")
        assert list(nombres(r)) == ["Verde"] and r["colores"][0]["presencia"] == 100.0


def test_mismo_pipeline_canvas_y_archivo():
    a = img(fondo=(0, 0, 255, 255))
    a[:, :20] = (255, 0, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(a, "RGBA").save(buf, "PNG")
    desde_archivo = analizar(load_image_bytes(buf.getvalue()), "archivo", comparar=False)
    desde_array = analizar(a, "archivo", comparar=False)
    assert desde_archivo["colores"] == desde_array["colores"]


def test_metodos_dan_mismo_resultado_y_explorar_menos_nodos():
    a = img(fondo=(255, 0, 0, 255))
    a[:, 20:40] = (0, 0, 255, 255)
    a[:, 40:] = (255, 255, 0, 255)
    r = analizar(a, "archivo", comparar=True)
    por = {f["metodo"]: f for f in r["comparacion"]}
    assert por["bfs"]["asignaciones"] == por["dfs"]["asignaciones"] == por["heuristica"]["asignaciones"]
    assert por["heuristica"]["nodos_explorados"] < por["bfs"]["nodos_explorados"]
    assert por["dfs"]["nodos_explorados"] <= por["bfs"]["nodos_explorados"]


def test_busquedas_son_optimas_frente_a_fuerza_bruta():
    rng = np.random.default_rng(0)
    for _ in range(300):
        costes = rng.uniform(0, 0.6, size=(4, 13)) * rng.choice([0.3, 1.0, 3.0])
        p = SearchProblem(costes, 1.0)
        totales = costes.sum(axis=0)
        mejor = int(totales.argmin())
        esperado = mejor if totales[mejor] <= 1.0 else None
        for nombre, fn in METODOS.items():
            res = fn(p)
            assert res.candidato == esperado, (nombre, res.candidato, esperado)
            if esperado is not None:
                assert abs(res.coste - totales[mejor]) < 1e-9


def test_imagen_vacia_y_archivos_invalidos():
    assert raises(ImagenSinColores, lambda: prepare(img(), "dibujo"))
    assert raises(ImagenSinColores, lambda: prepare(img(fondo=(0, 0, 0, 0)), "archivo"))
    assert raises(ImagenInvalida, lambda: load_image_bytes(b""))
    assert raises(ImagenInvalida, lambda: load_image_bytes(b"esto no es una imagen"))


def test_desconocido_con_tolerancia_estricta():
    a = img(fondo=(225, 45, 45, 255))   # rojo apagado, lejos del rojo puro
    pal = get_palette()
    fs = extract_features(prepare(a, "archivo"))
    compute_distances(fs, pal)
    cl = classify_features(fs, pal, Params(tolerancia=5.0).normalizado(), "heuristica")
    assert cl.asignaciones == [None] and (cl.labels_px == -1).all()


def test_ga_no_empeora_en_entrenamiento_y_es_reproducible():
    pal = get_palette()
    muestras = generate_synthetic(pal, n=30, seed=3, sigma_region=40)
    tr, va = split(muestras, 0.7)
    base = fitness_from_metrics(evaluate_samples(tr, pal, Params().normalizado()))
    r1 = GeneticAlgorithm(pal, generaciones=8, poblacion=10, paciencia=8, semilla=1,
                          objetivo_precision=1.1).run(tr)
    r2 = GeneticAlgorithm(pal, generaciones=8, poblacion=10, paciencia=8, semilla=1,
                          objetivo_precision=1.1).run(tr)
    assert r1["fitness"] >= base - 1e-9            # elitismo + individuo base en la población
    assert np.allclose(r1["mejor_individuo"], r2["mejor_individuo"])
    assert abs(r1["mejor_individuo"][:4].sum() - 1.0) < 1e-9


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_") and callable(fn):
            try:
                fn()
                print("OK   ", nombre)
            except Exception as e:      # noqa
                fallos += 1
                print("FALLA", nombre, "->", repr(e))
    sys.exit(1 if fallos else 0)
