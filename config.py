"""Configuración central del sistema de reconocimiento de colores."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_parameters.json"
CUSTOM_PALETTE_PATH = MODELS_DIR / "palette.json"   # opcional: sobrescribe la paleta
DATASETS_DIR = BASE_DIR / "datasets"
STATIC_DIR = BASE_DIR / "static"

# --- Imagen ---------------------------------------------------------------
MAX_IMAGE_SIDE = 256                 # lado máximo tras reducir (rendimiento)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "GIF"}
ALPHA_MIN = 128                      # alfa mínimo para que un píxel sea válido
BACKGROUND_MIN_CHANNEL = 245         # "casi blanco" = fondo cuando se ignora

# --- Rasgos ---------------------------------------------------------------
FEATURES = ("rgb", "hsv", "lab", "hist")   # orden fijo de los rasgos
KMEANS_MAX_K = 8
KMEANS_SAMPLE = 4000
CLUSTER_MERGE_DELTA_E = 5.0
HIST_SAMPLE_PIXELS = 64
HUE_BINS = 12

# --- Paleta (nombre, hex, factor de tolerancia) ---------------------------
# Cámbiala para que coincida EXACTAMENTE con los colores de tu página.
PALETTE_COLORS = [
    ("Rojo",     "#FF0000", 1.0),
    ("Naranja",  "#FF8000", 1.0),
    ("Amarillo", "#FFFF00", 1.0),
    ("Verde",    "#00C800", 1.0),
    ("Cian",     "#00FFFF", 1.0),
    ("Azul",     "#0000FF", 1.0),
    ("Índigo",   "#4B0082", 1.0),
    ("Violeta",  "#8B00FF", 1.0),
    ("Rosa",     "#FF69B4", 1.0),
    ("Marrón",   "#8B4513", 1.0),
    ("Blanco",   "#FFFFFF", 1.0),
    ("Gris",     "#808080", 1.0),
    ("Negro",    "#000000", 1.0),
]

# --- Parámetros de clasificación por defecto (antes de optimizar) ---------
DEFAULT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)   # rgb, hsv, lab, hist
DEFAULT_TOLERANCE = 20.0
TOLERANCE_RANGE = (5.0, 40.0)
SEARCH_COST_LIMIT = 1.0     # coste ponderado máximo aceptable (1.0 = tolerancia)

# --- Algoritmo genético ---------------------------------------------------
GA_DEFAULTS = {
    "poblacion": 24,
    "generaciones": 100,
    "objetivo_precision": 0.95,
    "paciencia": 15,
    "elitismo": 2,
    "torneo": 3,
    "prob_cruce": 0.85,
    "prob_mutacion": 0.25,
    "sigma_mutacion": 0.08,
    "semilla": 42,
}
FITNESS_WEIGHTS = {"alpha": 1.0, "beta": 0.15, "gamma": 0.05}
