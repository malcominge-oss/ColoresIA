"""Endpoints HTTP que consume el HTML existente."""
import threading
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import config
from classification.color_classifier import (METODOS, Params, load_params, normalizar_metodo,
                                              params_info)
from pipeline import analizar, get_palette
from processing.image_loader import ImagenInvalida, load_image_base64, load_image_bytes
from processing.preprocessing import ImagenSinColores
from training.train import train

router = APIRouter(prefix="/api")

_lock = threading.Lock()
_entrenamiento = {"estado": "inactivo"}


class EntrenarRequest(BaseModel):
    dataset: str = Field("sintetico", description="'sintetico' o carpeta dentro de datasets/")
    n_imagenes: int = Field(60, ge=8, le=500)
    metodo: str = "heuristica"
    generaciones: Optional[int] = Field(None, ge=1, le=300)
    poblacion: Optional[int] = Field(None, ge=4, le=100)
    objetivo_precision: Optional[float] = Field(None, gt=0, le=1)
    paciencia: Optional[int] = Field(None, ge=1, le=100)
    sigma_region: float = Field(30.0, ge=0, le=80)


@router.post("/detectar")
def detectar(
    file: Optional[UploadFile] = File(None),
    imagen_base64: Optional[str] = Form(None),
    origen: str = Form("archivo"),
    metodo: str = Form("heuristica"),
    comparar: bool = Form(True),
    ignorar_fondo: Optional[bool] = Form(None),
):
    """Recibe la imagen del canvas o un archivo y devuelve colores, porcentajes y datos técnicos."""
    if origen not in ("dibujo", "archivo"):
        raise HTTPException(400, "El campo 'origen' debe ser 'dibujo' o 'archivo'.")
    try:
        metodo = normalizar_metodo(metodo)
        if file is not None:
            datos = file.file.read(config.MAX_UPLOAD_BYTES + 1)
            rgba = load_image_bytes(datos)
        elif imagen_base64:
            rgba = load_image_base64(imagen_base64)
        else:
            raise HTTPException(400, "Envía una imagen en 'file' o en 'imagen_base64'.")
        return analizar(rgba, origen=origen, ignorar_fondo=ignorar_fondo, metodo=metodo, comparar=comparar)
    except ImagenInvalida as exc:
        raise HTTPException(413 if "tamaño" in str(exc) else 400, str(exc))
    except ImagenSinColores as exc:
        raise HTTPException(422, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/estado")
def estado():
    with _lock:
        ent = dict(_entrenamiento)
    return {"ok": True, "estado": "operativo", "colores_en_paleta": len(get_palette()),
            "parametros_optimizados": params_info() is not None,
            "parametros_actuales": load_params().to_dict(), "entrenamiento": ent,
            "metodos": list(METODOS)}


@router.get("/configuracion")
def configuracion():
    info = params_info()
    return {
        "ok": True,
        "paleta": get_palette().to_list(),
        "rasgos": list(config.FEATURES),
        "metodos_busqueda": list(METODOS),
        "parametros_por_defecto": Params().normalizado().to_dict(),
        "parametros_actuales": load_params().to_dict(),
        "informe_ultimo_entrenamiento": None if not info else {
            k: info[k] for k in ("fecha", "dataset", "adoptados", "generacion_mejor",
                                 "motivo_parada", "fitness_mejor") if k in info},
        "algoritmo_genetico": config.GA_DEFAULTS,
        "fitness": config.FITNESS_WEIGHTS,
        "limites": {"tolerancia": config.TOLERANCE_RANGE, "lado_max_imagen": config.MAX_IMAGE_SIDE,
                    "bytes_max": config.MAX_UPLOAD_BYTES},
    }


def _trabajo(cfg: dict):
    def cb(r):
        with _lock:
            _entrenamiento.update({"generacion": r["generacion"], "mejor_fitness": r["mejor_fitness"],
                                   "precision_entrenamiento": r["precision_mejor"]})
    try:
        inf = train(callback=cb, **cfg)
        resumen = {k: inf[k] for k in ("fecha", "dataset", "adoptados", "generacion_mejor",
                                       "generaciones_ejecutadas", "motivo_parada", "fitness_mejor",
                                       "parametros", "tiempo_total_s")}
        resumen["validacion"] = {"precision_antes": inf["validacion"]["antes"]["precision"],
                                 "precision_optimizada": inf["validacion"]["optimizado"]["precision"]}
        with _lock:
            _entrenamiento.clear()
            _entrenamiento.update({"estado": "completado", "resumen": resumen})
    except Exception as exc:     # el entrenamiento nunca debe tumbar el servidor
        with _lock:
            _entrenamiento.clear()
            _entrenamiento.update({"estado": "error", "error": str(exc)})


@router.post("/entrenar", status_code=202)
def entrenar(req: EntrenarRequest):
    """Lanza el entrenamiento genético en segundo plano (no se ejecuta al pulsar Detectar)."""
    try:
        normalizar_metodo(req.metodo)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    with _lock:
        if _entrenamiento.get("estado") == "entrenando":
            raise HTTPException(409, "Ya hay un entrenamiento en curso.")
        _entrenamiento.clear()
        _entrenamiento.update({"estado": "entrenando", "generacion": 0})
    threading.Thread(target=_trabajo, args=(req.model_dump(),), daemon=True).start()
    return {"ok": True, "mensaje": "Entrenamiento iniciado. Consulta GET /api/estado."}
