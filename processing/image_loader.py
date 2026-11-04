"""Carga y validación de imágenes (archivo subido, bytes o data URL del canvas)."""
import base64
import io
import binascii

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config

Image.MAX_IMAGE_PIXELS = 50_000_000


class ImagenInvalida(ValueError):
    """La imagen está vacía, es demasiado grande o no tiene un formato compatible."""


def load_image_bytes(data: bytes) -> np.ndarray:
    """Devuelve un arreglo RGBA uint8 (alto, ancho, 4)."""
    if not data:
        raise ImagenInvalida("La imagen está vacía.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ImagenInvalida("La imagen supera el tamaño máximo permitido (10 MB).")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in config.ALLOWED_FORMATS:
                raise ImagenInvalida(f"Formato no compatible: {img.format}.")
            img = ImageOps.exif_transpose(img)
            return np.array(img.convert("RGBA"), dtype=np.uint8)
    except ImagenInvalida:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImagenInvalida("No se pudo leer la imagen: archivo dañado o formato no compatible.") from exc


def load_image_base64(texto: str) -> np.ndarray:
    """Acepta 'data:image/png;base64,....' o base64 puro."""
    if not texto:
        raise ImagenInvalida("La imagen está vacía.")
    if "," in texto and texto.strip().lower().startswith("data:"):
        texto = texto.split(",", 1)[1]
    try:
        data = base64.b64decode(texto, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImagenInvalida("El contenido base64 no es válido.") from exc
    return load_image_bytes(data)
