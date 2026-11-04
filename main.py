"""Servidor FastAPI. Ejecutar:  uvicorn main:app --reload   (o  python main.py)"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from api.routes import router

app = FastAPI(title="IA de reconocimiento de colores",
              description="Búsqueda en espacio de rasgos (BFS/DFS/heurística) + algoritmo genético")

# Permite abrir el HTML desde otro origen (p. ej. file:// o Live Server) durante el desarrollo.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"ok": False, "error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    return JSONResponse({"ok": False, "error": "Solicitud incorrecta.", "detalle": exc.errors()[:5]},
                        status_code=422)


app.include_router(router)

if config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def raiz():
    index = config.STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"ok": True, "mensaje": "API activa. Documentación en /docs. Coloca tu HTML en static/index.html."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
