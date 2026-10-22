import hashlib
import io
import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Optional

import google.generativeai as genai
import pdfplumber
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Carga las variables del archivo .env (solo en local; en Render se usan las de Environment)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_MODELS", "gemini-3.1-flash-lite,gemini-3.5-flash-lite").split(",")
    if m.strip()
]
API_KEY = os.getenv("API_KEY", "")
# Si DATABASE_URL existe (Neon/Postgres) los sellos se guardan ahí y NO se pierden.
# Si no existe, se usa SQLite local (solo para pruebas en tu PC).
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_PATH = os.getenv("DB_PATH", "sellos.db")
MAX_PDF_MB = 10
MAX_AUDIO_MB = 15
MAX_CV_CHARS = 20000
UMBRAL_RESPUESTA = float(os.getenv("UMBRAL_RESPUESTA", "0.70"))

app = FastAPI(title="API Reclutamiento Transparente", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def verificar_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida o ausente")


# ---------------------------------------------------------------------------
# IA (Gemini) con respaldo entre modelos
# ---------------------------------------------------------------------------
def get_model(nombre: str) -> genai.GenerativeModel:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Falta GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        nombre,
        generation_config={"response_mime_type": "application/json", "temperature": 0.2},
    )


def parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw or "")
        if not match:
            raise HTTPException(status_code=502, detail="La IA no devolvió un JSON válido")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=502, detail=f"JSON inválido de la IA: {e}")


def llamar_ia(partes: list) -> dict:
    """Prueba los modelos en orden; si uno falla (cuota 429, modelo no disponible), usa el siguiente."""
    ultimo_error = None
    for nombre in GEMINI_MODELS:
        model = get_model(nombre)
        try:
            resp = model.generate_content(partes)
            return parse_json(resp.text)
        except HTTPException:
            raise
        except Exception as e:
            ultimo_error = e
    raise HTTPException(
        status_code=502,
        detail=f"Error al llamar a Gemini (modelos probados: {', '.join(GEMINI_MODELS)}): {ultimo_error}",
    )


# ---------------------------------------------------------------------------
# Base de datos de sellos: Postgres (persistente) o SQLite (solo local)
# ---------------------------------------------------------------------------
def _pg():
    import psycopg

    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=15)


def init_db():
    if DATABASE_URL:
        with _pg() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS sellos (
                    company_id TEXT PRIMARY KEY,
                    cumplidos TEXT NOT NULL,
                    emitido_en TEXT NOT NULL,
                    huella TEXT NOT NULL
                )"""
            )
    else:
        with closing(sqlite3.connect(DB_PATH)) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS sellos (
                    company_id TEXT PRIMARY KEY,
                    cumplidos TEXT NOT NULL,
                    emitido_en TEXT NOT NULL,
                    huella TEXT NOT NULL
                )"""
            )
            con.commit()


def db_guardar_sello(company_id: str, cumplidos: list, emitido: str, huella: str):
    valores = (company_id, json.dumps(cumplidos), emitido, huella)
    if DATABASE_URL:
        with _pg() as con:
            con.execute(
                """INSERT INTO sellos (company_id, cumplidos, emitido_en, huella)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (company_id) DO UPDATE SET
                     cumplidos = EXCLUDED.cumplidos,
                     emitido_en = EXCLUDED.emitido_en,
                     huella = EXCLUDED.huella""",
                valores,
            )
    else:
        with closing(sqlite3.connect(DB_PATH)) as con:
            con.execute("INSERT OR REPLACE INTO sellos VALUES (?, ?, ?, ?)", valores)
            con.commit()


def db_borrar_sello(company_id: str):
    if DATABASE_URL:
        with _pg() as con:
            con.execute("DELETE FROM sellos WHERE company_id = %s", (company_id,))
    else:
        with closing(sqlite3.connect(DB_PATH)) as con:
            con.execute("DELETE FROM sellos WHERE company_id = ?", (company_id,))
            con.commit()


def db_obtener_sello(company_id: str):
    if DATABASE_URL:
        with _pg() as con:
            return con.execute(
                "SELECT cumplidos, emitido_en, huella FROM sellos WHERE company_id = %s",
                (company_id,),
            ).fetchone()
    with closing(sqlite3.connect(DB_PATH)) as con:
        return con.execute(
            "SELECT cumplidos, emitido_en, huella FROM sellos WHERE company_id = ?",
            (company_id,),
        ).fetchone()


@app.on_event("startup")
def on_startup():
    try:
        init_db()
    except Exception as e:  # que /cv y /voz sigan funcionando aunque falle la base
        print(f"ADVERTENCIA: no se pudo iniciar la base de datos: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelos": GEMINI_MODELS,
        "gemini_configurado": bool(GEMINI_API_KEY),
        "base_de_datos": "postgres (persistente)" if DATABASE_URL else "sqlite local (temporal)",
    }


# ---------------------------------------------------------------------------
# /cv
# ---------------------------------------------------------------------------
class CVResult(BaseModel):
    puntaje: int = Field(ge=0, le=100)
    descripcion: str
    postulacion_ciega: bool


def extraer_texto_pdf(data: bytes) -> str:
    texto = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texto.append(t)
    return "\n".join(texto).strip()


def anonimizar(texto: str) -> str:
    texto = re.sub(r"[\w\.\-+]+@[\w\-]+\.[\w\.\-]+", "[CORREO]", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", "[ENLACE]", texto)
    texto = re.sub(r"(\+?\d[\d\s\-()]{7,}\d)", "[TELEFONO]", texto)
    return texto


@app.post("/cv", response_model=CVResult, dependencies=[Depends(verificar_api_key)])
async def analizar_cv(
    cv: UploadFile = File(..., description="CV en PDF"),
    requisitos: str = Form(..., description="Requisitos del puesto"),
    postulacion_ciega: bool = Form(True, description="Quitar datos de contacto antes de analizar"),
):
    if not (cv.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser PDF")

    data = await cv.read()
    if len(data) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"El PDF supera {MAX_PDF_MB} MB")

    try:
        texto = await run_in_threadpool(extraer_texto_pdf, data)
    except Exception:
        raise HTTPException(status_code=422, detail="No se pudo leer el PDF (¿está dañado?)")

    if not texto:
        raise HTTPException(
            status_code=422,
            detail="El PDF no tiene texto seleccionable (puede ser un escaneo/imagen).",
        )

    if postulacion_ciega:
        texto = anonimizar(texto)
    texto = texto[:MAX_CV_CHARS]

    prompt = f"""Eres un evaluador de CVs objetivo. Evalúa el CV solo por experiencia, estudios y habilidades
relacionadas con los requisitos. Ignora nombre, edad, género, foto, estado civil y cualquier dato personal.

REQUISITOS DEL PUESTO:
{requisitos}

CONTENIDO DEL CV:
{texto}

Responde SOLO con JSON con esta estructura exacta:
{{"puntaje": <entero 0-100>, "descripcion": "<explicación breve en español, máx. 4 oraciones, citando qué cumple y qué falta>"}}"""

    data_ia = await run_in_threadpool(llamar_ia, [prompt])
    try:
        puntaje = max(0, min(100, int(data_ia.get("puntaje", 0))))
    except (TypeError, ValueError):
        raise HTTPException(status_code=502, detail="La IA devolvió un puntaje inválido")

    return CVResult(
        puntaje=puntaje,
        descripcion=str(data_ia.get("descripcion", "")).strip(),
        postulacion_ciega=postulacion_ciega,
    )


# ---------------------------------------------------------------------------
# /voz
# ---------------------------------------------------------------------------
class VozResult(BaseModel):
    transcripcion: str
    estudios: list[str]
    experiencia: list[str]
    habilidades: list[str]
    audio_borrado: bool = True
    requiere_aprobacion_postulante: bool = True


MIME_AUDIO = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


@app.post("/voz", response_model=VozResult, dependencies=[Depends(verificar_api_key)])
async def cv_por_voz(audio: UploadFile = File(..., description="Audio de 60-90 s")):
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in MIME_AUDIO:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado '{ext}'. Usa: {', '.join(MIME_AUDIO)}",
        )

    data = await audio.read()
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"El audio supera {MAX_AUDIO_MB} MB")
    if not data:
        raise HTTPException(status_code=400, detail="El audio está vacío")

    prompt = """Transcribe fielmente este audio en español (una persona narrando su CV) y luego
estructura la información. No inventes datos que no se mencionen.

Responde SOLO con JSON con esta estructura exacta:
{"transcripcion": "<texto completo>",
 "estudios": ["<título, institución, año si se menciona>"],
 "experiencia": ["<puesto, empresa, responsabilidades>"],
 "habilidades": ["<competencia técnica o blanda>"]}"""

    resultado = await run_in_threadpool(
        llamar_ia, [prompt, {"mime_type": MIME_AUDIO[ext], "data": data}]
    )
    del data

    def lista(clave: str) -> list[str]:
        v = resultado.get(clave, [])
        return [str(x) for x in v] if isinstance(v, list) else []

    return VozResult(
        transcripcion=str(resultado.get("transcripcion", "")).strip(),
        estudios=lista("estudios"),
        experiencia=lista("experiencia"),
        habilidades=lista("habilidades"),
    )


# ---------------------------------------------------------------------------
# /sello
# ---------------------------------------------------------------------------
class CriteriosEmpresa(BaseModel):
    company_id: str = Field(min_length=1, max_length=80)
    publica_requisitos_verificables: bool
    acepta_postulacion_ciega: bool
    comunica_motivo_no_seleccion: bool
    tasa_respuesta: float = Field(ge=0.0, le=1.0, description="0.0 a 1.0")
    tiene_requisitos_discriminatorios: bool


class SelloResult(BaseModel):
    company_id: str
    sello_otorgado: bool
    criterios_cumplidos: list[str]
    criterios_pendientes: list[str]
    ledger_url: Optional[str] = None
    emitido_en: Optional[str] = None
    huella: Optional[str] = None


@app.post("/sello/evaluar", response_model=SelloResult, dependencies=[Depends(verificar_api_key)])
def evaluar_empresa(c: CriteriosEmpresa):
    checks = {
        "publica_requisitos_verificables": c.publica_requisitos_verificables,
        "acepta_postulacion_ciega": c.acepta_postulacion_ciega,
        "comunica_motivo_no_seleccion": c.comunica_motivo_no_seleccion,
        f"tasa_respuesta_mayor_o_igual_{int(UMBRAL_RESPUESTA * 100)}%": c.tasa_respuesta >= UMBRAL_RESPUESTA,
        "sin_requisitos_discriminatorios": not c.tiene_requisitos_discriminatorios,
    }
    cumplidos = [k for k, ok in checks.items() if ok]
    pendientes = [k for k, ok in checks.items() if not ok]
    otorgado = not pendientes

    emitido = huella = url = None
    try:
        if otorgado:
            emitido = datetime.now(timezone.utc).isoformat()
            huella = hashlib.sha256(
                json.dumps([c.company_id, cumplidos, emitido]).encode()
            ).hexdigest()
            db_guardar_sello(c.company_id, cumplidos, emitido, huella)
            base = os.getenv("PUBLIC_URL", "").rstrip("/")
            url = f"{base}/sello/{c.company_id}"
        else:
            db_borrar_sello(c.company_id)  # si ya no cumple, se revoca el sello anterior
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No se pudo acceder a la base de datos: {e}")

    return SelloResult(
        company_id=c.company_id,
        sello_otorgado=otorgado,
        criterios_cumplidos=cumplidos,
        criterios_pendientes=pendientes,
        ledger_url=url,
        emitido_en=emitido,
        huella=huella,
    )


@app.get("/sello/{company_id}")
def consultar_sello(company_id: str):
    try:
        row = db_obtener_sello(company_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No se pudo acceder a la base de datos: {e}")
    if not row:
        raise HTTPException(status_code=404, detail="Esta empresa no tiene el sello vigente")
    return {
        "company_id": company_id,
        "sello_vigente": True,
        "criterios_cumplidos": json.loads(row[0]),
        "emitido_en": row[1],
        "huella": row[2],
    }