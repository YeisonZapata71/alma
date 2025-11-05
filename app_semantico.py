"""
Alma - Asistente Inteligente de Soporte Técnico
Versión Híbrida Semántica + LLaMA Fallback (Ollama)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional
import sqlite3, json, numpy as np, faiss, re, os, requests
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz
import requests, json


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
DB_PATH = "soporte.db"
INDEX_FILE = "index.faiss"
MAP_FILE = "index_map.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Configuración LLaMA / Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
LLAMA_ENABLED = True
LLAMA_TIMEOUT = 60  # segundos

# =====================================================
# APP FASTAPI
# =====================================================
app = FastAPI(
    title="Alma - Asistente Inteligente de Soporte Técnico",
    description=(
        "Alma es un asistente híbrido (FAISS + Fuzzy + LLaMA) para soporte técnico. "
        "Prioriza la base de conocimiento, usa búsqueda semántica, y si no encuentra coincidencias, "
        "genera respuestas tentativas con LLaMA (Ollama)."
    ),
    version="3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODELOS DE DATOS
# =====================================================
class Consulta(BaseModel):
    pregunta: str
    contexto: Optional[str] = None

class Respuesta(BaseModel):
    nombre_asistente: str = "Alma"
    respuesta: str
    confianza: int
    metodo: str
    sugerencias: List[str] = []
    debe_escalar: bool = False

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================
def normalizar_texto(texto: str) -> str:
    texto = (texto or "").lower().strip()
    texto = re.sub(r'[¿?¡!.,;:]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

def expandir_sinonimos(texto: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT termino_original, sinonimo FROM sinonimos")
        sinonimos = cur.fetchall()
    except sqlite3.OperationalError:
        sinonimos = []
    conn.close()
    tex = texto or ""
    lower = tex.lower()
    for original, sinonimo in sinonimos:
        if sinonimo in lower:
            tex += f" {original}"
    return tex

def buscar_patron_rapido(pregunta: str) -> Optional[Tuple[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT patron, respuesta FROM respuestas_rapidas")
        patrones = cur.fetchall()
    except sqlite3.OperationalError:
        patrones = []
    conn.close()
    pregunta_norm = normalizar_texto(pregunta)
    for patron, respuesta in patrones:
        if re.search(patron, pregunta_norm, re.IGNORECASE):
            return (respuesta, "patron_rapido")
    return None

def buscar_por_palabras_clave(pregunta: str, umbral: int = 70) -> Optional[Tuple[str, str, int, List[str]]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT pregunta, respuesta, COALESCE(palabras_clave,'') FROM conocimientos")
    conocimientos = cur.fetchall()
    conn.close()
    pregunta_expandida = expandir_sinonimos(pregunta)
    pregunta_norm = normalizar_texto(pregunta_expandida)
    mejores = []
    for preg, resp, palabras in conocimientos:
        score_preg = fuzz.token_set_ratio(pregunta_norm, normalizar_texto(preg)) / 100.0
        score_kw   = fuzz.token_set_ratio(pregunta_norm, normalizar_texto(palabras)) / 100.0 if palabras else 0.0
        final = 0.7*score_preg + 0.3*score_kw
        if final >= (umbral/100.0 - 0.1):
            mejores.append((final, preg, resp))
    if mejores:
        mejores.sort(reverse=True, key=lambda x: x[0])
        top = mejores[0]
        sugerencias = [p for s, p, r in mejores[1:4] if s >= (umbral/100.0 - 0.1)]
        metodo = "busqueda_exacta" if top[0] >= 0.8 else "busqueda_difusa"
        return (top[2], metodo, int(top[0]*100), sugerencias)
    return None

def registrar_uso(pregunta: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        pregunta_norm = normalizar_texto(pregunta)
        cur.execute("SELECT id, pregunta FROM conocimientos")
        conocimientos = cur.fetchall()
        best_id, best = None, 0
        for i, preg in conocimientos:
            s = fuzz.ratio(pregunta_norm, normalizar_texto(preg)) / 100.0
            if s > best:
                best, best_id = s, i
        if best_id and best > 0.7:
            cur.execute("UPDATE conocimientos SET veces_usada = veces_usada + 1 WHERE id = ?", (best_id,))
            conn.commit()
        conn.close()
    except:
        pass

# =====================================================
# LLAMA / OLLAMA
# =====================================================
def llama_generate(user_question: str, context_hint: str = "") -> str:
    """
    Genera respuesta con Ollama LLaMA en modo STREAM y concatena todos los chunks
    hasta 'done'==True. También sube num_predict para evitar cortes.
    """
    try:
        url = f"{OLLAMA_HOST}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": (
                "Eres Alma, un asistente de soporte técnico prudente, claro y profesional. "
                "Responde en español con pasos numerados y recomendaciones seguras. "
                "Si faltan datos, indícalo y pide la información mínima para continuar.\n\n"
                f"Consulta: {user_question}\n"
                f"Contexto: {context_hint}\n"
            ),
            "stream": True,  # <--- CLAVE
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 1024  # <--- súbelo si aún se queda corto (p.ej. 1536/2048)
            }
        }

        # lee el stream línea a línea hasta done
        r = requests.post(url, json=payload, stream=True, timeout=max(LLAMA_TIMEOUT, 180))
        r.raise_for_status()

        partes = []
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except Exception:
                continue

            if "response" in data and data["response"]:
                partes.append(data["response"])

            if data.get("done"):
                break

        return "".join(partes).strip()

    except Exception as e:
        print("⚠️  LLAMA ERROR:", repr(e))
        try:
            # si falló después del request, tal vez hay texto en r.text
            print("LLAMA RESP:", r.text)
        except Exception as inner_e:
            print("No se pudo imprimir respuesta de LLaMA:", repr(inner_e))
        return ""


# =====================================================
# CARGA FAISS
# =====================================================
try:
    faiss_index = faiss.read_index(INDEX_FILE)
    id_map = json.load(open(MAP_FILE, "r", encoding="utf-8"))
    emb_model = SentenceTransformer(MODEL_NAME)
    SEMANTIC_READY = True
except Exception as e:
    print("⚠️ No se pudo cargar índice semántico o modelo. Ejecuta 'python build_index.py' primero.")
    print("Detalle:", e)
    faiss_index, id_map, emb_model = None, [], None
    SEMANTIC_READY = False

def retrieve_semantic(query: str, top_k: int = 5):
    if not SEMANTIC_READY:
        return []
    qv = emb_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = faiss_index.search(qv, top_k)
    hits = []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for pos, score in zip(idx[0], scores[0]):
        if pos < 0:
            continue
        row_id = id_map[pos]
        cur.execute("SELECT pregunta, respuesta, categoria FROM conocimientos WHERE id = ?", (row_id,))
        row = cur.fetchone()
        if row:
            preg, resp, cat = row
            hits.append({"id": row_id, "score": float(score), "pregunta": preg, "respuesta": resp, "categoria": cat})
    conn.close()
    return hits

def rerank_fuzzy(query: str, hits: List[dict]) -> List[dict]:
    for h in hits:
        text = f"{h['pregunta']} {h['respuesta']}"
        h["fuzzy"] = fuzz.token_set_ratio(query.lower(), text.lower()) / 100.0
        h["final"] = 0.7*h["score"] + 0.3*h["fuzzy"]
    return sorted(hits, key=lambda x: x["final"], reverse=True)

# =====================================================
# ENDPOINTS
# =====================================================
@app.get("/")
def root():
    return {
        "asistente": "Alma",
        "descripcion": "Asistente híbrido (FAISS + Fuzzy + LLaMA)",
        "version": "3.1",
        "metodo": "FAISS + Fuzzy + LLaMA fallback",
        "semantic_ready": SEMANTIC_READY
    }

@app.get("/health_llama")
def health_llama():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        ok = (r.status_code == 200)
    except Exception:
        ok = False
    return {
        "llama_enabled": LLAMA_ENABLED,
        "ollama_reachable": ok,
        "model": OLLAMA_MODEL
    }

@app.post("/consulta", response_model=Respuesta)
def consultar(consulta: Consulta):
    """
    Orquesta: patrones → FAISS → Fuzzy → (si score bajo) LLaMA → fallback.
    Forzado temporal a LLaMA si score difuso < 90 para validar integración.
    """
    try:
        pregunta = (consulta.pregunta or "").strip()
        if not pregunta or len(pregunta) < 3:
            raise HTTPException(400, "Pregunta demasiado corta")

        # 1) Patrones rápidos
        patron = buscar_patron_rapido(pregunta)
        if patron:
            return Respuesta(
                nombre_asistente="Alma",
                respuesta=patron[0],
                confianza=100,
                metodo="patron_rapido",
                sugerencias=[],
                debe_escalar=False
            )

        # 2) Recuperación semántica (FAISS + rerank)
        if SEMANTIC_READY:
            cands = retrieve_semantic(pregunta, top_k=5)
            if cands:
                cands = rerank_fuzzy(pregunta, cands)
                top = cands[0]
                if top["final"] >= 0.65:
                    registrar_uso(pregunta)
                    return Respuesta(
                        nombre_asistente="Alma",
                        respuesta=top["respuesta"],
                        confianza=int(top["final"] * 100),
                        metodo="semantic+rerank",
                        sugerencias=[c["pregunta"] for c in cands[1:4]],
                        debe_escalar=top["final"] < 0.7
                    )

        # 3) Búsqueda difusa clásica (fuzzy) + refuerzo LLaMA si score bajo
        resultado = buscar_por_palabras_clave(pregunta, umbral=70)
        if resultado:
            respuesta, metodo, score, sugerencias = resultado

            # Forzado temporal para comprobar LLaMA:
            if score < 90:
                generada = llama_generate(
                    user_question=pregunta,
                    context_hint=f"Coincidencia difusa: {score}% (umbral temporal 90%)."
                )
                if generada:
                    return Respuesta(
                        nombre_asistente="Alma",
                        respuesta=generada,
                        confianza=max(score, 60),
                        metodo="difusa+llama_fallback",
                        sugerencias=sugerencias[:3],
                        debe_escalar=True
                    )

            # Si el score fue alto, mantener fuzzy
            registrar_uso(pregunta)
            return Respuesta(
                nombre_asistente="Alma",
                respuesta=respuesta,
                confianza=score,
                metodo=metodo,
                sugerencias=sugerencias[:3],
                debe_escalar=score < 70
            )

        # 4) Fallback directo con LLaMA
        generada = llama_generate(
            user_question=pregunta,
            context_hint="No se encontró información en la base de conocimiento."
        )
        if generada:
            return Respuesta(
                nombre_asistente="Alma",
                respuesta=generada,
                confianza=60,
                metodo="llama_fallback",
                sugerencias=[
                    "¿Cómo reinicio un servidor?",
                    "Error 500 en la aplicación",
                    "¿Cómo restablezco mi contraseña en windows 10?"
                ],
                debe_escalar=True
            )

        # 5) Fallback estático final
        return Respuesta(
            nombre_asistente="Alma",
            respuesta=(
                "No encontré información específica sobre tu consulta.\n\n"
                "**Para ayudar mejor, indica:**\n"
                "1) Mensaje exacto del error\n"
                "2) Paso que estabas haciendo\n"
                "3) Sistema/versión\n\n"
                "O crea un ticket:\n"
                "- Email: 1@almaai.online\n"
                "- Portal: tickets.almaai.online\n"
                "- Tel: +57 3232897785"
            ),
            confianza=0,
            metodo="fallback",
            sugerencias=[
                "¿Cómo reinicio un servidor?",
                "Error 500 en la aplicación",
                "¿Cómo restablezco mi contraseña en windows 10?"
            ],
            debe_escalar=True
        )

    except HTTPException:
        raise
    except Exception as e:
        print("APP ERROR en /consulta:", repr(e))
        raise HTTPException(500, f"Error procesando consulta: {e}")

@app.get("/estadisticas")
def estadisticas():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM conocimientos")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT pregunta, veces_usada FROM conocimientos
        WHERE veces_usada > 0 ORDER BY veces_usada DESC LIMIT 5
    """)
    mas = [{"pregunta": p, "usos": u} for p, u in cur.fetchall()]
    cur.execute("SELECT categoria, COUNT(*) FROM conocimientos GROUP BY categoria")
    por_cat = dict(cur.fetchall())
    conn.close()
    return {
        "asistente": "Alma",
        "total_conocimientos": total,
        "por_categoria": por_cat,
        "mas_consultadas": mas,
        "semantic_ready": SEMANTIC_READY
    }
