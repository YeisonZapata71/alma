# build_index.py
# Genera index.faiss e index_map.json desde soporte.db (tabla conocimientos)

import sqlite3, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

DB = "soporte.db"
INDEX_FILE = "index.faiss"
MAP_FILE = "index_map.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB, rápido y muy bueno

print("Cargando BD...")
conn = sqlite3.connect(DB); cur = conn.cursor()
cur.execute("SELECT id, pregunta, respuesta, COALESCE(palabras_clave,'') FROM conocimientos")
rows = cur.fetchall()
conn.close()

if not rows:
    raise SystemExit("No hay filas en 'conocimientos'. Importa datos primero.")

print(f"Filas: {len(rows)}  | Cargando modelo: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

texts = [f"{p} || {r} || {k}" for (i,p,r,k) in rows]
embs = model.encode(texts, normalize_embeddings=True)  # (N, d) float32 recomendado
X = np.asarray(embs, dtype="float32")

print("Creando índice FAISS (dot-product sobre vectores normalizados = cos-sim)...")
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

faiss.write_index(index, INDEX_FILE)
json.dump([i for (i,_,_,_) in rows], open(MAP_FILE, "w", encoding="utf-8"))
print(f"✅ Listo: {INDEX_FILE} y {MAP_FILE}")
