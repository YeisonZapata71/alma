# importar_csv_v2.py
# Uso:
#   python importar_csv_v2.py                # lee faqs.csv
#   python importar_csv_v2.py faqs_soporte_tecnico_full.csv

import csv, sqlite3, re, sys, unicodedata
from pathlib import Path

DB_PATH = Path("soporte.db")
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("faqs.csv")

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def norm_key(s: str) -> str:
    s = s or ""
    s = strip_accents(s).lower().strip()
    s = re.sub(r"[\s_\-]+", "", s)  # quita espacios/guiones
    return s

# posibles nombres de columnas (acepta sinónimos)
COL_PREG = {"pregunta","question","preguntafaq","titulo","consulta"}
COL_RESP = {"respuesta","answer","solucion","descripcion","contenido"}
COL_CAT  = {"categoria","categoría","category","seccion","grupo"}

def abrir_csv(path: Path):
    # intentamos UTF-8 (con BOM o sin) y si no latin-1
    encodings = ["utf-8-sig","utf-8","latin-1"]
    sniff_sample = None
    for enc in encodings:
        try:
            f = open(path, "r", encoding=enc, newline="")
            sniff_sample = f.read(4096)
            f.seek(0)
            # detectar delimitador
            try:
                dialect = csv.Sniffer().sniff(sniff_sample, delimiters=",;\t|")
                has_header = csv.Sniffer().has_header(sniff_sample)
            except csv.Error:
                dialect = csv.get_dialect("excel")
                dialect.delimiter = ","
                has_header = True
            return f, dialect, has_header, enc
        except UnicodeError:
            continue
    raise RuntimeError("No pude abrir el CSV con UTF-8 ni latin-1.")

def detectar_campos(fieldnames):
    # mapea encabezados reales -> estandarizados
    mapping = {"pregunta":None, "respuesta":None, "categoria":None}
    report = {}
    for k in fieldnames:
        nk = norm_key(k)
        report[k] = nk
        if nk in COL_PREG and mapping["pregunta"] is None:
            mapping["pregunta"] = k
        elif nk in COL_RESP and mapping["respuesta"] is None:
            mapping["respuesta"] = k
        elif nk in COL_CAT and mapping["categoria"] is None:
            mapping["categoria"] = k
    ok = all(mapping.values())
    return ok, mapping, report

def normalizar_kw(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[¿?¡!.,;:()'\"/\\\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

PRIORIDAD_POR_CAT = {
    "Acceso": 1, "Errores": 1, "Servidores": 1,
    "Rendimiento": 2, "Reportes": 2, "Archivos": 2
}
KW_EXTRA = {
    "Acceso": "login contraseña clave password restablecer reset acceso no abre",
    "Errores": "error 500 404 server internal fallo problema issue",
    "Servidores": "servidor server reinicio reboot restart servicios",
    "Rendimiento": "lento lentitud performance demora tarda",
    "Reportes": "reporte exportar descargar excel pdf csv informe",
    "Archivos": "archivo subir cargar upload adjuntar formato tamaño"
}

# --- Crear tabla si no existe
conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS conocimientos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pregunta TEXT NOT NULL,
  respuesta TEXT NOT NULL,
  categoria TEXT,
  palabras_clave TEXT,
  prioridad INTEGER DEFAULT 1,
  veces_usada INTEGER DEFAULT 0
)""")
conn.commit()

print(f"📄 Archivo: {CSV_PATH}")
f, dialect, has_header, used_enc = abrir_csv(CSV_PATH)
print(f"✔ Codificación: {used_enc} | Delimitador detectado: '{dialect.delimiter}' | Header: {has_header}")

reader = csv.DictReader(f, dialect=dialect)
ok, mapping, rep = detectar_campos(reader.fieldnames or [])
print(f"Encabezados originales → normalizados:\n  " + "\n  ".join([f"{k} → {v}" for k,v in rep.items()]))

if not ok:
    f.close()
    print("\n❌ No encuentro las columnas requeridas.")
    print("   Necesito algo equivalente a: pregunta, respuesta, categoria")
    print("   Puedes renombrarlas en el CSV o pasarlo a este script con esos nombres.")
    raise SystemExit(1)

preg_k = mapping["pregunta"]; resp_k = mapping["respuesta"]; cat_k = mapping["categoria"]
print(f"\nUsando columnas → pregunta:'{preg_k}'  respuesta:'{resp_k}'  categoria:'{cat_k}'")

insertados, vacios, errores = 0, 0, 0
for i, row in enumerate(reader, start=1):
    try:
        preg = (row.get(preg_k) or "").strip()
        resp = (row.get(resp_k) or "").strip()
        cat  = (row.get(cat_k)  or "General").strip()
        if not preg or not resp:
            vacios += 1
            continue
        base_kw = normalizar_kw(preg)
        extra   = KW_EXTRA.get(cat, "")
        palabras = f"{base_kw} {extra}".strip()
        prioridad = PRIORIDAD_POR_CAT.get(cat, 2)
        cur.execute("""
          INSERT INTO conocimientos (pregunta, respuesta, categoria, palabras_clave, prioridad)
          VALUES (?, ?, ?, ?, ?)
        """, (preg, resp, cat, palabras, prioridad))
        insertados += 1
    except Exception as e:
        errores += 1
        print(f"  ⚠ Fila {i} con error: {e}")

conn.commit(); conn.close(); f.close()
print(f"\n✅ Importadas {insertados} filas desde {CSV_PATH} a {DB_PATH}")
if vacios:  print(f"ℹ Filas saltadas por campos vacíos: {vacios}")
if errores: print(f"ℹ Filas con error: {errores}")

