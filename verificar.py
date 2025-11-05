import sqlite3

DB_PATH = "soporte.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# total de registros
cur.execute("SELECT COUNT(*) FROM conocimientos")
total = cur.fetchone()[0]
print(f"Total registros: {total}")

# registros por categoría
cur.execute("SELECT categoria, COUNT(*) FROM conocimientos GROUP BY categoria")
print("\nRegistros por categoría:")
for cat, cnt in cur.fetchall():
    print(f" - {cat}: {cnt}")

# ejemplos
cur.execute("SELECT pregunta, respuesta FROM conocimientos LIMIT 3")
print("\nEjemplos de registros:")
for preg, resp in cur.fetchall():
    print(f"  > {preg}\n    {resp[:80]}...\n")

conn.close()
