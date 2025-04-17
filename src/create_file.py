import pandas as pd
import os
import sqlite3

# Cambia la ruta si los archivos están en otro lugar
current_dir = os.getcwd()
path_en = os.path.join(current_dir, r"text\en-es\CCMatrix.en-es.en")
path_es = os.path.join(current_dir, r"text\en-es\CCMatrix.en-es.es")
path_scores = os.path.join(current_dir, r"text\en-es\CCMatrix.en-es.scores")
db_path = os.path.join(current_dir, r"text\en-es\ccmatrix.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS translations (
    english TEXT,
    spanish TEXT,
    score REAL
)
''')

with open(path_en, encoding="utf-8") as f_en, \
     open(path_es, encoding="utf-8") as f_es, \
     open(path_scores, encoding="utf-8") as f_scores:

    for i, (en, es, sc) in enumerate(zip(f_en, f_es, f_scores)):
        cursor.execute("INSERT INTO translations VALUES (?, ?, ?)", 
                       (en.strip(), es.strip(), float(sc.strip())))

        if i % 100000 == 0:
            print(f"{i} líneas insertadas...")
            conn.commit()  # guarda cada bloque

conn.commit()
conn.close()
print("🎉 Datos guardados en ccmatrix.db")