cp soporte.db soporte_backup_$(date +%Y%m%d).db

---------------------------BACKUP----------------------------------

. .venv/Scripts/activate
------------ACTIVAR VENV-------------

python importar_csv.py faqs.csv soporte.db
-----------EJECUTAR IMPORTADOR-----------------------

python verificar.py
--VERIFICAR CONTEO---



