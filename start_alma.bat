@echo on
setlocal
cd /d %~dp0

REM 1) OLLAMA
start "OLLAMA SERVER" /MIN cmd /k "ollama serve"

REM 2) Backend
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=llama3:8b
set PYTHONIOENCODING=utf-8
start "ALMA BACKEND" cmd /k ".venv\Scripts\activate.bat && uvicorn app_semantico:app --host 127.0.0.1 --port 8000 --reload"

REM 3) UI
start "ALMA UI" cmd /k ".venv\Scripts\activate.bat && cd ui && python -m http.server 8080"

REM 4) Navegador
start "" "http://127.0.0.1:8080"
