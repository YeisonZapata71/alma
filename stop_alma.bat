@echo off
title STOP ALMA
echo Cerrando procesos de ALMA...

REM Matar uvicorn, http.server y ollama serve (solo los lanzados por estas ventanas)
for /f "tokens=2" %%p in ('tasklist ^| findstr /i "uvicorn.exe"') do taskkill /PID %%p /F >nul 2>nul
for /f "tokens=2" %%p in ('tasklist ^| findstr /i "python.exe" ^| findstr /i "http.server"') do taskkill /PID %%p /F >nul 2>nul

REM Si quieres parar tambien el servicio de Ollama lanzado en esta sesion:
for /f "tokens=2" %%p in ('tasklist ^| findstr /i "ollama.exe"') do taskkill /PID %%p /F >nul 2>nul

echo Listo.
pause
