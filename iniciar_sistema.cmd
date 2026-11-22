@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Ambiente .venv nao encontrado. Consulte a instalacao no README.md.
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -I "%~dp0iniciar_sistema.py" %*
exit /b %errorlevel%
