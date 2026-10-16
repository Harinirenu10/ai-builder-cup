@echo off
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0infra\make.py" %*
) else (
    python "%~dp0infra\make.py" %*
)
