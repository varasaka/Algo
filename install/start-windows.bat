@echo off
REM Algo Agent launcher for Windows. Double-click to install (first run) and start.
cd /d "%~dp0\.."
where python >nul 2>nul || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
if not exist .venv (
  echo First run: creating environment...
  python -m venv .venv || (pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -e . || (pause & exit /b 1)
)
echo Starting Algo Agent... (close this window to stop it)
.venv\Scripts\python -m algo_agent app %*
pause
