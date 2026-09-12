@echo off
cd /d "%~dp0"
title Lingyin Lite v1.0.0

echo ============================================================
echo   Lingyin Lite v1.0.0
echo ============================================================
echo.

set "VENV_PY=%~dp0venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    echo [INFO] Using project venv
    "%VENV_PY%" run.py %*
    goto :end
)

where python >nul 2>&1
if %errorlevel%==0 (
    echo [INFO] Using system Python to bootstrap venv
    python run.py %*
    goto :end
)

echo [ERROR] Python not found. Install Python 3.10+ and enable "Add to PATH".
echo https://www.python.org/downloads/
echo.
pause
exit /b 1

:end
echo.
echo Press any key to exit...
pause >nul
