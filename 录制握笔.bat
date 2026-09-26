@echo off
rem ================================================================
rem  Record a clip with the built-in laptop camera.
rem  Use this when the lid must be tilted forward (screen unusable).
rem  ASCII-only on purpose: cmd.exe reads .bat as GBK.
rem ================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set PY=envs\pg\Scripts\python.exe
if not exist "%PY%" (
  echo [ERROR] Python environment not found: %PY%
  echo         See README.md for setup steps.
  pause
  exit /b 1
)

"%PY%" tools\record.py --seconds 30 %*

echo.
pause
