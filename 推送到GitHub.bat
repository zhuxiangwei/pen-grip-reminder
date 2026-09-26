@echo off
rem ================================================================
rem  Push local repo to GitHub
rem  (ASCII-only content on purpose: cmd.exe reads .bat as GBK, and a
rem   BOM or non-ASCII byte here would break the first line.)
rem ================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist "envs\pg\Scripts\python.exe" (
  if exist "..\posture-reminder\envs\posture\Scripts\python.exe" (
    set PY=..\posture-reminder\envs\posture\Scripts\python.exe
  ) else (
    echo.
    echo [ERROR] No Python environment found.
    echo         Expected one of:
    echo           envs\pg\Scripts\python.exe
    echo           ..\posture-reminder\envs\posture\Scripts\python.exe
    echo         See README.md for setup steps.
    echo.
    pause
    exit /b 1
  )
) else (
  set PY=envs\pg\Scripts\python.exe
)

"%PY%" tools\push_to_github.py %*

echo.
pause
