@ECHO OFF
SETLOCAL
CD /D "%~dp0"

IF NOT EXIST "envs\pg\Scripts\python.exe" (
  ECHO [ERROR] venv not found: envs\pg
  ECHO   Run : python -m venv envs\pg
  ECHO   Then: envs\pg\Scripts\pip install -r requirements.txt
  PAUSE
  GOTO :EOF
)

ECHO [START] Pen-grip detector window...
ECHO   If it fails, full output is in app_console.log (and app_error.log)
envs\pg\Scripts\python.exe app.py %* > app_console.log 2>&1
IF ERRORLEVEL 1 (
  ECHO.
  ECHO [ERROR] exited with code %ERRORLEVEL%
  ECHO   See app_console.log / app_error.log
  ECHO ---- app_console.log (last 25 lines) ----
  TYPE app_console.log
  PAUSE
)
ENDLOCAL
