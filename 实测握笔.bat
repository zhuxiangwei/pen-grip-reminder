@echo off
rem ================================================================
rem  Pen-grip probe launcher  (left/right angle comparison)
rem  ASCII-only on purpose: cmd.exe reads .bat as GBK.
rem ================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set PY=envs\pg\Scripts\python.exe

if not exist "%PY%" (
  echo.
  echo [ERROR] Python environment not found:
  echo         %PY%
  echo.
  echo   Create it once with:
  echo     python -m venv envs\pg
  echo     envs\pg\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
  echo     envs\pg\Scripts\python.exe -m pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple mediapipe
  echo     envs\pg\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple absl-py certifi flatbuffers numpy
  echo.
  pause
  exit /b 1
)

if not exist "bench\models\hand_landmarker.task" (
  echo Downloading the hand model first...
  "%PY%" tools\fetch_models.py
  echo.
)

echo Starting probe.  Camera index 0 by default.
echo If the wrong camera opens, close this and run:
echo   envs\pg\Scripts\python.exe hand_probe.py --camera 1
echo.
"%PY%" hand_probe.py --camera 0

echo.
echo Probe exited.
pause
