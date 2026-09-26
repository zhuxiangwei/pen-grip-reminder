@ECHO OFF
REM ============================================================
REM  握笔姿势检测 · 窗口应用（双击运行）
REM  ASCII-only 文件名/内容，避免 cmd 读 GBK 乱码
REM ============================================================
SETLOCAL
CD /D "%~dp0"

IF NOT EXIST "envs\pg\Scripts\python.exe" (
  ECHO [错] 没找到虚拟环境 envs\pg
  ECHO      先跑一次： python -m venv envs\pg  然后装依赖
  PAUSE
  GOTO :EOF
)

ECHO [启动] 握笔姿势检测窗口…
envs\pg\Scripts\python.exe app.py %*
IF ERRORLEVEL 1 (
  ECHO.
  ECHO [错] 程序退出码 %ERRORLEVEL%
  PAUSE
)
ENDLOCAL
