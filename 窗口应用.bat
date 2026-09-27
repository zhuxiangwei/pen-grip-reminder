@ECHO OFF
REM ============================================================
REM  握笔姿势检测 · 窗口应用（双击运行）
REM  chcp 65001 让 cmd 以 UTF-8 读本文件，避免中文注释/提示被 GBK 误解析
REM  并把 python 全部输出(含报错)落盘到 app_console.log，出错时暂停不关窗
REM ============================================================
chcp 65001 >nul
SETLOCAL
CD /D "%~dp0"

IF NOT EXIST "envs\pg\Scripts\python.exe" (
  ECHO [错误] 没找到虚拟环境 envs\pg
  ECHO   请先在本目录执行： python -m venv envs\pg
  ECHO   然后安装依赖：    envs\pg\Scripts\pip install -r requirements.txt
  PAUSE
  GOTO :EOF
)

ECHO [启动] 握笔姿势检测窗口应用…
ECHO   （若启动失败，详细报错已写入 app_console.log，也可看 app_error.log）
envs\pg\Scripts\python.exe app.py %* > app_console.log 2>&1
IF ERRORLEVEL 1 (
  ECHO.
  ECHO [错误] 程序退出码 %ERRORLEVEL%
  ECHO   详细报错见 app_console.log（和 app_error.log）
  ECHO ---- app_console.log 末尾 20 行 ----
  TYPE app_console.log
  PAUSE
)
ENDLOCAL
