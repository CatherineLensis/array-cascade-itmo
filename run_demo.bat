@echo off
setlocal

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=server"

set "PYTHON_EXE="
call :accept_python_path "%CD%\.venv\Scripts\python.exe"
call :accept_python_path "%CD%\venv\Scripts\python.exe"
call :accept_python_path "E:\ML_works_folder\Fast_prototype\venv\Scripts\python.exe"
call :accept_python_path "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe"
call :accept_python_cmd "py -3.11"
call :accept_python_cmd "py -3"
call :accept_python_cmd "python"

if not defined PYTHON_EXE (
    echo [ERROR] Compatible Python not found.
    echo [ERROR] Need Python with torch, fastapi and uvicorn installed.
    exit /b 1
)

if /I "%MODE%"=="server" goto :server
if /I "%MODE%"=="smoke" goto :smoke
if /I "%MODE%"=="assets" goto :assets
if /I "%MODE%"=="help" goto :help
if /I "%MODE%"=="--help" goto :help
if /I "%MODE%"=="-h" goto :help

echo [ERROR] Unknown mode: %MODE%
echo.
goto :help

:server
echo [INFO] Starting Array Cascade ITMO demo server...
echo [INFO] Project root: %CD%
echo [INFO] URL: http://127.0.0.1:8080
start "" http://127.0.0.1:8080
call %PYTHON_EXE% -X utf8 ui\app.py
exit /b %ERRORLEVEL%

:smoke
echo [INFO] Running UI smoke tests...
call %PYTHON_EXE% -X utf8 -m pytest tests\ui\test_routes.py
exit /b %ERRORLEVEL%

:assets
echo [INFO] Capturing demo screenshots from the running UI...
call %PYTHON_EXE% -X utf8 scripts\package_demo_assets.py --skip-video
exit /b %ERRORLEVEL%

:help
echo Usage:
echo   run_demo.bat                ^(same as: run_demo.bat server^)
echo   run_demo.bat server         Start FastAPI demo server and open browser
echo   run_demo.bat smoke          Run UI/API smoke tests
echo   run_demo.bat assets         Recapture UI screenshots ^(server must already be running^)
echo.
echo Notes:
echo   - The script auto-detects a Python with torch + fastapi + uvicorn.
echo   - Main demo page: http://127.0.0.1:8080
echo   - If ML weights are missing, UI will still start but some paths may use physics fallback.
exit /b 0

:accept_python_path
if defined PYTHON_EXE exit /b 0
if not exist "%~1" exit /b 0
call "%~1" -X utf8 -c "import torch, fastapi, uvicorn" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYTHON_EXE="%~1""
exit /b 0

:accept_python_cmd
if defined PYTHON_EXE exit /b 0
call %~1 -X utf8 -c "import torch, fastapi, uvicorn" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYTHON_EXE=%~1"
exit /b 0
