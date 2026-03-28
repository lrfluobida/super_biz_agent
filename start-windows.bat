@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ====================================
echo Starting SuperBizAgent
echo ====================================
echo.

echo [1/7] Checking package manager...
where uv >nul 2>&1
if errorlevel 1 (
    echo [INFO] uv not found, using pip
    echo [TIP] Install uv for faster dependency sync: pip install uv
    set USE_UV=0
) else (
    echo [OK] uv detected
    set USE_UV=1
)
echo.

echo [2/7] Preparing Python version...
if exist .python-version (
    set /p PYTHON_VERSION=<.python-version
    echo [INFO] Current version: !PYTHON_VERSION!
    echo !PYTHON_VERSION! | findstr /C:"3.10" >nul
    if not errorlevel 1 (
        echo [WARN] Python 3.10 is not supported, switching to 3.13
        echo 3.13> .python-version
        echo [OK] Updated to Python 3.13
    )
) else (
    echo [INFO] Creating .python-version
    echo 3.13> .python-version
)
echo.

echo [3/7] Preparing virtual environment...
if exist .venv\Scripts\python.exe (
    echo [INFO] Existing virtual environment found
    if "%USE_UV%"=="1" (
        uv sync 2>nul
        if errorlevel 1 (
            echo [WARN] uv sync failed, falling back to pip install -e .
            call :ensure_pip .venv\Scripts\python.exe
            .venv\Scripts\python.exe -m pip install -e . -q
            if errorlevel 1 (
                echo [ERROR] Failed to update dependencies with pip
                pause
                exit /b 1
            )
        ) else (
            echo [OK] Dependency sync completed with uv
        )
    ) else (
        echo [INFO] Updating dependencies with pip...
        call :ensure_pip .venv\Scripts\python.exe
        .venv\Scripts\python.exe -m pip install -e . -q
        if errorlevel 1 (
            echo [ERROR] Failed to update dependencies with pip
            pause
            exit /b 1
        )
    )
) else (
    echo [INFO] Creating virtual environment...
    if "%USE_UV%"=="1" (
        echo [INFO] Trying uv sync first...
        uv sync 2>nul
        if not errorlevel 1 (
            echo [OK] Virtual environment created with uv
            goto :venv_created
        )
        echo [WARN] uv sync failed, falling back to python -m venv
    )

    echo [INFO] Running python -m venv .venv
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo [TIP] Make sure Python 3.11+ is installed
        pause
        exit /b 1
    )

    call :ensure_pip .venv\Scripts\python.exe
    echo [INFO] Installing project dependencies. This may take a few minutes...
    .venv\Scripts\python.exe -m pip install --upgrade pip -q
    .venv\Scripts\python.exe -m pip install -e . -q
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully
)

:venv_created
echo [OK] Virtual environment ready
echo.

set PYTHON_CMD=.venv\Scripts\python.exe

echo [4/7] Starting Milvus containers...
docker ps --format "{{.Names}}" | findstr "milvus-standalone" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Milvus is already running
) else (
    docker compose -f vector-database.yml up -d
    if errorlevel 1 (
        echo [ERROR] Failed to start Docker containers. Make sure Docker Desktop is running.
        pause
        exit /b 1
    )
    echo [INFO] Waiting 10 seconds for Milvus startup...
    timeout /t 10 /nobreak >nul
)
echo [OK] Milvus is ready
echo.

echo [5/7] Starting CLS MCP server...
if not exist logs mkdir logs
start "CLS MCP Server" /min cmd /c ""%PYTHON_CMD%" mcp_servers\cls_server.py > logs\mcp_cls.log 2>&1"
timeout /t 2 /nobreak >nul
echo [OK] CLS MCP server started
echo.

echo [6/7] Starting Monitor MCP server...
start "Monitor MCP Server" /min cmd /c ""%PYTHON_CMD%" mcp_servers\monitor_server.py > logs\mcp_monitor.log 2>&1"
timeout /t 2 /nobreak >nul
echo [OK] Monitor MCP server started
echo.

echo [7/7] Starting FastAPI service...
start "SuperBizAgent API" /min cmd /c ""%PYTHON_CMD%" -m uvicorn app.main:app --host 0.0.0.0 --port 9900 > logs\uvicorn_start.log 2>&1"

set API_READY=0
set API_WAIT_COUNT=0
:wait_api_health
set /a API_WAIT_COUNT+=1
curl -s http://localhost:9900/health >nul 2>&1
if not errorlevel 1 (
    set API_READY=1
    goto :api_health_done
)
if %API_WAIT_COUNT% geq 30 goto :api_health_done
echo [INFO] Waiting for API health... (%API_WAIT_COUNT%/30)
timeout /t 2 /nobreak >nul
goto :wait_api_health
:api_health_done
echo.

if "%API_READY%"=="1" (
    echo [OK] FastAPI service is healthy
    echo.
    echo [INFO] Uploading aiops-docs into the vector store...
    for %%f in (aiops-docs\*.md) do (
        echo   Uploading: %%~nxf
        curl -s -X POST http://localhost:9900/api/upload -F "file=@%%f" >nul 2>&1
    )
    echo [OK] Document upload completed
) else (
    echo [WARN] API did not become healthy within 60 seconds.
    echo [TIP] Check logs\uvicorn_start.log for startup errors.
)

echo.
echo ====================================
echo Startup complete
echo ====================================
echo Web UI: http://localhost:9900
echo API Docs: http://localhost:9900/docs
echo.
echo Logs:
echo   FastAPI startup: logs\uvicorn_start.log
echo   FastAPI runtime: logs\app_*.log
echo   CLS MCP: logs\mcp_cls.log
echo   Monitor MCP: logs\mcp_monitor.log
echo Stop services with: stop-windows.bat
echo ====================================
pause
goto :eof

:ensure_pip
%~1 -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] pip not found in virtual environment. Running ensurepip...
    %~1 -m ensurepip --upgrade >nul
)
exit /b 0

REM 保留的中文注释：
REM 检查 uv 是否安装（可选，如果没有会使用 pip）
REM 确保 Python 版本正确
REM 检查是否为 3.10（不兼容）
REM 创建或同步虚拟环境
REM 如果有 uv，尝试使用 uv sync
REM 使用传统 Python venv 创建
REM 安装依赖
REM 设置 Python 命令
REM 启动 Docker Compose
REM 启动 Milvus 向量数据库
REM 启动 CLS MCP 服务
REM 启动 Monitor MCP 服务
REM 启动 FastAPI 服务
REM 检查服务状态并上传文档
REM 调用 API 上传 aiops-docs 文档到向量数据库