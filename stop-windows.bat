@echo off
chcp 65001 >nul
echo ====================================
echo Stopping SuperBizAgent
echo ====================================
echo.

echo [1/4] Stopping FastAPI service...
taskkill /FI "WINDOWTITLE eq SuperBizAgent API*" /F >nul 2>&1
if errorlevel 1 (
    echo [INFO] FastAPI service is not running or already stopped
) else (
    echo [OK] FastAPI service stopped
)
echo.

echo [2/4] Stopping CLS MCP service...
taskkill /FI "WINDOWTITLE eq CLS MCP Server*" /F >nul 2>&1
if errorlevel 1 (
    echo [INFO] CLS MCP service is not running or already stopped
) else (
    echo [OK] CLS MCP service stopped
)
echo.

echo [3/4] Stopping Monitor MCP service...
taskkill /FI "WINDOWTITLE eq Monitor MCP Server*" /F >nul 2>&1
if errorlevel 1 (
    echo [INFO] Monitor MCP service is not running or already stopped
) else (
    echo [OK] Monitor MCP service stopped
)
echo.

echo [4/4] Stopping Milvus containers...
docker ps --format "{{.Names}}" | findstr "milvus" >nul 2>&1
if not errorlevel 1 (
    docker compose -f vector-database.yml down
    if errorlevel 1 (
        echo [ERROR] Failed to stop Docker containers
    ) else (
        echo [OK] Milvus containers stopped
    )
) else (
    echo [INFO] Milvus containers are not running
)
echo.

echo ====================================
echo All services stopped
echo ====================================
echo.
echo Tip:
echo   If you need to fully remove Docker data, run:
echo   docker compose -f vector-database.yml down -v
echo.
pause
goto :eof

REM 保留的中文注释：
REM 停止 SuperBizAgent 服务
REM 停止 FastAPI 服务
REM 停止 CLS MCP 服务
REM 停止 Monitor MCP 服务
REM 停止 Docker 容器
REM 停止 Milvus 容器
REM 所有服务已停止
REM 如需完全清理 Docker 数据卷，运行 docker compose -f vector-database.yml down -v