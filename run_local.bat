@echo off
setlocal
title OushudhCheck - Local Full-Stack Launcher

cd /d "%~dp0"

echo ============================================================================
echo   OushudhCheck: Multi-Strip Medicine Safety Scanner ^& Bangla Voice Assistant
echo ============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Creating virtual environment...
    python -m venv .venv
)

echo.
echo [1/5] Verifying Python dependencies in .venv...
.\.venv\Scripts\python.exe -m pip install -q -r backend\requirements.txt -r bot\requirements.txt

echo [2/5] Starting Gemma 3 1B Bangla SLM (llama.cpp llama-server) on http://127.0.0.1:8081 ...
if exist "models\oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf" (
    start "OushudhCheck Gemma 3 1B SLM (Port 8081)" cmd /k "llama-server -m models\oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf --host 127.0.0.1 --port 8081 -c 8192 -np 2 -fa on --cache-reuse 512"
) else (
    echo [WARN] models\oushudhcheck-gemma3-1b-bn.Q4_K_M.gguf not found: backend will use fallback summary mode.
)

echo [3/5] Starting FastAPI Backend Server on http://localhost:8000 ...
start "OushudhCheck FastAPI Backend (Port 8000)" cmd /k ".\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"

echo [4/5] Starting Telegram Bot (@OushudhCheck_bot)...
start "OushudhCheck Telegram Bot" cmd /k ".\.venv\Scripts\python.exe bot\telegram_bot.py"

echo [5/5] Starting Next.js Frontend on http://localhost:3000 ...
start "OushudhCheck Next.js Frontend (Port 3000)" cmd /k "cd /d frontend && npm run dev"

echo.
echo ============================================================================
echo   All services launched in separate windows!
echo   - Frontend Web App : http://localhost:3000
echo   - Strip Scanner UI : http://localhost:3000/scan
echo   - Medicine Database: http://localhost:3000/database
echo   - FastAPI Docs     : http://localhost:8000/docs
echo   - Gemma 3 1B SLM   : http://127.0.0.1:8081
echo   - Telegram Bot     : https://t.me/OushudhCheck_bot
echo ============================================================================
pause
