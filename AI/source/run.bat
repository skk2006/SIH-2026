@echo off
title SENTINEL-AI Server
:loop
echo ===================================================
echo   Starting SENTINEL-AI Application...
echo   (Press Ctrl+C in this window to stop the server)
echo ===================================================
..\..\.venv\Scripts\python.exe app.py
echo.
echo ===================================================
echo   [SENTINEL-AI] Process stopped. Restarting in 2s...
echo   (Press Ctrl+C to close completely)
echo ===================================================
timeout /t 2 /nobreak >nul
goto loop
