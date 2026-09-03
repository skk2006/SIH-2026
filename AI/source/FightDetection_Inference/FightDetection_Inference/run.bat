@echo off
title Fight Detection System

echo.
echo ============================================================
echo              FIGHT DETECTION SYSTEM
echo ============================================================
echo.

if "%~1"=="" (
    echo ERROR: No video supplied.
    echo.
    echo Usage:
    echo run.bat "test_videos\fight.mp4"
    echo.
    pause
    exit /b 1
)

echo Input video:
echo %~1
echo.

if not exist "%~1" (
    echo ERROR: Video file not found.
    echo.
    pause
    exit /b 1
)

if not exist "outputs" mkdir outputs

echo Starting fight detection...
echo.

python code\inference.py ^
    --source "%~1" ^
    --weights "models\best_model.pt" ^
    --pose_model "yolov8n-pose.pt" ^
    --output "outputs\result.mp4"

echo.
echo ============================================================
echo Detection finished.
echo ============================================================
echo.
echo Result:
echo outputs\result.mp4
echo.

pause
