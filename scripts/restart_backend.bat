@echo off
echo ========================================
echo Restarting Backend Server
echo ========================================
echo.

echo Stopping any running backend processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *uvicorn*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting backend server...
cd /d "%~dp0.."
start "Indoor Navigation Backend" python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

echo.
echo ========================================
echo Backend server is starting...
echo Check the new window for logs
echo ========================================
echo.
echo The server should be running at:
echo http://localhost:8000
echo.
pause
