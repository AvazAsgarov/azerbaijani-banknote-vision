@echo off
title AZN-Vision: Mobile App & GPU Vision Bridge
color 0A

echo ====================================================================
echo        AZERBAIJANI BANKNOTE VISION (AZN-VISION)
echo   Smart Glasses Companion & Mobile App Launcher (Expo SDK 57)
echo ====================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Node.js and Python runtime environment...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not found! Please install from https://nodejs.org/
    goto FAILED
)

set "PYTHON_CMD="
if exist "C:\Python312\python.exe" set "PYTHON_CMD=C:\Python312\python.exe"
if not defined PYTHON_CMD (
    where python >nul 2>&1 && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERROR] Python is not found! Please install Python 3.10+
    goto FAILED
)

echo [OK] Runtime ready.

echo.
echo [2/3] Checking GPU Inference Bridge (Port 8000)...
"%PYTHON_CMD%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1.5)" >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting GPU Inference Bridge on Port 8000...
    start "AZN-Vision GPU Bridge (Port 8000)" /D "%~dp0" cmd /k ""%PYTHON_CMD%" "%~dp0scripts\mobile_bridge.py" --host 0.0.0.0 --port 8000"
    timeout /t 3 /nobreak >nul
) else (
    echo [OK] GPU Inference Bridge is active on port 8000.
)

echo.
echo [3/3] Resolving Network Interfaces and Configuring Mobile App...
echo ====================================================================

rem Dynamically detect local Wi-Fi / LAN IP via outbound socket connection
for /f "delims=" %%a in ('"%PYTHON_CMD%" -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()"') do set "WIFI_IP=%%a"
if not defined WIFI_IP set "WIFI_IP=192.168.0.142"

set "REACT_NATIVE_PACKAGER_HOSTNAME=%WIFI_IP%"
set "EXPO_PUBLIC_VISION_BRIDGE_URL=http://%WIFI_IP%:8000"

rem Auto-configure mobile/.env file with active host IP
echo EXPO_PUBLIC_VISION_BRIDGE_URL=http://%WIFI_IP%:8000> "%~dp0mobile\.env"

echo [OK] Active Wi-Fi Host IP : %WIFI_IP%
echo [OK] GPU Inference Bridge : http://%WIFI_IP%:8000
echo [OK] Web Monitor & Stream : http://%WIFI_IP%:8000/view
echo.
echo ====================================================================
echo SELECT CONNECTION MODE:
echo   [1] LAN MODE (Recommended - Direct Local Wi-Fi, Real-Time 30 FPS)
echo       - Direct connection over your Wi-Fi router (zero cloud latency).
echo       - Requires iPhone to be on the same Wi-Fi (%WIFI_IP%).
echo   [2] TUNNEL MODE (Ngrok Cloud Tunnel)
echo       - Use if devices are on different networks or cellular data.
echo   [3] WEB MODE (Browser Preview)
echo       - Opens web preview directly in your PC browser.
echo ====================================================================
echo.
echo Launching [1] LAN MODE automatically in 5 seconds (or press 1, 2, 3)...
choice /c 123 /t 5 /d 1 /m "Enter your choice [1, 2, 3]: "
set "MODE_CHOICE=%errorlevel%"

rem Terminate any stale process occupying Metro bundler port 8081
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1

cd /d "%~dp0mobile"

if "%MODE_CHOICE%"=="1" (
    echo.
    echo ====================================================================
    echo [OK] Starting LAN Mode (exp://%WIFI_IP%:8081)
    echo Make sure your iPhone is connected to the same Wi-Fi network!
    echo Scan the QR code below with the Camera / Expo Go app on your phone.
    echo ====================================================================
    call npx expo start --host lan --clear
) else if "%MODE_CHOICE%"=="2" (
    echo.
    echo ====================================================================
    echo [OK] Starting Tunnel Mode (Global ngrok cloud tunnel)
    echo Scan the QR code below with the Expo Go app on your phone.
    echo ====================================================================
    call npx expo start --tunnel --clear
) else (
    echo.
    echo ====================================================================
    echo [OK] Starting Web Mode...
    echo ====================================================================
    call npx expo start --web
)
goto FINISHED

:FAILED
echo.
echo [ERROR] Startup failed! Check your Node.js and Python environment.

:FINISHED
echo.
echo ====================================================================
echo [STATUS] Session ended. Press any key to close this window...
echo ====================================================================
pause >nul
