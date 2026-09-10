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
powershell -NoProfile -Command "try { if ((Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 1).StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting GPU Inference Bridge on Port 8000...
    start "AZN-Vision GPU Bridge (Port 8000)" /D "%~dp0" cmd /k ""%PYTHON_CMD%" "%~dp0scripts\mobile_bridge.py" --host 0.0.0.0 --port 8000"
) else (
    echo [OK] GPU Inference Bridge is active on port 8000.
)

echo.
echo [3/3] Resolving Network Interfaces and Launching Expo Go...
echo ====================================================================

rem Dynamically detect local Wi-Fi / LAN IP (avoiding VPN and virtual adapters like team2)
for /f "delims=" %%a in ('powershell -NoProfile -Command "([System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) | Where-Object { $_.AddressFamily -eq 'InterNetwork' -and $_.IPAddressToString -like '192.168.*' } | Select-Object -First 1).IPAddressToString"') do set "WIFI_IP=%%a"
if not defined WIFI_IP set "WIFI_IP=127.0.0.1"

set "REACT_NATIVE_PACKAGER_HOSTNAME=%WIFI_IP%"

echo [OK] Active Wi-Fi Host IP : %WIFI_IP%
echo [OK] GPU Inference Bridge : http://%WIFI_IP%:8000
echo [OK] Expo Go Target URL   : exp://%WIFI_IP%:8081
echo.
echo Please scan the QR code below using the Expo Go app on your phone.
echo Ensure your phone is connected to the same Wi-Fi network (%WIFI_IP%).
echo ====================================================================

rem Terminate any stale process occupying Metro bundler port 8081
powershell -NoProfile -Command "$conns = Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue; foreach ($c in $conns) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1

cd /d "%~dp0mobile"
call npx expo start --clear --offline
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
