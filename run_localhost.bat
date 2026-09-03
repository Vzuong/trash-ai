@echo off
chcp 65001 >nul
title [Trash AI] He Thong AI Phan Loai Rac - Localhost
echo ===================================================
echo   DANG KHOI DONG TRASH AI LOCALHOST...
echo ===================================================

echo [1/2] Dang bat Python AI Service (Port 5001)...
start "Trash AI - Python Service" /min python server\yolo_service.py

timeout /t 2 /nobreak >nul

echo [2/2] Dang bat Web Server (Port 5000)...
start "Trash AI - Web Server" /min node server\server.js

timeout /t 2 /nobreak >nul

echo Mo trinh duyet...
start http://localhost:5000

echo.
echo ===================================================
echo   HE THONG DA CHAY TAI: http://localhost:5000
echo   Nhan phim bat ky de tat he thong.
echo ===================================================
pause >nul

taskkill /F /FI "WINDOWTITLE eq Trash AI - Python Service*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Trash AI - Web Server*" >nul 2>&1
