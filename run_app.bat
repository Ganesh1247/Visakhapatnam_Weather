@echo off
cd /d "%~dp0"
echo.
echo ========================================
echo   EcoGlance - Weather and Air Quality
echo ========================================
echo.
echo Starting server... Open http://127.0.0.1:5000 in your browser
echo.
python app.py
pause
