@echo off
echo ============================================
echo   Fluid Analyst — Data Science Co-Pilot
echo ============================================
echo.
echo Starting Streamlit app (auto-detecting port)...
streamlit run "%~dp0app.py"
pause
