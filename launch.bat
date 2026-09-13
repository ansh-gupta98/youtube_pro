@echo off
title YouTube Video Chatbot Pro v2.0
echo.
echo  ============================================================
echo   🎬  YouTube Video Chatbot Pro  v2.0
echo  ============================================================
echo.

REM ── Use the LangGraph virtualenv which has all dependencies ──────────────
set VENV=C:\Users\hp\Desktop\LangGraph\myenv\Scripts\python.exe
set APP=%~dp0app.py

echo  [1/2] Starting Streamlit on http://localhost:8501 ...
echo.

"%VENV%" -m streamlit run "%APP%" --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
