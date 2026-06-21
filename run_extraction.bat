@echo off
set /p ROOT_FOLDER=Paste root folder path: 
.\.venv\Scripts\python.exe main.py "%ROOT_FOLDER%"
pause
