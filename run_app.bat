@echo off
echo Starting MindTrack Application...
echo ================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo Error: Virtual environment not found!
    echo Please run setup.py first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Error: Streamlit not installed!
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Launch Streamlit app
echo Starting Streamlit application...
streamlit run app/app.py

pause