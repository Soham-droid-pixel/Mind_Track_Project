# MindTrack Application Launcher
# PowerShell script to run the MindTrack Streamlit application

Write-Host "Starting MindTrack Application..." -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup.py first." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Verify we're in the virtual environment
$pythonPath = & python -c "import sys; print(sys.executable)"
Write-Host "Using Python: $pythonPath" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "app\app.py")) {
    Write-Host "Error: app.py not found!" -ForegroundColor Red
    Write-Host "Please make sure you're in the MindTrack directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Launch Streamlit app
Write-Host "Starting Streamlit application..." -ForegroundColor Cyan
Write-Host "Your browser should open automatically." -ForegroundColor Yellow
Write-Host "If not, go to: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the application." -ForegroundColor Magenta

try {
    python -m streamlit run app/app.py
}
catch {
    Write-Host "Error starting application: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}