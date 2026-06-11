$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
    throw "Python virtual environment not found at .venv. Create it first."
}

Write-Host "Starting TrollFish local server on http://127.0.0.1:8000" -ForegroundColor Cyan
.venv\Scripts\python.exe -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

