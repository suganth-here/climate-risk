$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonCmd = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

Write-Host "Starting Django backend on http://127.0.0.1:8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$repoRoot'; $pythonCmd backend\manage.py runserver 127.0.0.1:8000"

Write-Host "Starting React frontend on http://127.0.0.1:5173 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$repoRoot\frontend'; npm run dev"

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5173"
