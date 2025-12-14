# Ngrok startup script
# Usage: .\start-ngrok.ps1

Write-Host "🚀 Starting Ngrok tunnel..." -ForegroundColor Green

# Check if ngrok is installed
$ngrokPath = $null

# First check current directory
if (Test-Path ".\ngrok.exe") {
    $ngrokPath = ".\ngrok.exe"
    Write-Host "✅ Found ngrok.exe in current directory" -ForegroundColor Green
} elseif (Test-Path "$PSScriptRoot\ngrok.exe") {
    $ngrokPath = "$PSScriptRoot\ngrok.exe"
    Write-Host "✅ Found ngrok.exe in script directory" -ForegroundColor Green
} else {
    # Check PATH
    try {
        $ngrokVersion = ngrok version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $ngrokPath = "ngrok"
            Write-Host "✅ Ngrok is in PATH" -ForegroundColor Green
        }
    } catch {
        # Continue searching
    }
}

if (-not $ngrokPath) {
    Write-Host "❌ Ngrok not found" -ForegroundColor Red
    Write-Host "Please install ngrok first:" -ForegroundColor Yellow
    Write-Host "1. Run: .\install-ngrok.ps1" -ForegroundColor Yellow
    Write-Host "2. Or visit https://ngrok.com/download/windows to download manually" -ForegroundColor Yellow
    Write-Host "3. Place ngrok.exe in current directory or add to PATH" -ForegroundColor Yellow
    exit 1
}

# Check if backend service is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -UseBasicParsing
    Write-Host "✅ Backend service is running (port 8000)" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backend service is not running. Please start it first:" -ForegroundColor Yellow
    Write-Host "   docker-compose up -d backend" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue starting ngrok anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Start ngrok
Write-Host ""
Write-Host "Starting ngrok tunnel to http://localhost:8000..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop ngrok" -ForegroundColor Yellow
Write-Host ""

# Start ngrok
if ($ngrokPath -eq "ngrok") {
    ngrok http 8000
} else {
    & $ngrokPath http 8000
}
