# Add ngrok to PATH environment variable
# Usage: .\add-ngrok-to-path.ps1

$ngrokDir = "$env:USERPROFILE\ngrok"

if (-not (Test-Path $ngrokDir)) {
    Write-Host "❌ Ngrok directory does not exist: $ngrokDir" -ForegroundColor Red
    Write-Host "Please run .\install-ngrok.ps1 first or install ngrok manually" -ForegroundColor Yellow
    exit 1
}

$ngrokExe = Join-Path $ngrokDir "ngrok.exe"
if (-not (Test-Path $ngrokExe)) {
    Write-Host "❌ ngrok.exe does not exist: $ngrokExe" -ForegroundColor Red
    exit 1
}

Write-Host "Adding $ngrokDir to PATH..." -ForegroundColor Cyan

# Get current user PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($currentPath -like "*$ngrokDir*") {
    Write-Host "✅ Already in PATH" -ForegroundColor Green
} else {
    try {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$ngrokDir", "User")
        Write-Host "✅ Added to user PATH" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to add: $_" -ForegroundColor Red
        Write-Host "Please manually add $ngrokDir to system PATH environment variable" -ForegroundColor Yellow
        exit 1
    }
}

# Refresh current session PATH
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')

Write-Host ""
Write-Host "✅ Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Testing if ngrok is available:" -ForegroundColor Cyan
try {
    $version = ngrok version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Ngrok is available: $version" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Please restart PowerShell and try again" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Please restart PowerShell and try again" -ForegroundColor Yellow
}
