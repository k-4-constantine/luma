# Ngrok quick setup script
# Usage: .\setup-ngrok.ps1

Write-Host "🔧 Ngrok Configuration Assistant" -ForegroundColor Green
Write-Host ""

# Check for ngrok.exe
$ngrokPath = Join-Path $PSScriptRoot "ngrok.exe"
if (-not (Test-Path $ngrokPath)) {
    Write-Host "❌ ngrok.exe not found at: $ngrokPath" -ForegroundColor Red
    Write-Host "Please ensure ngrok.exe is in the current directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Found ngrok.exe: $ngrokPath" -ForegroundColor Green
Write-Host ""

# Check if authtoken is already configured
Write-Host "Checking ngrok configuration..." -ForegroundColor Cyan
$configPath = "$env:USERPROFILE\.ngrok2\ngrok.yml"
if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw
    if ($config -match "authtoken:") {
        Write-Host "✅ Ngrok authtoken is already configured" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can start ngrok directly:" -ForegroundColor Cyan
        Write-Host "  & `"$ngrokPath`" http 8000" -ForegroundColor Yellow
        Write-Host "  Or run: .\start-ngrok.ps1" -ForegroundColor Yellow
        exit 0
    }
}

# Prompt for authtoken
Write-Host "⚠️  Ngrok authtoken is not configured" -ForegroundColor Yellow
Write-Host ""
Write-Host "Please get your authtoken:" -ForegroundColor Cyan
Write-Host "1. Visit: https://dashboard.ngrok.com/signup (sign up for an account)" -ForegroundColor White
Write-Host "2. After login, visit: https://dashboard.ngrok.com/get-started/your-authtoken" -ForegroundColor White
Write-Host "3. Copy your authtoken" -ForegroundColor White
Write-Host ""

$authtoken = Read-Host "Enter your authtoken (or press Enter to skip)"

if ([string]::IsNullOrWhiteSpace($authtoken)) {
    Write-Host ""
    Write-Host "Skipping configuration. You can manually run later:" -ForegroundColor Yellow
    Write-Host "  & `"$ngrokPath`" config add-authtoken YOUR_TOKEN" -ForegroundColor Cyan
    exit 0
}

# Configure authtoken
Write-Host ""
Write-Host "Configuring authtoken..." -ForegroundColor Cyan
try {
    & $ngrokPath config add-authtoken $authtoken
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Authtoken configured successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Now you can start ngrok:" -ForegroundColor Cyan
        Write-Host "  & `"$ngrokPath`" http 8000" -ForegroundColor Yellow
        Write-Host "  Or run: .\start-ngrok.ps1" -ForegroundColor Yellow
    } else {
        Write-Host "❌ Configuration failed, please check if authtoken is correct" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Configuration failed: $_" -ForegroundColor Red
}
