# Ngrok installation script
# Usage: .\install-ngrok.ps1

Write-Host "📦 Installing Ngrok..." -ForegroundColor Green
Write-Host ""

# Check if already installed
try {
    $ngrokVersion = ngrok version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Ngrok is already installed: $ngrokVersion" -ForegroundColor Green
        Write-Host "If it still doesn't work, please restart PowerShell or check PATH environment variable" -ForegroundColor Yellow
        exit 0
    }
} catch {
    Write-Host "Ngrok is not installed, starting installation..." -ForegroundColor Yellow
}

# Create temporary directory
$tempDir = "$env:TEMP\ngrok-install"
$ngrokDir = "$env:USERPROFILE\ngrok"

Write-Host "Step 1: Downloading ngrok..." -ForegroundColor Cyan

# Create directories
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
}
if (-not (Test-Path $ngrokDir)) {
    New-Item -ItemType Directory -Path $ngrokDir -Force | Out-Null
}

# Download ngrok
$downloadUrl = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
$zipPath = "$tempDir\ngrok.zip"

try {
    Write-Host "Downloading from $downloadUrl..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing
    Write-Host "✅ Download completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Download failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download manually:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://ngrok.com/download/windows" -ForegroundColor Yellow
    Write-Host "2. Download and extract to: $ngrokDir" -ForegroundColor Yellow
    Write-Host "3. Run: .\add-ngrok-to-path.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Step 2: Extracting ngrok..." -ForegroundColor Cyan

try {
    Expand-Archive -Path $zipPath -DestinationPath $ngrokDir -Force
    Write-Host "✅ Extraction completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Extraction failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 3: Adding to PATH..." -ForegroundColor Cyan

# Check if already in PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$ngrokDir*") {
    try {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$ngrokDir", "User")
        Write-Host "✅ Added to user PATH" -ForegroundColor Green
        Write-Host ""
        Write-Host "⚠️  Important: Please restart PowerShell or run the following command to refresh PATH:" -ForegroundColor Yellow
        Write-Host "   `$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')" -ForegroundColor Cyan
    } catch {
        Write-Host "❌ Failed to add to PATH: $_" -ForegroundColor Red
        Write-Host "Please manually add $ngrokDir to system PATH environment variable" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Already in PATH" -ForegroundColor Green
}

Write-Host ""
Write-Host "Step 4: Verifying installation..." -ForegroundColor Cyan

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')

try {
    $ngrokPath = Join-Path $ngrokDir "ngrok.exe"
    if (Test-Path $ngrokPath) {
        Write-Host "✅ Ngrok file exists: $ngrokPath" -ForegroundColor Green
        
        # Try to run
        $version = & $ngrokPath version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Ngrok is working: $version" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Ngrok file exists but cannot run" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Ngrok file does not exist" -ForegroundColor Red
    }
} catch {
    Write-Host "⚠️  Cannot verify, but file is installed at: $ngrokDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Installation completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Restart PowerShell or run the PATH refresh command above" -ForegroundColor White
Write-Host "2. Get authtoken: https://dashboard.ngrok.com/get-started/your-authtoken" -ForegroundColor White
Write-Host "3. Configure: ngrok config add-authtoken YOUR_TOKEN" -ForegroundColor White
Write-Host "4. Start: ngrok http 8000" -ForegroundColor White
