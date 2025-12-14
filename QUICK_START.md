# Ngrok Quick Start Guide

## Current Status

✅ **Ngrok is Running!**

Your public URL: `https://untrumpeted-hilma-unabstractedly.ngrok-free.dev`

## Usage

### 1. Start Backend Service
```powershell
docker-compose up -d backend
```

### 2. Start Ngrok Tunnel

**Method A: Using Script (Recommended)**
```powershell
cd D:\LUMA
.\start-ngrok.ps1
```

**Method B: Direct Run**
```powershell
cd D:\LUMA
.\ngrok.exe http 8000
```

### 3. Access Your Application

Access via ngrok URL:
- Main Page: `https://untrumpeted-hilma-unabstractedly.ngrok-free.dev/webpages/find.html`
- API Documentation: `https://untrumpeted-hilma-unabstractedly.ngrok-free.dev/docs`
- Health Check: `https://untrumpeted-hilma-unabstractedly.ngrok-free.dev/health`

## Important Tips

1. **PowerShell Script Execution**:
   - ✅ Correct: `.\start-ngrok.ps1`
   - ❌ Wrong: `start-ngrok.ps1`

2. **Ngrok URL Changes**:
   - URL changes every time you restart ngrok
   - Free tier generates a new URL on each start

3. **View Current Tunnel Information**:
   - Visit: http://127.0.0.1:4040 (ngrok Web Interface)
   - Or run: `Invoke-WebRequest http://127.0.0.1:4040/api/tunnels | ConvertFrom-Json`

4. **Stop Ngrok**:
   - Press `Ctrl+C` in the terminal running ngrok

## Common Commands

```powershell
# Check ngrok status
Invoke-WebRequest http://127.0.0.1:4040/api/tunnels | ConvertFrom-Json

# Check backend service
Invoke-WebRequest http://localhost:8000/health

# View ngrok version
.\ngrok.exe version

# Configure authtoken (if needed)
.\ngrok.exe config add-authtoken YOUR_TOKEN
```
