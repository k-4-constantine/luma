# Ngrok Deployment Guide

## Installing Ngrok

### Method 1: Using Chocolatey (Requires Administrator Privileges)
```powershell
# Run PowerShell as Administrator, then execute:
choco install ngrok -y
```

### Method 2: Manual Installation (Recommended)
1. Visit https://ngrok.com/download/windows
2. Download the ngrok zip package
3. Extract to any directory (e.g., `C:\ngrok`)
4. Add the ngrok.exe directory to your system PATH environment variable

### Method 3: Using Microsoft Store
1. Open Microsoft Store
2. Search for "ngrok"
3. Click Install

## Configuring Ngrok

### 1. Get Authtoken
1. Visit https://dashboard.ngrok.com/signup to register an account (free)
2. After logging in, visit https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your authtoken

### 2. Configure Authtoken
```powershell
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

## Starting Ngrok Tunnel

### Start Backend Service Tunnel (Port 8000)
```powershell
ngrok http 8000
```

### Start Frontend Service Tunnel (Port 8501, if using Streamlit)
```powershell
ngrok http 8501
```

## Using Docker Compose Auto-Start

### Option 1: Start ngrok in Background
```powershell
# Start backend service
docker-compose up -d backend

# Start ngrok in another terminal
ngrok http 8000
```

### Option 2: Using ngrok Configuration File
Create an `ngrok.yml` configuration file:
```yaml
version: "2"
authtoken: YOUR_AUTHTOKEN_HERE
tunnels:
  backend:
    addr: 8000
    proto: http
```

Then run:
```powershell
ngrok start backend
```

## Accessing Your Application

After starting ngrok, you will see output similar to:
```
Forwarding  https://xxxx-xx-xx-xx-xx.ngrok-free.app -> http://localhost:8000
```

Use this URL to access your local service from anywhere!

## Important Notes

1. **Free Tier Limitations**:
   - URL changes each time you restart
   - Connection limit restrictions
   - Requires ngrok account

2. **Security**:
   - Do not use free tier in production
   - Consider setting access password: `ngrok http 8000 --basic-auth="username:password"`

3. **CORS Issues**:
   - If frontend and backend are on different ports, ensure backend CORS configuration allows ngrok domain

## Updating Frontend API URL

If using ngrok, you may need to update the API URL in frontend code:

In `webpages/find.html`, the code automatically detects ngrok URLs:
```javascript
// If accessed via ngrok, use relative path or dynamic detection
const API_BASE = window.location.origin;
const API_URL = API_BASE + '/api';
```

The frontend code automatically detects ngrok domains and adjusts the API URL accordingly.
