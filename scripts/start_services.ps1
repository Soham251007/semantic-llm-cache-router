# Start Semantic LLM Cache & Complexity Router control plane locally
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Starting Semantic LLM Cache & Complexity Router Services...         " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Determine python command
$pythonCmd = "python"
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
    $pythonArgs = @("-3.13", "services/ml_service/app.py")
} else {
    $pythonArgs = @("services/ml_service/app.py")
}

# 1. Start Python ML Microservice
Write-Host "[1/2] Starting Python ML Microservice (port 8001)..." -ForegroundColor Yellow
$mlProc = Start-Process -FilePath $pythonCmd -ArgumentList $pythonArgs -PassThru -NoNewWindow
Start-Sleep -Seconds 3

# 2. Start Go Reverse Proxy
Write-Host "[2/2] Starting Go Reverse Proxy Gateway (port 8080)..." -ForegroundColor Yellow
$gatewayProc = Start-Process -FilePath "go" -ArgumentList "run", "./cmd/gateway" -PassThru -NoNewWindow

Write-Host ""
Write-Host "[+] Services are live!" -ForegroundColor Green
Write-Host "  - Gateway:    http://localhost:8080/v1/chat/completions" -ForegroundColor Green
Write-Host "  - ML Engine:  http://localhost:8001/health" -ForegroundColor Green
Write-Host "  - Metrics:    http://localhost:8080/metrics" -ForegroundColor Green
Write-Host "  - Cache Info: http://localhost:8080/api/cache/stats" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to terminate services." -ForegroundColor DarkGray
Write-Host ""

try {
    Wait-Process -Id $gatewayProc.Id
} finally {
    Write-Host "Stopping background services..." -ForegroundColor Yellow
    Stop-Process -Id $mlProc.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $gatewayProc.Id -ErrorAction SilentlyContinue
}
