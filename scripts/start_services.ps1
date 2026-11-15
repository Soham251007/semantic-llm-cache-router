# Start Semantic LLM Cache & Complexity Router control plane locally
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Starting Semantic LLM Cache & Complexity Router Services...         " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Start Python ML Microservice
Write-Host "[1/2] Starting Python ML Microservice (port 8001)..." -ForegroundColor Yellow
$mlProc = Start-Process -FilePath "py" -ArgumentList "-3.13", "services/ml_service/app.py" -PassThru -NoNewWindow
Start-Sleep -Seconds 3

# 2. Start Go Reverse Proxy
Write-Host "[2/2] Starting Go Reverse Proxy Gateway (port 8080)..." -ForegroundColor Yellow
$gatewayProc = Start-Process -FilePath "go" -ArgumentList "run", "./cmd/gateway" -PassThru -NoNewWindow

Write-Host "`n✓ Services are live!" -ForegroundColor Green
Write-Host "  - Gateway:    http://localhost:8080/v1/chat/completions" -ForegroundColor Green
Write-Host "  - ML Engine:  http://localhost:8001/health" -ForegroundColor Green
Write-Host "  - Metrics:    http://localhost:8080/metrics" -ForegroundColor Green
Write-Host "  - Cache Info: http://localhost:8080/api/cache/stats" -ForegroundColor Green
Write-Host "`nPress Ctrl+C to terminate services.`n" -ForegroundColor DarkGray

try {
    Wait-Process -Id $gatewayProc.Id
} finally {
    Stop-Process -Id $mlProc.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $gatewayProc.Id -ErrorAction SilentlyContinue
}
