param(
    [int]$TotalReqs = 2000,
    [int]$Concurrency = 60,
    [int]$RPS = 600
)

Write-Host "Executing 1,000 RPS Load Simulation..." -ForegroundColor Cyan
go run ./cmd/benchmarker -n $TotalReqs -c $Concurrency -rps $RPS
