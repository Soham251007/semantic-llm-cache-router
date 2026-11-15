#!/usr/bin/env bash
set -e

echo "======================================================================"
echo " Starting Semantic LLM Cache & Complexity Router Services...         "
echo "======================================================================"

# 1. Start ML Microservice
echo "[1/2] Starting Python ML Microservice (port 8001)..."
python services/ml_service/app.py &
ML_PID=$!
sleep 3

# 2. Start Go Gateway
echo "[2/2] Starting Go Reverse Proxy Gateway (port 8080)..."
go run ./cmd/gateway &
GATEWAY_PID=$!

trap "kill $ML_PID $GATEWAY_PID 2>/dev/null || true" EXIT INT TERM

echo ""
echo "✓ Services are live!"
echo "  - Gateway:    http://localhost:8080/v1/chat/completions"
echo "  - ML Engine:  http://localhost:8001/health"
echo "  - Metrics:    http://localhost:8080/metrics"
echo "  - Cache Info: http://localhost:8080/api/cache/stats"
echo ""
echo "Press Ctrl+C to terminate services."

wait $GATEWAY_PID
