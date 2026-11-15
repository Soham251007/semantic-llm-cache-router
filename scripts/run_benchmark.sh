#!/usr/bin/env bash
set -e

REQS=${1:-2000}
CONC=${2:-60}
RPS=${3:-600}

echo "Executing 1,000 RPS Load Simulation..."
go run ./cmd/benchmarker -n "$REQS" -c "$CONC" -rps "$RPS"
