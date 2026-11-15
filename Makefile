.PHONY: all build run test benchmark docker-up docker-down clean

all: build test

build:
	@echo "Building Go binaries..."
	go build -ldflags="-w -s" -o bin/gateway ./cmd/gateway
	go build -ldflags="-w -s" -o bin/benchmarker ./cmd/benchmarker

run-gateway:
	@echo "Starting Go Gateway..."
	go run ./cmd/gateway

run-ml:
	@echo "Starting Python ML Microservice..."
	python services/ml_service/app.py

test:
	@echo "Running Go test suite..."
	go test -v -race ./...
	@echo "Running Python ML test suite..."
	python services/ml_service/test_ml_service.py

benchmark:
	@echo "Running Benchmark & Load Simulator..."
	go run ./cmd/benchmarker -n 2000 -c 60 -rps 600

docker-up:
	@echo "Starting Docker Compose environment..."
	docker compose -f deployments/docker-compose.yml up --build -d

docker-down:
	@echo "Stopping Docker Compose environment..."
	docker compose -f deployments/docker-compose.yml down

clean:
	@echo "Cleaning binaries..."
	rm -rf bin/ gateway.exe benchmarker.exe
