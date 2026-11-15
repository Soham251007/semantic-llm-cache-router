package main

import (
	"fmt"
	"log"

	"github.com/semantic-llm-cache-router/internal/cache"
	"github.com/semantic-llm-cache-router/internal/config"
	"github.com/semantic-llm-cache-router/internal/metrics"
	"github.com/semantic-llm-cache-router/internal/proxy"
	"github.com/semantic-llm-cache-router/internal/router"
	"github.com/semantic-llm-cache-router/internal/upstream"
)

func main() {
	cfg := config.LoadFromEnv()

	// Initialize metrics
	met := metrics.NewMetrics("llm_control_plane")

	// Initialize Cache Layer
	var c cache.Cache
	if cfg.CacheBackend == "redis" {
		rc, err := cache.NewRedisCache(cfg.RedisURL, cfg.CacheTTL)
		if err != nil {
			log.Printf("[Warning] Failed to initialize Redis (%v). Falling back to In-Memory vector store.", err)
			c = cache.NewMemoryCache(cfg.MaxMemoryEntries, cfg.CacheTTL)
		} else {
			log.Println("[Cache] Connected to Redis Stack Vector Search successfully.")
			c = rc
		}
	} else {
		log.Println("[Cache] Initialized High-Throughput In-Memory Vector Store.")
		c = cache.NewMemoryCache(cfg.MaxMemoryEntries, cfg.CacheTTL)
	}

	// Initialize Router (ML Service Client)
	r := router.NewRouter(cfg)

	// Initialize Upstream LLM Client
	u := upstream.NewClient(cfg)

	// Initialize HTTP Server
	server := proxy.NewServer(cfg, c, r, u, met)

	if err := server.Start(); err != nil {
		log.Fatalf("Server startup failed: %v", err)
	}
	fmt.Println("Server stopped cleanly.")
}
