package proxy

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/semantic-llm-cache-router/internal/cache"
	"github.com/semantic-llm-cache-router/internal/config"
	"github.com/semantic-llm-cache-router/internal/metrics"
	"github.com/semantic-llm-cache-router/internal/router"
	"github.com/semantic-llm-cache-router/internal/upstream"
)

// Server coordinates HTTP lifecycle and graceful shutdown
type Server struct {
	cfg        *config.Config
	httpServer *http.Server
	cache      cache.Cache
}

func NewServer(
	cfg *config.Config,
	c cache.Cache,
	r *router.Router,
	u *upstream.Client,
	m *metrics.Metrics,
) *Server {
	handler := NewHandler(cfg, c, r, u, m)

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.Handle("/", handler)

	httpServer := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	return &Server{
		cfg:        cfg,
		httpServer: httpServer,
		cache:      c,
	}
}

// Start runs the HTTP server and listens for interrupt signals
func (s *Server) Start() error {
	printBanner(s.cfg)

	errChan := make(chan error, 1)
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errChan <- err
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errChan:
		return fmt.Errorf("server error: %w", err)
	case sig := <-quit:
		fmt.Printf("\n[Gateway] Received signal %v. Gracefully shutting down...\n", sig)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_ = s.cache.Close()
	return s.httpServer.Shutdown(ctx)
}

func printBanner(cfg *config.Config) {
	fmt.Println("======================================================================")
	fmt.Println("      SEMANTIC LLM CACHE & TIERED COMPLEXITY ROUTER CONTROL PLANE     ")
	fmt.Println("======================================================================")
	fmt.Printf(" Listening on port:       :%d\n", cfg.Port)
	fmt.Printf(" OpenAI Endpoint:         POST http://localhost:%d/v1/chat/completions\n", cfg.Port)
	fmt.Printf(" Prometheus Metrics:      GET  http://localhost:%d/metrics\n", cfg.Port)
	fmt.Printf(" Cache Backend:           %s\n", cfg.CacheBackend)
	fmt.Printf(" Cosine Threshold:        >= %.2f\n", cfg.SimilarityThreshold)
	fmt.Printf(" Complexity Threshold α:  %.2f\n", cfg.ComplexityAlpha)
	fmt.Printf(" Tier 1 Model (Fast):     %s\n", cfg.Tier1Model)
	fmt.Printf(" Tier 2 Model (Frontier): %s\n", cfg.Tier2Model)
	fmt.Printf(" Upstream Provider:       %s\n", cfg.UpstreamProvider)
	fmt.Println("======================================================================")
}
