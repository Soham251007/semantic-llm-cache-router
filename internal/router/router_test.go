package router

import (
	"context"
	"testing"

	"github.com/semantic-llm-cache-router/internal/config"
)

func TestRouter_FallbackEvaluation(t *testing.T) {
	cfg := &config.Config{
		MLServiceURL:    "http://127.0.0.1:59999", // non-existent port to force fallback
		ComplexityAlpha: 0.45,
	}
	r := NewRouter(cfg)

	// Simple prompt fallback
	simpleRes, err := r.AnalyzePrompt(context.Background(), "Summarize this memo")
	if err != nil {
		t.Fatalf("expected fallback without error, got %v", err)
	}
	if simpleRes.RecommendedTier != "tier-1-fast" {
		t.Errorf("expected tier-1-fast, got %s", simpleRes.RecommendedTier)
	}

	// Complex prompt fallback
	complexRes, err := r.AnalyzePrompt(context.Background(), "Prove raft consensus safety with lock-free atomic buffers")
	if err != nil {
		t.Fatalf("expected fallback without error, got %v", err)
	}
	if complexRes.RecommendedTier != "tier-2-frontier" {
		t.Errorf("expected tier-2-frontier, got %s", complexRes.RecommendedTier)
	}
}
