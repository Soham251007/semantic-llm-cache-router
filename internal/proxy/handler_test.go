package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/semantic-llm-cache-router/internal/cache"
	"github.com/semantic-llm-cache-router/internal/config"
	"github.com/semantic-llm-cache-router/internal/metrics"
	"github.com/semantic-llm-cache-router/internal/router"
	"github.com/semantic-llm-cache-router/internal/upstream"
)

func TestHandler_EndToEndFlow(t *testing.T) {
	cfg := &config.Config{
		Port:                8080,
		SimilarityThreshold: 0.94,
		ComplexityAlpha:     0.45,
		Tier1Model:          "gpt-4o-mini",
		Tier2Model:          "gpt-4o",
		UpstreamProvider:    "simulator",
		SimulateLatency:     false, // instant execution for unit tests
		CacheTTL:            1 * time.Hour,
		MaxMemoryEntries:    1000,
	}

	met := metrics.NewMetrics("test_llm_plane")
	memCache := cache.NewMemoryCache(1000, 1*time.Hour)
	r := router.NewRouter(cfg)
	u := upstream.NewClient(cfg)
	handler := NewHandler(cfg, memCache, r, u, met)

	ts := httptest.NewServer(handler)
	defer ts.Close()

	client := ts.Client()

	// 1. First Request: Cache Miss
	reqBody := upstream.ChatCompletionRequest{
		Model: "router-auto",
		Messages: []upstream.ChatMessage{
			{Role: "user", Content: "Summarize this paragraph in 3 bullet points."},
		},
	}
	data, _ := json.Marshal(reqBody)

	resp, err := client.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	cacheStatus := resp.Header.Get("X-Cache-Status")
	tier := resp.Header.Get("X-Router-Tier")

	if cacheStatus != "MISS" {
		t.Errorf("expected first request to be MISS, got %s", cacheStatus)
	}
	if tier != "tier-1-fast" {
		t.Errorf("expected summarization prompt to route to tier-1-fast, got %s", tier)
	}

	var completion upstream.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
		t.Fatalf("failed to parse completion: %v", err)
	}
	if len(completion.Choices) == 0 {
		t.Fatalf("expected non-empty completion choices")
	}

	// Allow asynchronous cache ingestion goroutine to write
	time.Sleep(50 * time.Millisecond)

	// 2. Second Identical Request: L1 Exact Match Hit (<2ms)
	resp2, err := client.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	defer resp2.Body.Close()

	cacheStatus2 := resp2.Header.Get("X-Cache-Status")
	if cacheStatus2 != "HIT-EXACT" {
		t.Errorf("expected second request to be HIT-EXACT, got %s", cacheStatus2)
	}

	// 3. Complex Request: Should route to tier-2-frontier
	complexReq := upstream.ChatCompletionRequest{
		Model: "router-auto",
		Messages: []upstream.ChatMessage{
			{Role: "user", Content: "Design a distributed raft consensus engine in Go with lock-free atomic queues."},
		},
	}
	complexData, _ := json.Marshal(complexReq)

	resp3, err := client.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(complexData))
	if err != nil {
		t.Fatalf("complex request failed: %v", err)
	}
	defer resp3.Body.Close()

	tier3 := resp3.Header.Get("X-Router-Tier")
	if tier3 != "tier-2-frontier" {
		t.Errorf("expected distributed raft prompt to route to tier-2-frontier, got %s", tier3)
	}
}
