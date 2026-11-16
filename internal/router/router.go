package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/semantic-llm-cache-router/internal/config"
)

// AnalyzeResult contains the output of embedding and complexity evaluation
type AnalyzeResult struct {
	Embedding            []float32 `json:"embedding"`
	Dimension            int       `json:"dimension"`
	ComplexityScore      float64   `json:"complexity_score"`
	RecommendedTier      string    `json:"recommended_tier"`
	IsComplex            bool      `json:"is_complex"`
	Signals              []string  `json:"signals"`
	SemanticSignature   string    `json:"semantic_signature"`
	Polarity            string    `json:"polarity"`
	EmbedLatencyMs       float64   `json:"embed_latency_ms"`
	ComplexityLatencyMs  float64   `json:"complexity_latency_ms"`
	TotalLatencyMs       float64   `json:"total_latency_ms"`
}

type analyzeRequest struct {
	Text      string  `json:"text"`
	Alpha     float64 `json:"alpha"`
	MaxLength int     `json:"max_length"`
}

// Router coordinates ML inference and tiered model decisions
type Router struct {
	cfg        *config.Config
	httpClient *http.Client
}

func NewRouter(cfg *config.Config) *Router {
	transport := &http.Transport{
		MaxIdleConns:        1000,
		MaxIdleConnsPerHost: 500,
		IdleConnTimeout:     90 * time.Second,
	}

	return &Router{
		cfg: cfg,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   5 * time.Second,
		},
	}
}

// AnalyzePrompt issues a single HTTP request to the Python microservice /analyze endpoint
func (r *Router) AnalyzePrompt(ctx context.Context, prompt string) (*AnalyzeResult, error) {
	reqBody := analyzeRequest{
		Text:      prompt,
		Alpha:     r.cfg.ComplexityAlpha,
		MaxLength: 128,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/analyze", strings.TrimRight(r.cfg.MLServiceURL, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		// Fallback to local heuristic evaluation if ML microservice is unreachable
		return r.fallbackEvaluation(prompt), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ml service error (%d): %s", resp.StatusCode, string(errBytes))
	}

	var result AnalyzeResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// fallbackEvaluation provides resilience if the ML microservice is temporarily restarting
func (r *Router) fallbackEvaluation(prompt string) *AnalyzeResult {
	lower := strings.ToLower(prompt)
	complexKeywords := []string{"prove", "concurrency", "lock-free", "distributed", "raft", "algorithm", "exploit", "kernel"}
	isComplex := false
	for _, kw := range complexKeywords {
		if strings.Contains(lower, kw) {
			isComplex = true
			break
		}
	}

	tier := "tier-1-fast"
	score := 0.20
	if isComplex {
		tier = "tier-2-frontier"
		score = 0.85
	}

	return &AnalyzeResult{
		Embedding:       nil,
		Dimension:       0,
		ComplexityScore: score,
		RecommendedTier: tier,
		IsComplex:       isComplex,
		Signals:         []string{"fallback_heuristic"},
		TotalLatencyMs:  0.1,
	}
}
