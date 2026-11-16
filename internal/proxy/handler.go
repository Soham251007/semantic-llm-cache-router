package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/semantic-llm-cache-router/internal/cache"
	"github.com/semantic-llm-cache-router/internal/config"
	"github.com/semantic-llm-cache-router/internal/metrics"
	"github.com/semantic-llm-cache-router/internal/router"
	"github.com/semantic-llm-cache-router/internal/upstream"
)

type Handler struct {
	cfg        *config.Config
	cache      cache.Cache
	router     *router.Router
	upstream   *upstream.Client
	streamer   *upstream.Streamer
	metrics    *metrics.Metrics
}

func NewHandler(
	cfg *config.Config,
	c cache.Cache,
	r *router.Router,
	u *upstream.Client,
	m *metrics.Metrics,
) *Handler {
	return &Handler{
		cfg:      cfg,
		cache:    c,
		router:   r,
		upstream: u,
		streamer: upstream.NewStreamer(),
		metrics:  m,
	}
}

// ServeHTTP handles all incoming API traffic
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Enable CORS for web apps & testing dashboards
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	path := r.URL.Path

	switch {
	case path == "/healthz" || path == "/readyz":
		h.handleHealth(w, r)
	case path == "/api/cache/stats":
		h.handleCacheStats(w, r)
	case path == "/v1/chat/completions":
		if r.Method != http.MethodPost {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}
		h.handleChatCompletions(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":        "healthy",
		"service":       "semantic-llm-cache-router",
		"cache_backend": h.cfg.CacheBackend,
		"cached_items":  h.cache.Len(),
		"threshold":     h.cfg.SimilarityThreshold,
		"alpha":         h.cfg.ComplexityAlpha,
		"time":          time.Now().UTC().Format(time.RFC3339),
	})
}

func (h *Handler) handleCacheStats(w http.ResponseWriter, r *http.Request) {
	stats := h.cache.Stats()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (h *Handler) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	h.metrics.ActiveConnections.Inc()
	defer h.metrics.ActiveConnections.Dec()

	ctx := r.Context()

	var req upstream.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.metrics.RequestsTotal.WithLabelValues("400", "ERROR", "none").Inc()
		http.Error(w, fmt.Sprintf("invalid JSON payload: %v", err), http.StatusBadRequest)
		return
	}

	// 1. Extract canonical prompt for hashing & user prompt for semantic evaluation
	_, canonicalPrompt, hashKey := cache.ExtractCanonicalPrompt(&req)
	userPrompt := cache.ExtractLastUserPrompt(&req)

	if userPrompt == "" {
		h.metrics.RequestsTotal.WithLabelValues("400", "ERROR", "none").Inc()
		http.Error(w, "no messages provided in chat completions request", http.StatusBadRequest)
		return
	}

	// 2. L1 Exact Match Check (O(1) hash lookup in <1ms)
	exactMatch, err := h.cache.GetExact(ctx, hashKey)
	if err == nil && exactMatch.Hit {
		h.recordHitMetrics(exactMatch, "HIT-EXACT", start)
		h.injectTelemetryHeaders(w, exactMatch.Type, 1.0, 0.0, exactMatch.Item.Tier, start, exactMatch.Item.TokensSaved, exactMatch.Item.CostSavedUSD)

		if req.Stream {
			flusher, ok := w.(http.Flusher)
			if ok {
				_ = h.streamer.ReplayCachedStream(w, flusher, exactMatch.Item.Response, 0)
				return
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(exactMatch.Item.Response)
		return
	}

	// 3. Request embedding & complexity evaluation from ML microservice in a single call
	analysis, err := h.router.AnalyzePrompt(ctx, userPrompt)
	if err != nil {
		// Log error and fallback gracefully
		fmt.Printf("[Router] Warning: ML service analysis failed: %v\n", err)
	}

	// 4. L2 Semantic Vector Cache Check (Cosine Similarity >= 0.94 in <15ms)
	var semanticMatch *cache.MatchResult
	if analysis != nil && len(analysis.Embedding) > 0 {
		semanticMatch, _ = h.cache.GetSemantic(ctx, analysis.Embedding, h.cfg.SimilarityThreshold, analysis.SemanticSignature)
		if semanticMatch != nil && semanticMatch.Hit {
			h.recordHitMetrics(semanticMatch, "HIT-SEMANTIC", start)
			h.injectTelemetryHeaders(w, semanticMatch.Type, semanticMatch.Similarity, analysis.ComplexityScore, semanticMatch.Item.Tier, start, semanticMatch.Item.TokensSaved, semanticMatch.Item.CostSavedUSD)

			if req.Stream {
				flusher, ok := w.(http.Flusher)
				if ok {
					_ = h.streamer.ReplayCachedStream(w, flusher, semanticMatch.Item.Response, 0)
					return
				}
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(semanticMatch.Item.Response)
			return
		}
	}

	// 5. Cache MISS: Tiered Complexity Routing
	h.metrics.CacheMissesTotal.Inc()

	tier := "tier-1-fast"
	var complexityScore float64 = 0.20
	if analysis != nil {
		tier = analysis.RecommendedTier
		complexityScore = analysis.ComplexityScore
		h.metrics.ComplexityScores.Observe(complexityScore)
	}

	// Dispatch to upstream LLM
	upstreamResp, err := h.upstream.Dispatch(ctx, &req, tier, userPrompt)
	if err != nil {
		h.metrics.RequestsTotal.WithLabelValues("502", "MISS", tier).Inc()
		http.Error(w, fmt.Sprintf("upstream LLM failure: %v", err), http.StatusBadGateway)
		return
	}

	costUSD := upstream.CalculateCost(tier, upstreamResp.Usage)

	// 6. Asynchronous Cache Ingestion
	var emb []float32
	var sig string
	if analysis != nil {
		emb = analysis.Embedding
		sig = analysis.SemanticSignature
	}
	go func(itemResp *upstream.ChatCompletionResponse, emb []float32, prompt, canonical, hKey, t string, cUSD float64, signature string) {
		cacheItem := &cache.CacheItem{
			ID:           uuid.New().String(),
			Prompt:       prompt,
			Canonical:    canonical,
			HashKey:      hKey,
			Embedding:    emb,
			Signature:    signature,
			Response:     itemResp,
			Model:        itemResp.Model,
			Tier:         t,
			TokensSaved:  itemResp.Usage.TotalTokens,
			CostSavedUSD: cUSD,
			CreatedAt:    time.Now(),
			ExpiresAt:    time.Now().Add(h.cfg.CacheTTL),
		}
		_ = h.cache.Set(context.Background(), cacheItem)
	}(upstreamResp, emb, userPrompt, canonicalPrompt, hashKey, tier, costUSD, sig)

	// 7. Inject Headers & Deliver Response
	var simScore float64 = 0.0
	if semanticMatch != nil {
		simScore = semanticMatch.Similarity
	}

	h.metrics.RequestsTotal.WithLabelValues("200", "MISS", tier).Inc()
	h.metrics.RequestDuration.WithLabelValues("MISS", tier).Observe(time.Since(start).Seconds())
	h.injectTelemetryHeaders(w, cache.CacheTypeNone, simScore, complexityScore, tier, start, 0, 0)

	if req.Stream {
		flusher, ok := w.(http.Flusher)
		if ok {
			_ = h.streamer.ReplayCachedStream(w, flusher, upstreamResp, 0)
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(upstreamResp)
}

func (h *Handler) recordHitMetrics(match *cache.MatchResult, cacheStatus string, start time.Time) {
	h.metrics.RequestsTotal.WithLabelValues("200", cacheStatus, match.Item.Tier).Inc()
	h.metrics.RequestDuration.WithLabelValues(cacheStatus, match.Item.Tier).Observe(time.Since(start).Seconds())
	
	cacheTypeLabel := "exact"
	if match.Type == cache.CacheTypeSemantic {
		cacheTypeLabel = "semantic"
		h.metrics.SimilarityScores.Observe(match.Similarity)
	}
	h.metrics.CacheHitsTotal.WithLabelValues(cacheTypeLabel).Inc()
	h.metrics.TokensSavedTotal.Add(float64(match.Item.TokensSaved))
	h.metrics.CostSavedUSDTotal.Add(match.Item.CostSavedUSD)
}

func (h *Handler) injectTelemetryHeaders(
	w http.ResponseWriter,
	cacheType cache.CacheType,
	similarity float64,
	complexity float64,
	tier string,
	start time.Time,
	tokensSaved int,
	costSavedUSD float64,
) {
	latencyMs := float64(time.Since(start).Microseconds()) / 1000.0

	w.Header().Set("X-Cache-Status", string(cacheType))
	w.Header().Set("X-Cache-Similarity", fmt.Sprintf("%.4f", similarity))
	w.Header().Set("X-Router-Complexity", fmt.Sprintf("%.4f", complexity))
	w.Header().Set("X-Router-Tier", tier)
	w.Header().Set("X-Latency-Ms", fmt.Sprintf("%.2f", latencyMs))
	w.Header().Set("X-Tokens-Saved", strconv.Itoa(tokensSaved))
	w.Header().Set("X-Cost-Saved-USD", fmt.Sprintf("%.6f", costSavedUSD))
}
