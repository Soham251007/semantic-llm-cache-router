package cache

import (
	"context"
	"time"

	"github.com/semantic-llm-cache-router/internal/upstream"
)

// CacheType represents the level of cache hit
type CacheType string

const (
	CacheTypeNone     CacheType = "MISS"
	CacheTypeExact    CacheType = "HIT-EXACT"
	CacheTypeSemantic CacheType = "HIT-SEMANTIC"
)

// CacheItem represents a stored completion item in the cache
type CacheItem struct {
	ID           string                          `json:"id"`
	Prompt       string                          `json:"prompt"`
	Canonical    string                          `json:"canonical"`
	HashKey      string                          `json:"hash_key"`
	Embedding    []float32                       `json:"embedding"`
	Response     *upstream.ChatCompletionResponse `json:"response"`
	Model        string                          `json:"model"`
	Tier         string                          `json:"tier"`
	TokensSaved  int                             `json:"tokens_saved"`
	CostSavedUSD float64                         `json:"cost_saved_usd"`
	CreatedAt    time.Time                       `json:"created_at"`
	ExpiresAt    time.Time                       `json:"expires_at"`
}

// MatchResult represents the outcome of a cache lookup
type MatchResult struct {
	Type        CacheType
	Hit         bool
	Similarity  float64
	Item        *CacheItem
	Duration    time.Duration
}

// CacheStats provides operational telemetry for observability
type CacheStats struct {
	TotalEntries   int     `json:"total_entries"`
	ExactHits      int64   `json:"exact_hits"`
	SemanticHits   int64   `json:"semantic_hits"`
	Misses         int64   `json:"misses"`
	HitRate        float64 `json:"hit_rate"`
	TokensSaved    int64   `json:"tokens_saved"`
	CostSavedUSD   float64 `json:"cost_saved_usd"`
}

// Cache defines the contract for semantic & exact caching implementations
type Cache interface {
	// GetExact performs an O(1) hash lookup on the canonicalized prompt
	GetExact(ctx context.Context, hashKey string) (*MatchResult, error)

	// GetSemantic searches the vector index for nearest neighbor >= threshold
	GetSemantic(ctx context.Context, embedding []float32, threshold float64) (*MatchResult, error)

	// Set stores a new entry into both exact-hash and semantic vector indices
	Set(ctx context.Context, item *CacheItem) error

	// Len returns the current number of cached items
	Len() int

	// Stats returns aggregated telemetry metrics
	Stats() CacheStats

	// Close cleans up connections or resources
	Close() error
}
