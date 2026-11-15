package cache

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// MemoryCache is a high-performance in-memory cache combining:
// 1. O(1) Exact Hash Index (L1)
// 2. Vector Cosine Similarity Search with unrolled dot products (L2)
type MemoryCache struct {
	mu           sync.RWMutex
	exactIndex   map[string]*CacheItem
	vectorItems  []*CacheItem
	maxCapacity  int
	ttl          time.Duration

	// Atomic telemetry counters
	exactHits    atomic.Int64
	semanticHits atomic.Int64
	misses       atomic.Int64
	tokensSaved  atomic.Int64
	costSavedUSD atomic.Uint64 // stored as micro-USD (USD * 1e6) for atomic updates
}

// NewMemoryCache creates an initialized in-memory vector cache
func NewMemoryCache(maxCapacity int, ttl time.Duration) *MemoryCache {
	if maxCapacity <= 0 {
		maxCapacity = 100000
	}
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}

	mc := &MemoryCache{
		exactIndex:  make(map[string]*CacheItem, 4096),
		vectorItems: make([]*CacheItem, 0, 4096),
		maxCapacity: maxCapacity,
		ttl:         ttl,
	}

	// Periodic cleanup routine for expired items
	go mc.evictionLoop()

	return mc
}

// GetExact performs an O(1) lookup on the prompt hash
func (m *MemoryCache) GetExact(ctx context.Context, hashKey string) (*MatchResult, error) {
	start := time.Now()
	m.mu.RLock()
	item, exists := m.exactIndex[hashKey]
	m.mu.RUnlock()

	if !exists {
		return &MatchResult{
			Type:     CacheTypeNone,
			Hit:      false,
			Duration: time.Since(start),
		}, nil
	}

	// Check expiration
	if time.Now().After(item.ExpiresAt) {
		return &MatchResult{
			Type:     CacheTypeNone,
			Hit:      false,
			Duration: time.Since(start),
		}, nil
	}

	m.exactHits.Add(1)
	m.tokensSaved.Add(int64(item.TokensSaved))
	m.costSavedUSD.Add(uint64(item.CostSavedUSD * 1e6))

	return &MatchResult{
		Type:       CacheTypeExact,
		Hit:        true,
		Similarity: 1.0,
		Item:       item,
		Duration:   time.Since(start),
	}, nil
}

// GetSemantic iterates over normalized vector embeddings and calculates cosine similarity
func (m *MemoryCache) GetSemantic(ctx context.Context, embedding []float32, threshold float64) (*MatchResult, error) {
	start := time.Now()
	if len(embedding) == 0 {
		m.misses.Add(1)
		return &MatchResult{Type: CacheTypeNone, Hit: false, Duration: time.Since(start)}, nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	var bestItem *CacheItem
	var bestSimilarity float64 = -1.0
	now := time.Now()

	for _, item := range m.vectorItems {
		if now.After(item.ExpiresAt) {
			continue
		}
		sim := DotProductNormalized(embedding, item.Embedding)
		if sim > bestSimilarity {
			bestSimilarity = sim
			bestItem = item
		}
	}

	if bestSimilarity >= threshold && bestItem != nil {
		m.semanticHits.Add(1)
		m.tokensSaved.Add(int64(bestItem.TokensSaved))
		m.costSavedUSD.Add(uint64(bestItem.CostSavedUSD * 1e6))

		return &MatchResult{
			Type:       CacheTypeSemantic,
			Hit:        true,
			Similarity: bestSimilarity,
			Item:       bestItem,
			Duration:   time.Since(start),
		}, nil
	}

	m.misses.Add(1)
	return &MatchResult{
		Type:       CacheTypeNone,
		Hit:        false,
		Similarity: bestSimilarity,
		Duration:   time.Since(start),
	}, nil
}

// Set saves an item into both exact and vector index
func (m *MemoryCache) Set(ctx context.Context, item *CacheItem) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	if item.CreatedAt.IsZero() {
		item.CreatedAt = now
	}
	if item.ExpiresAt.IsZero() {
		item.ExpiresAt = now.Add(m.ttl)
	}

	// Evict oldest if capacity exceeded
	if len(m.vectorItems) >= m.maxCapacity {
		m.evictOldestLocked()
	}

	// Update indices
	m.exactIndex[item.HashKey] = item
	m.vectorItems = append(m.vectorItems, item)

	return nil
}

func (m *MemoryCache) evictOldestLocked() {
	if len(m.vectorItems) == 0 {
		return
	}
	// Evict first element (FIFO / oldest)
	oldest := m.vectorItems[0]
	delete(m.exactIndex, oldest.HashKey)
	m.vectorItems = m.vectorItems[1:]
}

func (m *MemoryCache) evictionLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		m.mu.Lock()
		now := time.Now()
		newVectors := make([]*CacheItem, 0, len(m.vectorItems))
		for _, item := range m.vectorItems {
			if now.Before(item.ExpiresAt) {
				newVectors = append(newVectors, item)
			} else {
				delete(m.exactIndex, item.HashKey)
			}
		}
		m.vectorItems = newVectors
		m.mu.Unlock()
	}
}

// Len returns current stored items count
func (m *MemoryCache) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.vectorItems)
}

// Stats returns operational metrics
func (m *MemoryCache) Stats() CacheStats {
	m.mu.RLock()
	total := len(m.vectorItems)
	m.mu.RUnlock()

	exact := m.exactHits.Load()
	semantic := m.semanticHits.Load()
	miss := m.misses.Load()
	totalHits := exact + semantic
	totalReqs := totalHits + miss

	var hitRate float64
	if totalReqs > 0 {
		hitRate = float64(totalHits) / float64(totalReqs)
	}

	costMicro := m.costSavedUSD.Load()

	return CacheStats{
		TotalEntries: total,
		ExactHits:    exact,
		SemanticHits: semantic,
		Misses:       miss,
		HitRate:      hitRate,
		TokensSaved:  m.tokensSaved.Load(),
		CostSavedUSD: float64(costMicro) / 1e6,
	}
}

// Close is a no-op for in-memory cache
func (m *MemoryCache) Close() error {
	return nil
}

// DotProductNormalized computes cosine similarity between two L2-normalized float32 vectors.
// Uses 4-way loop unrolling for maximum hardware vectorization and CPU pipelining.
func DotProductNormalized(a, b []float32) float64 {
	n := len(a)
	if n != len(b) || n == 0 {
		return 0.0
	}

	var sum float64
	i := 0
	for ; i+3 < n; i += 4 {
		sum += float64(a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3])
	}
	for ; i < n; i++ {
		sum += float64(a[i] * b[i])
	}

	return sum
}
