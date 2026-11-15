package cache

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisCache implements RediSearch HNSW Vector Similarity Search for semantic caching
type RedisCache struct {
	client       *redis.Client
	indexName    string
	prefix       string
	ttl          time.Duration

	exactHits    atomic.Int64
	semanticHits atomic.Int64
	misses       atomic.Int64
	tokensSaved  atomic.Int64
	costSavedUSD atomic.Uint64
}

// NewRedisCache creates a connected Redis Cache with automatic index creation
func NewRedisCache(redisURL string, ttl time.Duration) (*RedisCache, error) {
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("invalid redis URL: %w", err)
	}

	client := redis.NewClient(opt)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping failed: %w", err)
	}

	rc := &RedisCache{
		client:    client,
		indexName: "idx:llm_semantic_cache",
		prefix:    "llm_cache:",
		ttl:       ttl,
	}

	// Ensure RediSearch vector index exists
	if err := rc.ensureIndex(ctx); err != nil {
		// Log warning but continue; index might already exist or require module
		fmt.Printf("[Redis] Note on index creation: %v\n", err)
	}

	return rc, nil
}

func (r *RedisCache) ensureIndex(ctx context.Context) error {
	// Check if index already exists
	res, err := r.client.Do(ctx, "FT._LIST").Result()
	if err == nil {
		if indices, ok := res.([]interface{}); ok {
			for _, idx := range indices {
				if str, ok := idx.(string); ok && str == r.indexName {
					return nil // index exists
				}
			}
		}
	}

	// Create HNSW Cosine Index for 384-dimensional MiniLM embeddings
	cmd := r.client.Do(ctx,
		"FT.CREATE", r.indexName,
		"ON", "HASH",
		"PREFIX", "1", r.prefix,
		"SCHEMA",
		"hash_key", "TAG",
		"vector", "VECTOR", "HNSW", "6",
		"TYPE", "FLOAT32",
		"DIM", "384",
		"DISTANCE_METRIC", "COSINE",
	)
	return cmd.Err()
}

// GetExact checks exact-match key in Redis
func (r *RedisCache) GetExact(ctx context.Context, hashKey string) (*MatchResult, error) {
	start := time.Now()
	key := fmt.Sprintf("%sexact:%s", r.prefix, hashKey)

	val, err := r.client.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return &MatchResult{Type: CacheTypeNone, Hit: false, Duration: time.Since(start)}, nil
	} else if err != nil {
		return nil, err
	}

	var item CacheItem
	if err := json.Unmarshal(val, &item); err != nil {
		return nil, err
	}

	r.exactHits.Add(1)
	r.tokensSaved.Add(int64(item.TokensSaved))
	r.costSavedUSD.Add(uint64(item.CostSavedUSD * 1e6))

	return &MatchResult{
		Type:       CacheTypeExact,
		Hit:        true,
		Similarity: 1.0,
		Item:       &item,
		Duration:   time.Since(start),
	}, nil
}

// GetSemantic queries Redis Stack Vector Search using FT.SEARCH with KNN
func (r *RedisCache) GetSemantic(ctx context.Context, embedding []float32, threshold float64) (*MatchResult, error) {
	start := time.Now()
	if len(embedding) == 0 {
		r.misses.Add(1)
		return &MatchResult{Type: CacheTypeNone, Hit: false, Duration: time.Since(start)}, nil
	}

	vecBytes := Float32ToBytes(embedding)
	query := "*=>[KNN 1 @vector $vec AS vector_distance]"

	cmd := r.client.Do(ctx,
		"FT.SEARCH", r.indexName,
		query,
		"PARAMS", "2", "vec", vecBytes,
		"SORTBY", "vector_distance",
		"RETURN", "2", "payload", "vector_distance",
		"DIALECT", "2",
	)

	res, err := cmd.Result()
	if err != nil {
		r.misses.Add(1)
		return &MatchResult{Type: CacheTypeNone, Hit: false, Duration: time.Since(start)}, nil
	}

	// Parse FT.SEARCH result: [total_results, doc_id, [fields...]]
	resultsList, ok := res.([]interface{})
	if !ok || len(resultsList) < 3 {
		r.misses.Add(1)
		return &MatchResult{Type: CacheTypeNone, Hit: false, Duration: time.Since(start)}, nil
	}

	fields, ok := resultsList[2].([]interface{})
	if !ok {
		r.misses.Add(1)
		return &MatchResult{Type: CacheTypeNone, Hit: false, Duration: time.Since(start)}, nil
	}

	var payloadJSON string
	var distance float64 = 1.0

	for i := 0; i < len(fields)-1; i += 2 {
		key := fmt.Sprintf("%v", fields[i])
		val := fmt.Sprintf("%v", fields[i+1])
		if key == "payload" {
			payloadJSON = val
		} else if key == "vector_distance" {
			if d, err := strconv.ParseFloat(val, 64); err == nil {
				distance = d
			}
		}
	}

	similarity := 1.0 - distance
	if similarity >= threshold && payloadJSON != "" {
		var item CacheItem
		if err := json.Unmarshal([]byte(payloadJSON), &item); err == nil {
			r.semanticHits.Add(1)
			r.tokensSaved.Add(int64(item.TokensSaved))
			r.costSavedUSD.Add(uint64(item.CostSavedUSD * 1e6))

			return &MatchResult{
				Type:       CacheTypeSemantic,
				Hit:        true,
				Similarity: similarity,
				Item:       &item,
				Duration:   time.Since(start),
			}, nil
		}
	}

	r.misses.Add(1)
	return &MatchResult{
		Type:       CacheTypeNone,
		Hit:        false,
		Similarity: similarity,
		Duration:   time.Since(start),
	}, nil
}

// Set stores entry in Redis with vector and JSON payload
func (r *RedisCache) Set(ctx context.Context, item *CacheItem) error {
	vecBytes := Float32ToBytes(item.Embedding)
	payloadBytes, err := json.Marshal(item)
	if err != nil {
		return err
	}

	key := fmt.Sprintf("%sitem:%s", r.prefix, item.ID)
	pipe := r.client.Pipeline()

	// 1. Store in HASH with vector for semantic search
	pipe.HSet(ctx, key, map[string]interface{}{
		"hash_key": item.HashKey,
		"vector":   vecBytes,
		"payload":  string(payloadBytes),
	})
	pipe.Expire(ctx, key, r.ttl)

	// 2. Store in exact match key
	exactKey := fmt.Sprintf("%sexact:%s", r.prefix, item.HashKey)
	pipe.Set(ctx, exactKey, payloadBytes, r.ttl)

	_, err = pipe.Exec(ctx)
	return err
}

func (r *RedisCache) Len() int {
	return 0
}

func (r *RedisCache) Stats() CacheStats {
	exact := r.exactHits.Load()
	semantic := r.semanticHits.Load()
	miss := r.misses.Load()
	totalHits := exact + semantic
	totalReqs := totalHits + miss

	var hitRate float64
	if totalReqs > 0 {
		hitRate = float64(totalHits) / float64(totalReqs)
	}

	return CacheStats{
		ExactHits:    exact,
		SemanticHits: semantic,
		Misses:       miss,
		HitRate:      hitRate,
		TokensSaved:  r.tokensSaved.Load(),
		CostSavedUSD: float64(r.costSavedUSD.Load()) / 1e6,
	}
}

func (r *RedisCache) Close() error {
	return r.client.Close()
}

// Float32ToBytes packs a float32 slice into IEEE-754 little-endian binary bytes for Redis Vector Search
func Float32ToBytes(floats []float32) []byte {
	bytes := make([]byte, len(floats)*4)
	for i, f := range floats {
		bits := math.Float32bits(f)
		binary.LittleEndian.PutUint32(bytes[i*4:], bits)
	}
	return bytes
}

// BytesToFloat32 unpacks binary bytes into a float32 slice
func BytesToFloat32(bytes []byte) []float32 {
	floats := make([]float32, len(bytes)/4)
	for i := range floats {
		bits := binary.LittleEndian.Uint32(bytes[i*4:])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}
