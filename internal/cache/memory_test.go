package cache

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/semantic-llm-cache-router/internal/upstream"
)

func TestDotProductNormalized(t *testing.T) {
	// Identical normalized vectors -> dot product must be 1.0
	v1 := []float32{1.0, 0.0, 0.0}
	v2 := []float32{1.0, 0.0, 0.0}
	sim := DotProductNormalized(v1, v2)
	if math.Abs(sim-1.0) > 1e-5 {
		t.Errorf("DotProductNormalized identical vectors = %f; want 1.0", sim)
	}

	// Orthogonal vectors -> dot product must be 0.0
	v3 := []float32{0.0, 1.0, 0.0}
	simOrth := DotProductNormalized(v1, v3)
	if math.Abs(simOrth) > 1e-5 {
		t.Errorf("DotProductNormalized orthogonal vectors = %f; want 0.0", simOrth)
	}
}

func TestMemoryCache_ExactAndSemantic(t *testing.T) {
	ctx := context.Background()
	c := NewMemoryCache(1000, 1*time.Hour)

	// Create normalized 384-dim dummy vector
	vecA := make([]float32, 384)
	vecA[0] = 1.0 // normalized

	// Near vector (similarity = 0.96)
	vecNear := make([]float32, 384)
	vecNear[0] = 0.96
	vecNear[1] = float32(math.Sqrt(1.0 - 0.96*0.96)) // normalize

	// Far vector (similarity = 0.50)
	vecFar := make([]float32, 384)
	vecFar[0] = 0.50
	vecFar[1] = float32(math.Sqrt(1.0 - 0.50*0.50))

	item := &CacheItem{
		ID:        "item-1",
		Prompt:    "How do I reset my password?",
		Canonical: "how do i reset my password?",
		HashKey:   HashPrompt("how do i reset my password?"),
		Embedding: vecA,
		Response: &upstream.ChatCompletionResponse{
			ID: "cmpl-test",
			Choices: []upstream.ChatCompletionChoice{
				{Message: upstream.ChatMessage{Role: "assistant", Content: "Click on Settings -> Security -> Reset Password."}},
			},
		},
		Tier:         "tier-1-fast",
		TokensSaved:  120,
		CostSavedUSD: 0.0001,
	}

	if err := c.Set(ctx, item); err != nil {
		t.Fatalf("failed to set cache item: %v", err)
	}

	// 1. Test Exact Match
	exactResult, err := c.GetExact(ctx, item.HashKey)
	if err != nil || !exactResult.Hit {
		t.Fatalf("expected exact match hit, got err=%v hit=%v", err, exactResult.Hit)
	}
	if exactResult.Type != CacheTypeExact {
		t.Errorf("expected CacheTypeExact, got %s", exactResult.Type)
	}

	// 2. Test Semantic Match (Threshold 0.94)
	semanticResult, err := c.GetSemantic(ctx, vecNear, 0.94)
	if err != nil || !semanticResult.Hit {
		t.Fatalf("expected semantic match hit, got err=%v hit=%v", err, semanticResult.Hit)
	}
	if semanticResult.Type != CacheTypeSemantic {
		t.Errorf("expected CacheTypeSemantic, got %s", semanticResult.Type)
	}
	if semanticResult.Similarity < 0.94 {
		t.Errorf("expected similarity >= 0.94, got %f", semanticResult.Similarity)
	}

	// 3. Test Semantic Miss for Far Vector (Similarity ~0.50 < 0.94)
	farResult, err := c.GetSemantic(ctx, vecFar, 0.94)
	if err != nil || farResult.Hit {
		t.Fatalf("expected semantic miss for far vector, got hit=%v", farResult.Hit)
	}
	if farResult.Type != CacheTypeNone {
		t.Errorf("expected CacheTypeNone on miss, got %s", farResult.Type)
	}
}
