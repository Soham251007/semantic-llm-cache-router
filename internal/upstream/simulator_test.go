package upstream

import (
	"context"
	"testing"
)

func TestSimulator_GenerateCompletion(t *testing.T) {
	sim := NewSimulator(false) // no artificial sleep in tests

	req := &ChatCompletionRequest{
		Model: "test-model",
	}

	// 1. Tier-1 Fast Model
	resp1, err := sim.GenerateCompletion(context.Background(), req, "tier-1-fast", "Summarize this article")
	if err != nil {
		t.Fatalf("tier-1 generation failed: %v", err)
	}
	if len(resp1.Choices) == 0 {
		t.Fatalf("expected non-empty choices")
	}
	if resp1.Usage.TotalTokens <= 0 {
		t.Errorf("expected positive total tokens")
	}

	cost1 := CalculateCost("tier-1-fast", resp1.Usage)
	if cost1 <= 0 {
		t.Errorf("expected positive cost calculation")
	}

	// 2. Tier-2 Frontier Model
	resp2, err := sim.GenerateCompletion(context.Background(), req, "tier-2-frontier", "Prove P != NP or verify raft consensus invariants")
	if err != nil {
		t.Fatalf("tier-2 generation failed: %v", err)
	}
	cost2 := CalculateCost("tier-2-frontier", resp2.Usage)

	// Frontier cost per token is much higher than fast model
	if cost2 <= cost1 {
		t.Errorf("expected frontier cost (%f) to be significantly higher than tier-1 (%f)", cost2, cost1)
	}
}

func TestEstimateTokens(t *testing.T) {
	short := "Hello"
	long := "This is a much longer prompt string designed to test token estimation length."

	if EstimateTokens(short) != 1 {
		t.Errorf("expected 1 token for short text, got %d", EstimateTokens(short))
	}
	if EstimateTokens(long) <= EstimateTokens(short) {
		t.Errorf("expected longer text to yield more estimated tokens")
	}
}
