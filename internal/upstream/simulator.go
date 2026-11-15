package upstream

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ModelPricing defines costs per 1,000 tokens in USD
type ModelPricing struct {
	PromptPer1k     float64
	CompletionPer1k float64
}

var PricingTable = map[string]ModelPricing{
	"tier-1-fast": {
		PromptPer1k:     0.00015, // $0.15 / 1M tokens
		CompletionPer1k: 0.00060, // $0.60 / 1M tokens
	},
	"tier-2-frontier": {
		PromptPer1k:     0.00250, // $2.50 / 1M tokens (16.6x higher!)
		CompletionPer1k: 0.01000, // $10.00 / 1M tokens
	},
}

// Simulator provides deterministic, realistic LLM responses with realistic latencies
type Simulator struct {
	simulateLatency bool
}

func NewSimulator(simulateLatency bool) *Simulator {
	return &Simulator{simulateLatency: simulateLatency}
}

// GenerateCompletion produces a full ChatCompletionResponse
func (s *Simulator) GenerateCompletion(ctx context.Context, req *ChatCompletionRequest, tier string, promptText string) (*ChatCompletionResponse, error) {
	isTier2 := strings.Contains(tier, "tier-2") || strings.Contains(tier, "frontier")

	// Calculate realistic delays
	if s.simulateLatency {
		var delay time.Duration
		if isTier2 {
			// Frontier model latency: 320ms - 480ms p95
			delay = time.Duration(320+rand.Intn(160)) * time.Millisecond
		} else {
			// Fast model latency: 50ms - 90ms p95
			delay = time.Duration(50+rand.Intn(40)) * time.Millisecond
		}

		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Generate contextual synthetic completion content
	var content string
	var completionTokens int
	promptTokens := EstimateTokens(promptText)

	if isTier2 {
		completionTokens = 250 + rand.Intn(200)
		content = fmt.Sprintf(
			"### Frontier Reasoning Execution [%s]\n\n"+
				"1. **Architectural Analysis**: The system invariants require lock-free synchronization or partition tolerance.\n"+
				"2. **Invariant Proof**: Formally verifying state transitions ensures zero race conditions across concurrent threads.\n"+
				"3. **Implementation Synthesis**: Production-grade verification completed with safety proofs.\n\n"+
				"Execution result synthesized successfully for query: \"%s\"",
			req.Model, truncateStr(promptText, 60),
		)
	} else {
		completionTokens = 60 + rand.Intn(50)
		content = fmt.Sprintf(
			"### Fast Tier Execution [%s]\n\n"+
				"Task executed directly with high-throughput extraction/formatting.\n"+
				"Result: Output formatted successfully for prompt: \"%s\".",
			req.Model, truncateStr(promptText, 60),
		)
	}

	resp := &ChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String()[:12],
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []ChatCompletionChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}

	return resp, nil
}

// CalculateCost calculates estimated USD cost for a given usage and tier
func CalculateCost(tier string, usage Usage) float64 {
	pricing, ok := PricingTable[tier]
	if !ok {
		pricing = PricingTable["tier-1-fast"]
	}
	return (float64(usage.PromptTokens)/1000.0)*pricing.PromptPer1k +
		(float64(usage.CompletionTokens)/1000.0)*pricing.CompletionPer1k
}

// EstimateTokens provides a fast 4 chars per token estimation
func EstimateTokens(text string) int {
	if len(text) == 0 {
		return 1
	}
	tokens := len(text) / 4
	if tokens < 1 {
		return 1
	}
	return tokens
}

func truncateStr(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}
