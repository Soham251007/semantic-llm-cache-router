package upstream

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/semantic-llm-cache-router/internal/config"
)

// Client abstracts LLM dispatching across simulated, OpenAI, and Gemini backends
type Client struct {
	cfg        *config.Config
	simulator  *Simulator
	httpClient *http.Client
}

func NewClient(cfg *config.Config) *Client {
	transport := &http.Transport{
		MaxIdleConns:        500,
		MaxIdleConnsPerHost: 200,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression: false,
	}

	return &Client{
		cfg:       cfg,
		simulator: NewSimulator(cfg.SimulateLatency),
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   60 * time.Second,
		},
	}
}

// Dispatch routes the completion request to the assigned tier's target model
func (c *Client) Dispatch(ctx context.Context, req *ChatCompletionRequest, tier string, promptText string) (*ChatCompletionResponse, error) {
	// Set target model based on complexity tier decision
	targetModel := c.cfg.Tier1Model
	if tier == "tier-2-frontier" {
		targetModel = c.cfg.Tier2Model
	}
	req.Model = targetModel

	switch c.cfg.UpstreamProvider {
	case "openai":
		return c.callOpenAI(ctx, req)
	case "gemini":
		return c.callGeminiOpenAICompat(ctx, req)
	default:
		// Default to high-fidelity simulator for deterministic benchmarking and testing
		return c.simulator.GenerateCompletion(ctx, req, tier, promptText)
	}
}

func (c *Client) callOpenAI(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	if c.cfg.OpenAIAPIKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY is not configured")
	}

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("%s/chat/completions", c.cfg.OpenAIBaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.cfg.OpenAIAPIKey))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upstream openai error (status %d): %s", resp.StatusCode, string(errBytes))
	}

	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, err
	}

	return &chatResp, nil
}

func (c *Client) callGeminiOpenAICompat(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	if c.cfg.GeminiAPIKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is not configured")
	}

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	// Google Gemini provides OpenAI-compatible endpoint
	endpoint := "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.cfg.GeminiAPIKey))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upstream gemini error (status %d): %s", resp.StatusCode, string(errBytes))
	}

	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, err
	}

	return &chatResp, nil
}
