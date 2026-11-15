package cache

import (
	"testing"

	"github.com/semantic-llm-cache-router/internal/upstream"
)

func TestNormalizeText(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"  Hello   World!  ", "hello world!"},
		{"Extract\t\n  Invoice \n Data", "extract invoice data"},
		{"WHAT IS THE CAPITAL OF FRANCE?", "what is the capital of france?"},
	}

	for _, tt := range tests {
		got := NormalizeText(tt.input)
		if got != tt.expected {
			t.Errorf("NormalizeText(%q) = %q; want %q", tt.input, got, tt.expected)
		}
	}
}

func TestHashPrompt(t *testing.T) {
	h1 := HashPrompt("hello world")
	h2 := HashPrompt("hello world")
	h3 := HashPrompt("different query")

	if h1 != h2 {
		t.Errorf("expected deterministic hash output for identical input")
	}
	if h1 == h3 {
		t.Errorf("expected different hashes for different inputs")
	}
}

func TestExtractCanonicalPrompt(t *testing.T) {
	req := &upstream.ChatCompletionRequest{
		Messages: []upstream.ChatMessage{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "  Summarize this text. "},
		},
	}

	raw, canonical, hashKey := ExtractCanonicalPrompt(req)
	if raw == "" || canonical == "" || hashKey == "" {
		t.Fatalf("expected non-empty extraction fields")
	}

	userPrompt := ExtractLastUserPrompt(req)
	if userPrompt != "Summarize this text." {
		t.Errorf("ExtractLastUserPrompt got %q; want %q", userPrompt, "Summarize this text.")
	}
}
