package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"unicode"

	"github.com/semantic-llm-cache-router/internal/upstream"
)

// ExtractCanonicalPrompt extracts and normalizes the conversational prompt from an OpenAI request
func ExtractCanonicalPrompt(req *upstream.ChatCompletionRequest) (raw string, canonical string, hashKey string) {
	if len(req.Messages) == 0 {
		return "", "", ""
	}

	var sb strings.Builder
	for i, msg := range req.Messages {
		trimmedContent := strings.TrimSpace(msg.Content)
		if trimmedContent == "" {
			continue
		}
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>", msg.Role, trimmedContent))
	}

	raw = sb.String()
	canonical = NormalizeText(raw)
	hashKey = HashPrompt(canonical)
	return raw, canonical, hashKey
}

// ExtractLastUserPrompt gets the newest user prompt for semantic embedding & complexity scoring
func ExtractLastUserPrompt(req *upstream.ChatCompletionRequest) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if strings.EqualFold(req.Messages[i].Role, "user") {
			return strings.TrimSpace(req.Messages[i].Content)
		}
	}
	if len(req.Messages) > 0 {
		return strings.TrimSpace(req.Messages[len(req.Messages)-1].Content)
	}
	return ""
}

// NormalizeText canonicalizes prompt text: lowercases, collapses whitespace, trims
func NormalizeText(input string) string {
	var b strings.Builder
	b.Grow(len(input))

	inWhitespace := false
	for _, r := range input {
		if unicode.IsSpace(r) {
			if !inWhitespace {
				b.WriteRune(' ')
				inWhitespace = true
			}
		} else {
			b.WriteRune(unicode.ToLower(r))
			inWhitespace = false
		}
	}

	return strings.TrimSpace(b.String())
}

// HashPrompt returns SHA-256 hexadecimal string for fast O(1) exact-match caching
func HashPrompt(text string) string {
	h := sha256.New()
	h.Write([]byte(text))
	return hex.EncodeToString(h.Sum(nil))
}
