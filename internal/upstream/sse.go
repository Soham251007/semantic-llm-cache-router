package upstream

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
)

// Streamer handles Server-Sent Events (SSE) streaming for LLM chunk delivery
type Streamer struct{}

func NewStreamer() *Streamer {
	return &Streamer{}
}

// ReplayCachedStream streams a full cached completion back to a client expecting SSE
func (s *Streamer) ReplayCachedStream(w http.ResponseWriter, flusher http.Flusher, cachedResp *ChatCompletionResponse, pacing time.Duration) error {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	if len(cachedResp.Choices) == 0 {
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return nil
	}

	fullText := cachedResp.Choices[0].Message.Content
	words := strings.Fields(fullText)
	chunkID := "chatcmpl-stream-" + uuid.New().String()[:8]
	created := time.Now().Unix()

	for i, word := range words {
		deltaText := word
		if i > 0 {
			deltaText = " " + word
		}

		chunk := ChatCompletionChunk{
			ID:      chunkID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   cachedResp.Model,
			Choices: []ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: ChatMessage{
						Role:    "assistant",
						Content: deltaText,
					},
					FinishReason: nil,
				},
			},
		}

		data, err := json.Marshal(chunk)
		if err != nil {
			return err
		}

		fmt.Fprintf(w, "data: %s\n\n", string(data))
		flusher.Flush()

		if pacing > 0 {
			time.Sleep(pacing)
		}
	}

	// Send terminal chunk
	finishReason := "stop"
	finalChunk := ChatCompletionChunk{
		ID:      chunkID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   cachedResp.Model,
		Choices: []ChatCompletionChunkChoice{
			{
				Index:        0,
				Delta:        ChatMessage{},
				FinishReason: &finishReason,
			},
		},
	}
	data, _ := json.Marshal(finalChunk)
	fmt.Fprintf(w, "data: %s\n\n", string(data))
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()

	return nil
}
