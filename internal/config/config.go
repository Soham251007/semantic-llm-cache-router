package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds runtime configuration loaded from environment variables
type Config struct {
	Port                int           `json:"port"`
	MLServiceURL        string        `json:"ml_service_url"`
	CacheBackend        string        `json:"cache_backend"` // "memory" or "redis"
	RedisURL            string        `json:"redis_url"`
	SimilarityThreshold float64       `json:"similarity_threshold"` // default 0.94
	ComplexityAlpha     float64       `json:"complexity_alpha"`     // default 0.45
	UpstreamProvider    string        `json:"upstream_provider"`    // "simulator", "openai", "gemini"
	OpenAIAPIKey        string        `json:"openai_api_key"`
	OpenAIBaseURL       string        `json:"openai_base_url"`
	GeminiAPIKey        string        `json:"gemini_api_key"`
	Tier1Model          string        `json:"tier1_model"` // fast/cheap e.g. llama-3-8b, gpt-4o-mini, gemini-1.5-flash
	Tier2Model          string        `json:"tier2_model"` // frontier e.g. gpt-4o, claude-3-5-sonnet, gemini-1.5-pro
	CacheTTL            time.Duration `json:"cache_ttl"`
	MaxMemoryEntries    int           `json:"max_memory_entries"`
	EnableMetrics       bool          `json:"enable_metrics"`
	SimulateLatency     bool          `json:"simulate_latency"` // whether simulator adds realistic token streaming latency
}

// LoadFromEnv loads configuration with high-performance production defaults
func LoadFromEnv() *Config {
	return &Config{
		Port:                getEnvInt("PORT", 8080),
		MLServiceURL:        getEnvString("ML_SERVICE_URL", "http://localhost:8001"),
		CacheBackend:        getEnvString("CACHE_BACKEND", "memory"),
		RedisURL:            getEnvString("REDIS_URL", "redis://localhost:6379"),
		SimilarityThreshold: getEnvFloat("SIMILARITY_THRESHOLD", 0.94),
		ComplexityAlpha:     getEnvFloat("COMPLEXITY_ALPHA", 0.45),
		UpstreamProvider:    getEnvString("UPSTREAM_PROVIDER", "simulator"),
		OpenAIAPIKey:        getEnvString("OPENAI_API_KEY", ""),
		OpenAIBaseURL:       getEnvString("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		GeminiAPIKey:        getEnvString("GEMINI_API_KEY", ""),
		Tier1Model:          getEnvString("TIER1_MODEL", "gpt-4o-mini"),
		Tier2Model:          getEnvString("TIER2_MODEL", "gpt-4o"),
		CacheTTL:            time.Duration(getEnvInt("CACHE_TTL_SECONDS", 86400)) * time.Second,
		MaxMemoryEntries:    getEnvInt("MAX_MEMORY_ENTRIES", 100000),
		EnableMetrics:       getEnvBool("ENABLE_METRICS", true),
		SimulateLatency:     getEnvBool("SIMULATE_LATENCY", true),
	}
}

func getEnvString(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		return val
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		if parsed, err := strconv.Atoi(val); err == nil {
			return parsed
		}
	}
	return fallback
}

func getEnvFloat(key string, fallback float64) float64 {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		if parsed, err := strconv.ParseFloat(val, 64); err == nil {
			return parsed
		}
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		if parsed, err := strconv.ParseBool(val); err == nil {
			return parsed
		}
	}
	return fallback
}
