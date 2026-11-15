package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

type Metrics struct {
	RequestsTotal       *prometheus.CounterVec
	RequestDuration     *prometheus.HistogramVec
	CacheHitsTotal      *prometheus.CounterVec
	CacheMissesTotal    prometheus.Counter
	SimilarityScores    prometheus.Histogram
	ComplexityScores    prometheus.Histogram
	TokensSavedTotal    prometheus.Counter
	CostSavedUSDTotal   prometheus.Counter
	ActiveConnections   prometheus.Gauge
}

func NewMetrics(namespace string) *Metrics {
	if namespace == "" {
		namespace = "llm_control_plane"
	}

	return &Metrics{
		RequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "requests_total",
				Help:      "Total count of LLM proxy requests processed",
			},
			[]string{"status", "cache_status", "tier"},
		),
		RequestDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "request_duration_seconds",
				Help:      "End-to-end request latency in seconds",
				Buckets:   []float64{0.005, 0.015, 0.030, 0.050, 0.100, 0.200, 0.350, 0.500, 1.0, 2.5},
			},
			[]string{"cache_status", "tier"},
		),
		CacheHitsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "cache_hits_total",
				Help:      "Total count of cache hits by type",
			},
			[]string{"cache_type"}, // "exact" or "semantic"
		),
		CacheMissesTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "cache_misses_total",
				Help:      "Total count of cache misses",
			},
		),
		SimilarityScores: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "cache_similarity_scores",
				Help:      "Distribution of cosine similarity scores on cache checks",
				Buckets:   []float64{0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00},
			},
		),
		ComplexityScores: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "router_complexity_scores",
				Help:      "Distribution of prompt complexity scores evaluated by DistilBERT",
				Buckets:   []float64{0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95},
			},
		),
		TokensSavedTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "tokens_saved_total",
				Help:      "Total number of LLM tokens saved by cache hits",
			},
		),
		CostSavedUSDTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "cost_saved_usd_total",
				Help:      "Total estimated API spend saved in USD by cache hits and routing",
			},
		),
		ActiveConnections: promauto.NewGauge(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "active_connections",
				Help:      "Number of currently active HTTP client connections",
			},
		),
	}
}
