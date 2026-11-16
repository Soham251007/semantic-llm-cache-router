"""
High-Performance ML Microservice for Semantic Caching & Complexity Routing.
Powered by FastAPI, ONNX Runtime (INT8 all-MiniLM-L6-v2), and fine-tuned DistilBERT.
"""

import os
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

try:
    from embedding_engine import EmbeddingEngine
    from complexity_scorer import ComplexityScorer
    from semantic_guard import SemanticGuard
except ImportError:
    from services.ml_service.embedding_engine import EmbeddingEngine
    from services.ml_service.complexity_scorer import ComplexityScorer
    from services.ml_service.semantic_guard import SemanticGuard

app = FastAPI(
    title="Semantic Cache & Complexity Router ML Service",
    description="Sub-10ms ONNX sentence embeddings & calibrated prompt complexity classifier",
    version="1.0.0"
)

# Global models initialized at startup
embedding_engine: Optional[EmbeddingEngine] = None
complexity_scorer: Optional[ComplexityScorer] = None

@app.on_event("startup")
def startup_event():
    global embedding_engine, complexity_scorer
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    print(f"Loading ML models from {models_dir}...")
    embedding_engine = EmbeddingEngine(models_dir)
    complexity_scorer = ComplexityScorer(models_dir)
    print("✓ All ML models loaded and warmed up successfully.")

class EmbedRequest(BaseModel):
    text: str = Field(..., description="Prompt text to embed")
    max_length: Optional[int] = Field(128, description="Maximum sequence length")

class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int
    latency_ms: float

class BatchEmbedRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 128

class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    count: int
    latency_ms: float

class ComplexityRequest(BaseModel):
    text: str = Field(..., description="Prompt text to analyze")
    alpha: Optional[float] = Field(0.45, description="Routing threshold alpha")

class ComplexityResponse(BaseModel):
    complexity_score: float
    neural_prob: float
    threshold_alpha: float
    recommended_tier: str
    is_complex: bool
    signals: List[str]
    latency_ms: float

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Prompt text to analyze and embed")
    alpha: Optional[float] = Field(0.45, description="Routing threshold alpha")
    max_length: Optional[int] = 128

class AnalyzeResponse(BaseModel):
    embedding: List[float]
    dimension: int
    complexity_score: float
    recommended_tier: str
    is_complex: bool
    signals: List[str]
    semantic_signature: str = Field("", description="Normalized semantic constraint & polarity signature")
    polarity: str = Field("positive", description="Query logical polarity: positive or negative")
    embed_latency_ms: float
    complexity_latency_ms: float
    total_latency_ms: float

class VerifyRequest(BaseModel):
    query_text: str = Field(..., description="Incoming prompt text")
    candidate_text: str = Field(..., description="Cached candidate prompt text")
    similarity: float = Field(..., description="Cosine similarity score between embeddings")
    threshold: Optional[float] = Field(0.94, description="Similarity threshold")

class VerifyResponse(BaseModel):
    is_valid_cache_hit: bool
    reason: str
    similarity: float
    query_signature: Optional[str] = ""
    candidate_signature: Optional[str] = ""
    latency_ms: float

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": embedding_engine is not None and complexity_scorer is not None,
        "engine": "ONNX Runtime INT8 + DistilBERT"
    }

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not embedding_engine:
        raise HTTPException(status_code=503, detail="Embedding engine not initialized")
    t0 = time.perf_counter()
    emb = embedding_engine.embed(req.text, max_length=req.max_length)
    dur = (time.perf_counter() - t0) * 1000.0
    return EmbedResponse(
        embedding=emb,
        dimension=len(emb),
        latency_ms=round(dur, 2)
    )

@app.post("/embed/batch", response_model=BatchEmbedResponse)
def embed_batch(req: BatchEmbedRequest):
    if not embedding_engine:
        raise HTTPException(status_code=503, detail="Embedding engine not initialized")
    t0 = time.perf_counter()
    embs = embedding_engine.embed_batch(req.texts, max_length=req.max_length)
    dur = (time.perf_counter() - t0) * 1000.0
    return BatchEmbedResponse(
        embeddings=embs,
        count=len(embs),
        latency_ms=round(dur, 2)
    )

@app.post("/score-complexity", response_model=ComplexityResponse)
def score_complexity(req: ComplexityRequest):
    if not complexity_scorer:
        raise HTTPException(status_code=503, detail="Complexity scorer not initialized")
    result = complexity_scorer.score(req.text, alpha=req.alpha)
    return ComplexityResponse(**result)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Combined single-pass endpoint: Computes embedding vector AND evaluates prompt difficulty.
    Enables the Go reverse proxy to issue only 1 network hop on incoming requests.
    """
    if not embedding_engine or not complexity_scorer:
        raise HTTPException(status_code=503, detail="Models not initialized")
        
    t_start = time.perf_counter()
    
    # Embedding
    t0 = time.perf_counter()
    emb = embedding_engine.embed(req.text, max_length=req.max_length)
    embed_ms = (time.perf_counter() - t0) * 1000.0
    
    # Complexity
    t1 = time.perf_counter()
    comp = complexity_scorer.score(req.text, alpha=req.alpha)
    comp_ms = (time.perf_counter() - t1) * 1000.0

    # Semantic Guard Signature Extraction (Polarity, Quantifiers, Constraints)
    guard_sig = SemanticGuard.extract_signature(req.text)
    
    total_ms = (time.perf_counter() - t_start) * 1000.0
    
    return AnalyzeResponse(
        embedding=emb,
        dimension=len(emb),
        complexity_score=comp["complexity_score"],
        recommended_tier=comp["recommended_tier"],
        is_complex=comp["is_complex"],
        signals=comp["signals"],
        semantic_signature=guard_sig["signature"],
        polarity=guard_sig["polarity"],
        embed_latency_ms=round(embed_ms, 2),
        complexity_latency_ms=round(comp_ms, 2),
        total_latency_ms=round(total_ms, 2)
    )

@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    """
    Two-stage cache validation: verifies whether a candidate cache entry is functionally
    and semantically identical to the query prompt, eliminating false-positive cache hits.
    """
    valid, reason, meta = SemanticGuard.verify(
        req.query_text,
        req.candidate_text,
        req.similarity,
        threshold=req.threshold or 0.94
    )
    return VerifyResponse(
        is_valid_cache_hit=valid,
        reason=reason,
        similarity=req.similarity,
        query_signature=meta.get("query_signature", ""),
        candidate_signature=meta.get("candidate_signature", ""),
        latency_ms=round(meta.get("latency_ms", 0.0), 3)
    )

if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
