import pytest
from datetime import datetime, timedelta, UTC
from embeddings import (
    serialize_embedding, deserialize_embedding, estimate_tokens,
    batch_by_tokens, compute_time_penalty, EMBEDDING_DIM, MAX_TOKENS_PER_REQUEST,
    TIME_DECAY_HALF_LIFE_DAYS, format_temporal_prefix, enrich_for_embedding
)


def test_serialize_deserialize_roundtrip():
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    blob = serialize_embedding(embedding)
    result = deserialize_embedding(blob)
    
    assert len(result) == len(embedding)
    for a, b in zip(result, embedding):
        assert abs(a - b) < 1e-6


def test_serialize_embedding_size():
    embedding = [0.0] * 10
    blob = serialize_embedding(embedding)
    assert len(blob) == 10 * 4


def test_deserialize_embedding_full_dim():
    embedding = [float(i) / EMBEDDING_DIM for i in range(EMBEDDING_DIM)]
    blob = serialize_embedding(embedding)
    result = deserialize_embedding(blob)
    assert len(result) == EMBEDDING_DIM


def test_embedding_dim_constant():
    assert EMBEDDING_DIM == 1536


def test_max_tokens_constant():
    assert MAX_TOKENS_PER_REQUEST == 250000


def test_estimate_tokens():
    assert estimate_tokens("12345678") == 2
    assert estimate_tokens("") == 0


def test_batch_by_tokens_single_batch():
    texts = ["short", "texts", "here"]
    batches = batch_by_tokens(texts)
    assert len(batches) == 1
    assert batches[0] == texts


def test_batch_by_tokens_splits_large():
    texts = ["x" * 1000 for _ in range(10)]
    batches = batch_by_tokens(texts, max_tokens=500)
    assert len(batches) > 1
    all_texts = [t for b in batches for t in b]
    assert all_texts == texts


def test_batch_by_tokens_empty():
    assert batch_by_tokens([]) == []


def test_compute_time_penalty_now():
    now = datetime.now(UTC)
    penalty = compute_time_penalty(now)
    assert penalty < 0.01


def test_compute_time_penalty_half_life():
    half_life_ago = datetime.now(UTC) - timedelta(days=TIME_DECAY_HALF_LIFE_DAYS)
    penalty = compute_time_penalty(half_life_ago)
    assert 0.49 < penalty < 0.51


def test_compute_time_penalty_old():
    old = datetime.now(UTC) - timedelta(days=365)
    penalty = compute_time_penalty(old)
    assert penalty > 0.99


def test_compute_time_penalty_none():
    penalty = compute_time_penalty(None)
    assert penalty == 0.5


def test_format_temporal_prefix_single_timestamp():
    ts = datetime(2025, 6, 15, tzinfo=UTC)
    assert format_temporal_prefix(ts) == "In June 2025: "


def test_format_temporal_prefix_same_month():
    start = datetime(2025, 6, 1, tzinfo=UTC)
    end = datetime(2025, 6, 30, tzinfo=UTC)
    assert format_temporal_prefix(start, end) == "In June 2025: "


def test_format_temporal_prefix_different_months_same_year():
    start = datetime(2025, 3, 1, tzinfo=UTC)
    end = datetime(2025, 6, 30, tzinfo=UTC)
    assert format_temporal_prefix(start, end) == "From March to June 2025: "


def test_format_temporal_prefix_different_years():
    start = datetime(2024, 11, 1, tzinfo=UTC)
    end = datetime(2025, 2, 28, tzinfo=UTC)
    assert format_temporal_prefix(start, end) == "From November 2024 to February 2025: "


def test_format_temporal_prefix_none():
    assert format_temporal_prefix(None) == ""


def test_enrich_for_embedding():
    ts = datetime(2025, 6, 15, tzinfo=UTC)
    result = enrich_for_embedding("User relocated to Austin", ts)
    assert result == "In June 2025: User relocated to Austin"


def test_enrich_for_embedding_no_timestamp():
    result = enrich_for_embedding("Some fact", None)
    assert result == "Some fact"
