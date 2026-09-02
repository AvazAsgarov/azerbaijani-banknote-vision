"""Unit tests for DINOv2 embedding cache."""
import numpy as np
import pytest
from src.embeddings.embedding_cache import EmbeddingCache

class TestEmbeddingCache:
    def test_round_trip(self, tmp_path):
        cache = EmbeddingCache(tmp_path)
        emb = np.random.randn(1024).astype(np.float32)
        cache.put("/img/001.jpg", emb)
        np.testing.assert_array_almost_equal(emb, cache.get("/img/001.jpg"))

    def test_missing_returns_none(self, tmp_path):
        assert EmbeddingCache(tmp_path).get("/does/not/exist.jpg") is None

    def test_len(self, tmp_path):
        cache = EmbeddingCache(tmp_path)
        for i in range(5):
            cache.put("/img/{}.jpg".format(i), np.zeros(1024, dtype=np.float32))
        assert len(cache) == 5

    def test_clear(self, tmp_path):
        cache = EmbeddingCache(tmp_path)
        cache.put("/img/a.jpg", np.zeros(1024, dtype=np.float32))
        cache.clear()
        assert len(cache) == 0
