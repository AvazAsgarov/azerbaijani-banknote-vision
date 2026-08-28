"""On-disk LRU cache for DINOv2 ViT-L/14 1024-d embeddings."""
import hashlib, json
import numpy as np
from pathlib import Path

class EmbeddingCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "cache_index.json"
        self._index = json.loads(self._index_path.read_text()) if self._index_path.exists() else {}

    def _key(self, path):
        return hashlib.sha256(path.encode()).hexdigest()[:16]

    def get(self, image_path):
        npy = self.cache_dir / "{}.npy".format(self._key(image_path))
        return np.load(str(npy)) if npy.exists() else None

    def put(self, image_path, embedding):
        key = self._key(image_path)
        np.save(str(self.cache_dir / "{}.npy".format(key)), embedding)
        self._index[image_path] = key
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def clear(self):
        deleted = 0
        for npy in self.cache_dir.glob("*.npy"):
            npy.unlink(); deleted += 1
        self._index.clear()
        if self._index_path.exists(): self._index_path.unlink()
        return deleted

    def __len__(self):
        return len(list(self.cache_dir.glob("*.npy")))
