"""
Feature Store — Manages feature retrieval and caching for real-time prediction.
In production this would connect to Feast, Tecton, or a Redis feature store.
"""
import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Simple in-memory feature store with TTL-based expiry.
    In production: replace _cache with Redis or a dedicated feature store.
    """

    DEFAULT_TTL_SECONDS = 3600  # 1 hour

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._hit_count = 0
        self._miss_count = 0

    def get(self, entity_id: str, feature_names: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Retrieve features for an entity from the store."""
        key = self._make_key(entity_id)
        entry = self._cache.get(key)

        if entry is None:
            self._miss_count += 1
            return None

        # Check TTL
        if datetime.utcnow() > entry["expires_at"]:
            del self._cache[key]
            self._miss_count += 1
            return None

        self._hit_count += 1
        features = entry["features"]

        if feature_names:
            return {k: v for k, v in features.items() if k in feature_names}
        return features

    def set(
        self,
        entity_id: str,
        features: Dict[str, Any],
        ttl_seconds: int = DEFAULT_TTL_SECONDS
    ):
        """Store features for an entity."""
        key = self._make_key(entity_id)
        self._cache[key] = {
            "features": features,
            "entity_id": entity_id,
            "stored_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl_seconds)
        }

    def invalidate(self, entity_id: str):
        """Remove an entity's features from the store."""
        key = self._make_key(entity_id)
        self._cache.pop(key, None)

    def get_stats(self) -> Dict:
        """Return cache hit/miss statistics."""
        total = self._hit_count + self._miss_count
        return {
            "total_requests": total,
            "cache_hits": self._hit_count,
            "cache_misses": self._miss_count,
            "hit_rate": round(self._hit_count / max(total, 1), 3),
            "entries_cached": len(self._cache)
        }

    def _make_key(self, entity_id: str) -> str:
        return hashlib.md5(entity_id.encode()).hexdigest()


# Singleton instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store
