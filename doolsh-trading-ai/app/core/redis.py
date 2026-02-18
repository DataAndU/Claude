"""In-memory cache replacing Redis — no external dependencies.

On Userland we avoid running a Redis server. Provides the same async
interface (get/set/setex/incr/expire) backed by a dict with TTL support.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class _MemoryCache:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._ttls: Dict[str, float] = {}

    def _evict(self, key: str) -> bool:
        if key in self._ttls and time.time() > self._ttls[key]:
            self._store.pop(key, None)
            self._ttls.pop(key, None)
            return True
        return False

    async def get(self, key: str) -> Optional[str]:
        self._evict(key)
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._ttls[key] = time.time() + ttl

    async def incr(self, key: str) -> int:
        self._evict(key)
        val = int(self._store.get(key, 0)) + 1
        self._store[key] = str(val)
        return val

    async def expire(self, key: str, ttl: int) -> None:
        self._ttls[key] = time.time() + ttl

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._ttls.pop(key, None)

    async def close(self) -> None:
        self._store.clear()
        self._ttls.clear()


_cache = _MemoryCache()


async def get_redis() -> _MemoryCache:
    return _cache


async def close_redis() -> None:
    await _cache.close()
