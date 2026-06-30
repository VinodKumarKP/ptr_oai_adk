import pytest
import time
from unittest.mock import patch
from oai_agent_server.utils.query_cache import (
    CacheKey, CacheEntry, QueryResultCache, CacheableQueryMixin
)

def test_cache_key():
    key1 = CacheKey("query1", {"a": 1, "b": 2})
    key2 = CacheKey("query1", {"b": 2, "a": 1})
    key3 = CacheKey("query2", {"a": 1})
    
    assert hash(key1) == hash(key2)
    assert key1 == key2
    assert key1 != key3
    assert key1 != "not-a-cache-key"
    assert repr(key1) == "CacheKey(query1, {'a': 1, 'b': 2})"

def test_cache_entry():
    entry = CacheEntry("data", ttl=10)
    assert not entry.is_expired()
    assert repr(entry).startswith("CacheEntry(")
    
    with patch('time.time', return_value=time.time() + 15):
        assert entry.is_expired()

@pytest.mark.asyncio
async def test_query_result_cache():
    cache = QueryResultCache(max_size=10, default_ttl=5)
    
    key1 = CacheKey("q1")
    key2 = CacheKey("q2")
    key3 = CacheKey("q3")
    
    # Miss
    assert await cache.get(key1) is None
    
    # Set & Get
    await cache.set(key1, "data1")
    assert await cache.get(key1) == "data1"
    
    # Update existing
    await cache.set(key1, "data1-new")
    assert await cache.get(key1) == "data1-new"
    
    # Evict LRU
    lru_cache = QueryResultCache(max_size=2, default_ttl=5)
    await lru_cache.set(key1, "data1")
    await lru_cache.set(key2, "data2")
    # key1 is LRU since key2 was added last. Adding key3 should evict key1.
    await lru_cache.set(key3, "data3")
    
    assert await lru_cache.get(key1) is None
    assert await lru_cache.get(key2) == "data2"
    assert await lru_cache.get(key3) == "data3"
    
    # Expired entry
    await cache.set(key1, "data1", ttl=1)
    with patch('time.time', return_value=time.time() + 2):
        assert await cache.get(key1) is None
        
    # Invalidate
    await cache.set(key2, "data2")
    await cache.invalidate(key2)
    assert await cache.get(key2) is None
    
    # Invalidate pattern
    await cache.set(CacheKey("typeA", {"id": 1}), "a1")
    await cache.set(CacheKey("typeA", {"id": 2}), "a2")
    await cache.set(CacheKey("typeB"), "b")
    
    count = await cache.invalidate_pattern("typeA")
    assert count == 2
    
    # Clear
    await cache.clear()
    assert len(cache._cache) == 0
    
    # Cleanup expired
    await cache.set(key1, "d1", ttl=1)
    await cache.set(key2, "d2", ttl=10)
    with patch('time.time', return_value=time.time() + 2):
        cleaned = await cache.cleanup_expired()
        assert cleaned == 1
        
    # Stats
    stats = cache.get_stats()
    assert stats["max_size"] == 10
    assert "hits" in stats

@pytest.mark.asyncio
async def test_cacheable_query_mixin():
    class TestLogger(CacheableQueryMixin):
        pass
        
    logger = TestLogger()
    
    # Check cache empty
    assert await logger.check_cache("q1", {"id": 5}) is None
    
    # Cache result
    await logger.cache_result("q1", "result-data", {"id": 5}, ttl=60)
    assert await logger.check_cache("q1", {"id": 5}) == "result-data"
    
    # Invalidate
    await logger.invalidate_cache("q1")
    assert await logger.check_cache("q1", {"id": 5}) is None
    
    # Stats
    stats = logger.get_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
