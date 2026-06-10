"""Query result caching for the database logger.

Implements TTL-based caching for frequently accessed database queries
to reduce database load and improve response times.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict


class CacheKey:
    """Represents a cache key for database queries."""
    
    def __init__(self, query_type: str, filters: Optional[Dict[str, Any]] = None):
        """Initialize cache key.
        
        Args:
            query_type: Type of query (e.g., 'interaction_by_id', 'sessions_for_user')
            filters: Query filters/parameters
        """
        self.query_type = query_type
        self.filters = filters or {}
    
    def __hash__(self) -> int:
        """Generate hash for cache key."""
        filters_json = json.dumps(self.filters, sort_keys=True, default=str)
        return hash((self.query_type, filters_json))
    
    def __eq__(self, other: Any) -> bool:
        """Check equality of cache keys."""
        if not isinstance(other, CacheKey):
            return False
        return self.query_type == other.query_type and self.filters == other.filters
    
    def __repr__(self) -> str:
        return f"CacheKey({self.query_type}, {self.filters})"


class CacheEntry:
    """Represents a cached query result."""
    
    def __init__(self, data: Any, ttl: int):
        """Initialize cache entry.
        
        Args:
            data: Cached data
            ttl: Time to live in seconds
        """
        self.data = data
        self.created_at = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl
    
    def __repr__(self) -> str:
        age = time.time() - self.created_at
        return f"CacheEntry(age={age:.1f}s, ttl={self.ttl}s, expired={self.is_expired()})"


class QueryResultCache:
    """LRU cache for database query results with TTL support.
    
    Caches frequently accessed queries with configurable TTL.
    Automatically evicts expired entries and maintains LRU ordering.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """Initialize query cache.
        
        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if found and not expired, None otherwise
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.data
    
    async def set(self, key: CacheKey, data: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        async with self._lock:
            ttl = ttl or self.default_ttl
            entry = CacheEntry(data, ttl)
            
            if key in self._cache:
                # Update existing entry
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = entry
                
                # Evict LRU entry if cache is full
                if len(self._cache) > self.max_size:
                    evicted_key, _ = self._cache.popitem(last=False)
                    self._evictions += 1
    
    async def invalidate(self, key: CacheKey) -> None:
        """Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    async def invalidate_pattern(self, query_type: str) -> int:
        """Invalidate all entries matching a query type.
        
        Args:
            query_type: Query type pattern to invalidate
            
        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            invalidated = 0
            keys_to_delete = [
                key for key in self._cache.keys()
                if key.query_type == query_type
            ]
            for key in keys_to_delete:
                del self._cache[key]
                invalidated += 1
            return invalidated
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    async def cleanup_expired(self) -> int:
        """Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            keys_to_delete = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "evictions": self._evictions,
        }


class CacheableQueryMixin:
    """Mixin to add caching support to database logger.
    
    Subclasses should call cache_result() to store query results
    and check_cache() before executing queries.
    """
    
    def __init__(self):
        self._query_cache = QueryResultCache()
    
    async def check_cache(self, query_type: str, filters: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Check if result is cached.
        
        Args:
            query_type: Type of query
            filters: Query filters
            
        Returns:
            Cached result if found, None otherwise
        """
        key = CacheKey(query_type, filters)
        return await self._query_cache.get(key)
    
    async def cache_result(self, query_type: str, result: Any, 
                          filters: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> None:
        """Cache a query result.
        
        Args:
            query_type: Type of query
            result: Result to cache
            filters: Query filters (used for cache key)
            ttl: Time-to-live in seconds
        """
        key = CacheKey(query_type, filters)
        await self._query_cache.set(key, result, ttl)
    
    async def invalidate_cache(self, query_type: str) -> None:
        """Invalidate cache for a query type.
        
        Args:
            query_type: Query type to invalidate
        """
        await self._query_cache.invalidate_pattern(query_type)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return self._query_cache.get_stats()
