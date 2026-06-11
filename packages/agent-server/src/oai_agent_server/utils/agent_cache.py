"""Caching layer for agent information and logs.

Provides:
- Agent metadata caching (capabilities, config, status)
- Agent interaction logs caching with pagination
- Automatic cache invalidation
- Statistics tracking
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class CachedAgent:
    """Cached agent metadata."""
    agent_id: str
    agent_name: str
    capabilities: List[str]
    config: Dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime
    cached_at: datetime = field(default_factory=datetime.now)
    ttl: int = 300  # 5 minutes default
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.now() - self.cached_at).total_seconds()
        return age > self.ttl


@dataclass
class CachedAgentLogs:
    """Cached agent interaction logs."""
    agent_id: str
    logs: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    cached_at: datetime = field(default_factory=datetime.now)
    ttl: int = 60  # 1 minute for logs (more volatile)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.now() - self.cached_at).total_seconds()
        return age > self.ttl


class AgentInfoCache:
    """Cache for agent metadata."""
    
    def __init__(
        self,
        max_agents: int = 100,
        default_ttl: int = 300,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize agent info cache.
        
        Args:
            max_agents: Maximum agents to cache (default: 100)
            default_ttl: Default TTL in seconds (default: 300)
            logger: Logger instance
        """
        self.max_agents = max_agents
        self.default_ttl = default_ttl
        self.logger = logger or logging.getLogger(__name__)
        self._cache: Dict[str, CachedAgent] = {}
        self._access_order: List[str] = []  # For LRU eviction
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    async def get(self, agent_id: str) -> Optional[CachedAgent]:
        """Get cached agent info.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            CachedAgent or None if not found or expired
        """
        async with self._lock:
            if agent_id not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[agent_id]
            if entry.is_expired():
                del self._cache[agent_id]
                self._access_order.remove(agent_id)
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            # Move to end for LRU
            self._access_order.remove(agent_id)
            self._access_order.append(agent_id)
            
            return entry
    
    async def set(
        self,
        agent_id: str,
        agent_info: CachedAgent,
    ) -> None:
        """Cache agent info.
        
        Args:
            agent_id: Agent ID
            agent_info: Agent info to cache
        """
        async with self._lock:
            agent_info.ttl = self.default_ttl
            
            if agent_id in self._cache:
                self._access_order.remove(agent_id)
            elif len(self._cache) >= self.max_agents:
                # Evict least recently used
                lru_id = self._access_order.pop(0)
                del self._cache[lru_id]
                self._stats["evictions"] += 1
                self.logger.debug(f"Evicted agent {lru_id} from cache (LRU)")
            
            self._cache[agent_id] = agent_info
            self._access_order.append(agent_id)
            self.logger.debug(f"Cached agent info: {agent_id}")
    
    async def invalidate(self, agent_id: str) -> None:
        """Invalidate cache entry for agent.
        
        Args:
            agent_id: Agent ID
        """
        async with self._lock:
            if agent_id in self._cache:
                del self._cache[agent_id]
                self._access_order.remove(agent_id)
                self.logger.debug(f"Invalidated agent cache: {agent_id}")
    
    async def invalidate_all(self) -> None:
        """Invalidate all cached agents."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self.logger.info("Invalidated all agent cache")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Stats dict with hits, misses, evictions, hit_rate
        """
        async with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                (self._stats["hits"] / total * 100)
                if total > 0
                else 0
            )
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "total_requests": total,
                "hit_rate_percent": round(hit_rate, 2),
                "cached_agents": len(self._cache),
            }


class AgentLogsCache:
    """Cache for agent interaction logs."""
    
    def __init__(
        self,
        max_pages: int = 50,
        default_ttl: int = 60,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize agent logs cache.
        
        Args:
            max_pages: Maximum page results to cache (default: 50)
            default_ttl: Default TTL in seconds (default: 60)
            logger: Logger instance
        """
        self.max_pages = max_pages
        self.default_ttl = default_ttl
        self.logger = logger or logging.getLogger(__name__)
        self._cache: Dict[str, List[CachedAgentLogs]] = {}  # agent_id -> list of pages
        self._access_order: List[tuple] = []  # (agent_id, page) for LRU
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    def _make_cache_key(self, agent_id: str, page: int) -> str:
        """Make cache key from agent_id and page."""
        return f"{agent_id}:{page}"
    
    async def get(
        self,
        agent_id: str,
        page: int = 1,
    ) -> Optional[CachedAgentLogs]:
        """Get cached logs for agent.
        
        Args:
            agent_id: Agent ID
            page: Page number (default: 1)
            
        Returns:
            CachedAgentLogs or None
        """
        async with self._lock:
            if agent_id not in self._cache:
                self._stats["misses"] += 1
                return None
            
            pages = self._cache[agent_id]
            page_idx = page - 1  # Convert to 0-indexed
            
            if page_idx >= len(pages):
                self._stats["misses"] += 1
                return None
            
            entry = pages[page_idx]
            if entry.is_expired():
                del pages[page_idx]
                if not pages:
                    del self._cache[agent_id]
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            # Move to end for LRU
            cache_key = self._make_cache_key(agent_id, page)
            if (agent_id, page) in self._access_order:
                self._access_order.remove((agent_id, page))
            self._access_order.append((agent_id, page))
            
            return entry
    
    async def set(
        self,
        agent_id: str,
        logs_info: CachedAgentLogs,
    ) -> None:
        """Cache logs for agent.
        
        Args:
            agent_id: Agent ID
            logs_info: Logs info to cache
        """
        async with self._lock:
            logs_info.ttl = self.default_ttl
            page = logs_info.page
            
            if agent_id not in self._cache:
                self._cache[agent_id] = []
            
            pages = self._cache[agent_id]
            page_idx = page - 1
            
            # Expand list if needed
            while len(pages) <= page_idx:
                pages.append(None)
            
            cache_key = self._make_cache_key(agent_id, page)
            if (agent_id, page) in self._access_order:
                self._access_order.remove((agent_id, page))
            
            # Evict if at capacity
            if len(self._access_order) >= self.max_pages:
                lru_agent, lru_page = self._access_order.pop(0)
                if lru_agent in self._cache:
                    lru_idx = lru_page - 1
                    if lru_idx < len(self._cache[lru_agent]):
                        del self._cache[lru_agent][lru_idx]
                        if not self._cache[lru_agent]:
                            del self._cache[lru_agent]
                self._stats["evictions"] += 1
                self.logger.debug(f"Evicted logs {lru_agent}:{lru_page} (LRU)")
            
            pages[page_idx] = logs_info
            self._access_order.append((agent_id, page))
            self.logger.debug(f"Cached agent logs: {agent_id} page {page}")
    
    async def invalidate(self, agent_id: str) -> None:
        """Invalidate all cached logs for agent.
        
        Args:
            agent_id: Agent ID
        """
        async with self._lock:
            if agent_id in self._cache:
                pages = self._cache[agent_id]
                for page in range(1, len(pages) + 1):
                    key = (agent_id, page)
                    if key in self._access_order:
                        self._access_order.remove(key)
                
                del self._cache[agent_id]
                self.logger.debug(f"Invalidated all logs for agent: {agent_id}")
    
    async def invalidate_all(self) -> None:
        """Invalidate all cached logs."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self.logger.info("Invalidated all logs cache")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Stats dict with hits, misses, etc.
        """
        async with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                (self._stats["hits"] / total * 100)
                if total > 0
                else 0
            )
            
            total_pages = len(self._access_order)
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "total_requests": total,
                "hit_rate_percent": round(hit_rate, 2),
                "cached_pages": total_pages,
            }


class AgentCacheManager:
    """Manager for both agent info and logs caches."""
    
    def __init__(
        self,
        info_max_agents: int = 100,
        info_ttl: int = 300,
        logs_max_pages: int = 50,
        logs_ttl: int = 60,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize cache manager.
        
        Args:
            info_max_agents: Max agents in info cache
            info_ttl: Agent info TTL
            logs_max_pages: Max log pages in cache
            logs_ttl: Logs TTL
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.info = AgentInfoCache(info_max_agents, info_ttl, self.logger)
        self.logs = AgentLogsCache(logs_max_pages, logs_ttl, self.logger)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get stats for both caches.
        
        Returns:
            Dict with info and logs cache stats
        """
        return {
            "agent_info": await self.info.get_stats(),
            "agent_logs": await self.logs.get_stats(),
        }
    
    async def invalidate_agent(self, agent_id: str) -> None:
        """Invalidate all cache for specific agent.
        
        Args:
            agent_id: Agent ID
        """
        await self.info.invalidate(agent_id)
        await self.logs.invalidate(agent_id)
        self.logger.info(f"Invalidated all cache for agent: {agent_id}")
    
    async def clear_all(self) -> None:
        """Clear all caches."""
        await self.info.invalidate_all()
        await self.logs.invalidate_all()
        self.logger.info("Cleared all caches")
