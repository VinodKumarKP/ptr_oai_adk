"""API versioning support for endpoints.

Enables endpoints to be versioned:
- /v1/chat, /v2/chat, /v3/chat
- Default to latest version if not specified
- Version negotiation and deprecation
"""

import logging
from typing import Callable, Dict, List, Optional, Any
from enum import Enum
from fastapi import APIRouter, Request


class APIVersion(str, Enum):
    """API versions."""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class VersionedEndpoint:
    """Represents a versioned endpoint."""
    
    def __init__(
        self,
        path: str,
        versions: Dict[APIVersion, Callable],
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize versioned endpoint.
        
        Args:
            path: Endpoint path (without version prefix)
            versions: Dict mapping APIVersion to handler function
            logger: Logger instance
        """
        self.path = path
        self.versions = versions
        self.logger = logger or logging.getLogger(__name__)
        self.latest_version = max(versions.keys(), key=lambda v: int(v.value[1:]))
    
    async def execute(
        self,
        version: Optional[str],
        *args,
        **kwargs
    ) -> Any:
        """Execute versioned endpoint.
        
        Args:
            version: API version (e.g., "v1") or None for latest
            *args: Positional arguments to pass to handler
            **kwargs: Keyword arguments to pass to handler
            
        Returns:
            Handler result
            
        Raises:
            ValueError: If version not supported
        """
        if version is None:
            version = self.latest_version.value
        
        # Normalize version (remove / prefix if present)
        version = version.lstrip("/")
        
        try:
            api_version = APIVersion(version)
        except ValueError:
            raise ValueError(f"Unsupported API version: {version}")
        
        if api_version not in self.versions:
            supported = ", ".join(v.value for v in self.versions.keys())
            raise ValueError(
                f"Version {version} not available for {self.path}. "
                f"Supported versions: {supported}"
            )
        
        self.logger.debug(f"Executing {self.path} with version {version}")
        
        handler = self.versions[api_version]
        return await handler(*args, **kwargs) if callable(handler) else handler


class APIVersionRegistry:
    """Registry for managing versioned endpoints."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize version registry.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._endpoints: Dict[str, VersionedEndpoint] = {}
    
    def register(
        self,
        path: str,
        versions: Dict[APIVersion, Callable],
    ) -> None:
        """Register versioned endpoint.
        
        Args:
            path: Endpoint path
            versions: Dict mapping APIVersion to handler
        """
        endpoint = VersionedEndpoint(path, versions, self.logger)
        self._endpoints[path] = endpoint
        
        supported_versions = ", ".join(v.value for v in versions.keys())
        latest = endpoint.latest_version.value
        
        self.logger.info(
            f"Registered versioned endpoint: {path} "
            f"(versions: {supported_versions}, latest: {latest})"
        )
    
    async def execute(
        self,
        path: str,
        version: Optional[str] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute versioned endpoint.
        
        Args:
            path: Endpoint path
            version: API version (None for latest)
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Handler result
            
        Raises:
            ValueError: If endpoint or version not found
        """
        if path not in self._endpoints:
            raise ValueError(f"No versioned endpoint registered for: {path}")
        
        endpoint = self._endpoints[path]
        return await endpoint.execute(version, *args, **kwargs)
    
    def get_endpoint_info(self, path: str) -> Dict[str, Any]:
        """Get information about versioned endpoint.
        
        Args:
            path: Endpoint path
            
        Returns:
            Dict with versions, latest version
            
        Raises:
            ValueError: If endpoint not found
        """
        if path not in self._endpoints:
            raise ValueError(f"No versioned endpoint registered for: {path}")
        
        endpoint = self._endpoints[path]
        
        return {
            "path": endpoint.path,
            "versions": sorted(
                (v.value for v in endpoint.versions.keys()),
                key=lambda x: int(x[1:])
            ),
            "latest_version": endpoint.latest_version.value,
        }
    
    def get_all_endpoints_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all versioned endpoints.
        
        Returns:
            Dict mapping path to endpoint info
        """
        return {
            path: self.get_endpoint_info(path)
            for path in self._endpoints.keys()
        }


class VersionedRouter:
    """Helper for creating versioned routes with FastAPI."""
    
    def __init__(
        self,
        prefix: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize versioned router.
        
        Args:
            prefix: Route prefix (e.g., "/api")
            logger: Logger instance
        """
        self.prefix = prefix
        self.logger = logger or logging.getLogger(__name__)
        self.registry = APIVersionRegistry(logger)
    
    def create_versioned_path(
        self,
        path: str,
        version: str,
    ) -> str:
        """Create full versioned path.
        
        Args:
            path: Endpoint path
            version: API version
            
        Returns:
            Full versioned path (e.g., "/api/v1/chat")
        """
        return f"{self.prefix}/{version}{path}"
    
    def add_versioned_endpoint(
        self,
        router: APIRouter,
        path: str,
        methods: List[str],
        versions: Dict[APIVersion, Callable],
        tags: Optional[List[str]] = None,
    ) -> None:
        """Add versioned endpoint to FastAPI router.
        
        Args:
            router: FastAPI APIRouter instance
            path: Endpoint path (without version)
            methods: HTTP methods (e.g., ["GET", "POST"])
            versions: Dict mapping APIVersion to handler
            tags: OpenAPI tags
        """
        self.registry.register(path, versions)
        
        # Create wrapper that delegates to versioned handler
        async def versioned_handler(
            request: Request,
            **kwargs
        ):
            # Extract version from URL path
            version = self._extract_version_from_path(request.url.path)
            
            # Get the appropriate handler
            endpoint = self.registry._endpoints[path]
            return await endpoint.execute(version, request=request, **kwargs)
        
        # Register for each version
        for api_version in versions.keys():
            versioned_path = self.create_versioned_path(path, api_version.value)
            
            # Register the endpoint
            for method in methods:
                route_decorator = getattr(router, method.lower())
                route_decorator(
                    versioned_path,
                    tags=tags or [],
                )(versioned_handler)
            
            self.logger.debug(
                f"Registered route: {versioned_path} [{', '.join(methods)}]"
            )
        
        # Register unversioned route (defaults to latest)
        unversioned_path = f"{self.prefix}{path}"
        
        async def unversioned_handler(request: Request, **kwargs):
            endpoint = self.registry._endpoints[path]
            return await endpoint.execute(None, request=request, **kwargs)
        
        for method in methods:
            route_decorator = getattr(router, method.lower())
            route_decorator(
                unversioned_path,
                tags=tags or [],
            )(unversioned_handler)
        
        self.logger.debug(
            f"Registered unversioned route: {unversioned_path} "
            f"(defaults to latest) [{', '.join(methods)}]"
        )
    
    def _extract_version_from_path(self, path: str) -> Optional[str]:
        """Extract version from URL path.
        
        Args:
            path: Full URL path
            
        Returns:
            Version string (e.g., "v1") or None
        """
        # Path format: /api/v1/endpoint or /api/endpoint
        parts = path.strip("/").split("/")
        
        for i, part in enumerate(parts):
            if part.startswith("v") and part[1:].isdigit():
                return part
        
        return None


def create_versioned_router(
    prefix: str = "/api",
    logger: Optional[logging.Logger] = None,
) -> VersionedRouter:
    """Factory function to create versioned router.
    
    Args:
        prefix: Route prefix
        logger: Logger instance
        
    Returns:
        VersionedRouter instance
    """
    return VersionedRouter(prefix, logger)
