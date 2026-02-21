from fastapi import Request


def get_request_headers(request: Request):
    """Dependency to extract headers from the request."""
    return dict(request.headers)
