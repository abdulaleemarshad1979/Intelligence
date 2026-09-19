"""API Package: discovery, tracking, and alert dispatch routers."""
from app.api.routes_discovery import router as discovery_router
from app.api.routes_tracking import router as tracking_router
from app.api.routes_alerts import router as alerts_router

__all__ = [
    "discovery_router",
    "tracking_router",
    "alerts_router",
]
