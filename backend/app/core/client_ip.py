"""Client IP behind the Istio sidecar and ingress."""
from __future__ import annotations

import ipaddress
from functools import lru_cache

from fastapi import Request

from app.core.settings import settings


@lru_cache(maxsize=1)
def _trusted_networks(cidrs: tuple[str, ...]) -> tuple[ipaddress._BaseNetwork, ...]:
    return tuple(ipaddress.ip_network(c, strict=False) for c in cidrs)


def is_trusted_proxy(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in net for net in _trusted_networks(tuple(settings.trusted_proxy_cidrs)))


def client_ip(request: Request) -> str:
    """The first address, reading X-Forwarded-For from the right, that is not a
    trusted proxy. Entries a client adds on the left are never reached while a
    real address sits to their right, so they cannot spoof the result."""
    peer = request.client.host if request.client else ""
    if not is_trusted_proxy(peer):
        return peer or "unknown"
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    for hop in reversed(hops):
        if not is_trusted_proxy(hop):
            return hop
    return hops[0] if hops else (peer or "unknown")
