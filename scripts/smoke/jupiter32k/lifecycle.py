"""Bound the queue wait separately from the allocation and proxy lifetime."""

import math
import ipaddress

QUEUE_SECONDS = 8 * 3600
ALLOCATION_SECONDS = 30 * 60
PROXY_SECONDS = 9 * 3600
CLEANUP_SECONDS = 10 * 60


def select_bind(addresses):
    internal = ipaddress.ip_network("10.128.0.0/16")
    candidates = {
        str(ipaddress.ip_address(a))
        for a in addresses.split()
        if ipaddress.ip_address(a) in internal
    }
    if len(candidates) != 1:
        raise RuntimeError("Cannot identify unique internal login address")
    return candidates.pop()


def proxy_window(now):
    if not math.isfinite(now):
        raise ValueError("Invalid clock")
    return {
        "created_epoch": now,
        "queue_deadline_epoch": now + QUEUE_SECONDS,
        "expires_epoch": now + PROXY_SECONDS,
    }


def validate_lifetime(ready, now):
    times = [
        ready[k] for k in ("created_epoch", "queue_deadline_epoch", "expires_epoch")
    ]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in [now, *times]):
        raise ValueError("Invalid proxy timing")
    created, deadline, expires = times
    if deadline != created + QUEUE_SECONDS or expires != created + PROXY_SECONDS:
        raise ValueError("Proxy timing does not match approved bounds")
    if now < created or now >= deadline:
        raise RuntimeError("Queue allowance exhausted or clock moved backwards")
    if expires - now < ALLOCATION_SECONDS + CLEANUP_SECONDS:
        raise RuntimeError("Proxy cannot cover execution and cleanup")
