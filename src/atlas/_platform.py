"""Platform-specific runtime setup.

Windows: Python 3.8+ defaults to ProactorEventLoop, and psycopg 3's async mode
cannot run on it -- it needs SelectorEventLoop. Without this, every async
database call fails at connection time with:

    InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async
    mode.

This is applied once at package import rather than at each entry point, because
there are four of them (the CLI, uvicorn, pytest-asyncio, and ad-hoc scripts) and
missing one produces a failure that looks like a database problem rather than an
event-loop problem.

Trade-off: SelectorEventLoop on Windows caps out around 512 concurrent sockets
and does not support asyncio subprocesses. Neither constrains Atlas, which opens
a bounded connection pool and spawns no subprocesses. On Linux -- where this will
actually be deployed -- nothing here applies and the default loop is used.
"""

from __future__ import annotations

import asyncio
import sys


def configure_event_loop() -> None:
    """Install a psycopg-compatible event loop policy on Windows. No-op elsewhere."""
    if sys.platform != "win32":
        return

    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is None:  # pragma: no cover - non-Windows or a future removal
        return

    # Do not stomp on a policy someone has deliberately installed.
    if isinstance(asyncio.get_event_loop_policy(), policy):
        return

    asyncio.set_event_loop_policy(policy())
