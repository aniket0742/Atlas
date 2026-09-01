"""Atlas - AI knowledge and retrieval platform."""

from atlas._platform import configure_event_loop

# Applied at import so every entry point (CLI, uvicorn, pytest, scripts) gets a
# psycopg-compatible event loop on Windows. See atlas/_platform.py.
configure_event_loop()

__version__ = "0.1.0"

__all__ = ["__version__", "configure_event_loop"]
