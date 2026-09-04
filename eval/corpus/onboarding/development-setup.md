# Development Setup

## Prerequisites

Python 3.11 or later, Docker Desktop, and Git. Nothing else is required; if a
setup step needs something not on this list, the step is wrong.

## Getting started

    git clone <repository>
    cp .env.example .env
    docker compose up -d --build

This starts the API, PostgreSQL and Redis, applies database migrations, and
serves the console on port 8000.

## Running tests

    pytest -m "not integration"

Unit tests need no database, no API key and no network. Integration tests
require the compose stack and are selected with `-m integration`.

## Common problems

**Port 5432 already in use** — a local PostgreSQL install is bound to it. Stop
it or change the published port in the compose file.

**Embedding model downloads on every rebuild** — the model volume was removed.
It is a named volume specifically so the download survives rebuilds.

**Database connection errors inside the container** — the connection string is
pointing at localhost. Inside a container localhost is the container itself; use
the compose service name.
