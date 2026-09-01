# Atlas API image.
#
# Local development image, not a production one. Two choices worth explaining:
#
#  * Editable install (`pip install -e .`) with the source kept at /app. The
#    migration runner locates migrations relative to the package
#    (src/atlas/db/migrate.py -> ../../../migrations), which resolves correctly
#    for a source layout but not for a package copied into site-packages. An
#    editable install keeps that resolution honest without adding a config knob.
#    Phase 7 revisits this when a real deployment image is needed.
#
#  * The embedding model is NOT baked in. It downloads on first use into
#    /models, which compose mounts as a named volume, so it survives rebuilds
#    without adding ~67MB to every image layer.

FROM python:3.11-slim

# onnxruntime -- fastembed's inference backend -- links against libgomp at
# import time. Without it, every embedding call fails with a missing shared
# object rather than anything that names the real cause.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # onnxruntime is a large wheel and the default 15s socket timeout drops it
    # on a slow link, failing the build with a stack trace that looks like a
    # dependency problem rather than a network one.
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    ATLAS_MODEL_CACHE_DIR=/models

WORKDIR /app

# Dependency metadata first: as long as pyproject.toml is unchanged, editing
# source does not force pip to resolve and download everything again.
COPY pyproject.toml README.md ./
RUN mkdir -p src/atlas \
    && printf '__version__ = "0.0.0"\n' > src/atlas/__init__.py \
    && pip install --no-cache-dir -e . \
    && rm -rf src

COPY src ./src
COPY migrations ./migrations
COPY eval ./eval

# Re-run with the real package present so the editable install points at it.
RUN pip install --no-cache-dir --no-deps -e .

RUN mkdir -p /models

EXPOSE 8000

# No curl in the slim image; use the interpreter that is definitely present.
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=6 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "atlas.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
