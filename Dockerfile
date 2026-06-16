# ==============================================================================
# AllianceAI — container image
#
# NOTE ON BASE IMAGE: we use Debian "slim", NOT Alpine, even though the host
# server is Alpine. The scientific stack (numpy/scipy/statsmodels/prophet) ships
# prebuilt manylinux wheels for glibc but NOT for Alpine's musl libc — on Alpine
# everything would compile from source (slow, fragile). A Debian container runs
# perfectly on an Alpine Docker host; the host OS and the image OS are decoupled.
# ==============================================================================
FROM python:3.12-slim AS base

# System libs a few wheels link against at runtime (duckdb, plotly, prophet).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- Dependency layer (cached unless pyproject changes) ---
COPY pyproject.toml ./
COPY allianceai/__init__.py ./allianceai/__init__.py
RUN pip install --upgrade pip && \
    pip install -e . && \
    # Extras: Prophet for the better forecaster, JupyterLab for the UI.
    # If the Prophet wheel ever fails on your arch, drop it — the code falls
    # back to Holt-Winters automatically.
    pip install jupyterlab prophet

# --- Application code ---
COPY . .
RUN pip install -e .

# Writable data dirs (also mounted as volumes in compose for persistence).
RUN mkdir -p /app/data /app/reports /app/logs /app/notebooks

EXPOSE 8888

# Default: launch JupyterLab. Token is read from JUPYTER_TOKEN at runtime.
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/app --IdentityProvider.token=${JUPYTER_TOKEN:-allianceai}"]
