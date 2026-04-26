# =============================================================================
# OpenEnv Urban Planner — Dockerfile
# =============================================================================
# Builds a self-contained image of the environment server.  Used for both:
#   - HuggingFace Spaces (port 7860 is mandatory for sdk: docker)
#   - Local docker development (same port for consistency)
#
# The image:
#   1. Installs uv and curl (curl is needed for HEALTHCHECK).
#   2. Copies the repo into /app and runs `uv sync`.
#   3. Starts uvicorn against `server.app:app` on port 7860.
#
# Note on imports:
#   The server entry-point (`server/app.py`) uses bare imports (`from models
#   import ...`).  This works because /app is on sys.path inside the container,
#   so `models.py`, `client.py`, and the `server/` package are all directly
#   resolvable.  Do NOT change the CMD to `openenv_urban_planner.server.app:app`
#   without first restructuring the source layout.
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

# install uv + curl (curl needed for HEALTHCHECK)
RUN pip install --no-cache-dir uv && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# copy everything
COPY . .

# install deps
RUN uv sync

# HF Spaces requires port 7860
EXPOSE 7860

# Healthcheck — fails the container if the server doesn't respond
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# run server
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
