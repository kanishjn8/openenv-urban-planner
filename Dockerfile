# =============================================================================
# OpenEnv Urban Planner — Dockerfile
# =============================================================================
# Builds a self-contained image of the environment server. Used for both:
#   - HuggingFace Spaces (port 7860 is mandatory for sdk: docker)
#   - Local docker development (same port for consistency)
#
# The image:
#   1. Installs uv and curl (curl is needed for HEALTHCHECK).
#   2. Copies the repo into /app and runs `uv sync` (which installs the
#      `openenv-urban-planner` package via hatch's force-include + dev-mode-dirs).
#   3. Starts uvicorn against `openenv_urban_planner.server.app:app` on port 7860.
#
# =============================================================================

FROM python:3.12-slim

# install uv + curl (curl needed for HEALTHCHECK)
RUN pip install --no-cache-dir uv && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Repo lives at /workspace/openenv_urban_planner so that dev-mode-dirs=[".."]
# puts /workspace on sys.path and `import openenv_urban_planner` resolves here.
WORKDIR /workspace/openenv_urban_planner

# copy everything (the build context is the repo root)
COPY . .

# install deps + project in editable mode
RUN uv sync

# HF Spaces requires port 7860
EXPOSE 7860

# Healthcheck — fails the container if the server doesn't respond
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# Start the server. Both forms work; we use `uv run server` because it
# matches the OpenEnv multi-mode deployment entry point.
CMD ["uv", "run", "server"]
