# =============================================================================
# OpenEnv Urban Planner — Dockerfile
# =============================================================================
# Builds from the OpenEnv base image and copies the environment code.
# Health check ensures the server is responding before marking healthy.
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

# install uv
RUN pip install uv

# copy everything
COPY . .

# install deps
RUN uv sync

# HF port
EXPOSE 7860

# run server
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]