# =============================================================================
# OpenEnv Urban Planner — Dockerfile
# =============================================================================
# Builds from the OpenEnv base image and copies the environment code.
# Health check ensures the server is responding before marking healthy.
# =============================================================================

ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install Python dependencies
COPY openenv_urban_planner/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy OpenEnv core and environment source
COPY src/openenv/core/ /app/src/openenv/core/
COPY openenv_urban_planner/ /app/openenv_urban_planner/

# Health check — ensure the server responds within 3 seconds
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the uvicorn server
CMD ["uvicorn", "openenv_urban_planner.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
