# =============================================================================
# OpenEnv Urban Planner — Dockerfile
# =============================================================================
# Builds from the OpenEnv base image and copies the environment code.
# Health check ensures the server is responding before marking healthy.
# =============================================================================

FROM openenv-base:latest  

COPY . /app

WORKDIR /app

RUN pip install -r server/requirements.txt

EXPOSE 7860

HEALTHCHECK CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]