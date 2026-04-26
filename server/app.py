# =============================================================================
# OpenEnv Urban Planner — FastAPI App Entry Point
# =============================================================================
# Creates the OpenEnv-compliant FastAPI application using `create_app`.
# This is the file referenced in openenv.yaml and the Dockerfile CMD.
#
# Start locally with:
#   uv run uvicorn openenv_urban_planner.server.app:app --host 0.0.0.0 --port 7860
#   uv run server   # uses [project.scripts] entry point (see pyproject.toml)
# =============================================================================

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from ..models import UrbanPlannerAction, UrbanPlannerObservation
from .urban_planner_environment import UrbanPlannerEnvironment

# Create the OpenEnv-compliant FastAPI app.
# This registers the /reset, /step, /state, /close HTTP endpoints and
# wires them to our UrbanPlannerEnvironment.
app = create_app(
    UrbanPlannerEnvironment,
    UrbanPlannerAction,
    UrbanPlannerObservation,
    env_name="openenv-urban-planner",
)


def main() -> None:
    """CLI entry point for `uv run server` (OpenEnv multi-mode deployment)."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT") or os.environ.get("OPENENV_PORT") or "7860")
    uvicorn.run(
        "openenv_urban_planner.server.app:app",
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
