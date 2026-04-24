# =============================================================================
# OpenEnv Urban Planner — FastAPI App Entry Point
# =============================================================================
# Creates the OpenEnv-compliant FastAPI application using `create_app`.
# This is the file referenced in openenv.yaml and the Dockerfile CMD.
#
# Start locally with:
#   uv run uvicorn openenv_urban_planner.server.app:app --host 0.0.0.0 --port 8000
# =============================================================================

from openenv.core.env_server import create_app

from openenv_urban_planner.models import UrbanPlannerAction, UrbanPlannerObservation
from openenv_urban_planner.server.urban_planner_environment import UrbanPlannerEnvironment

# Create the OpenEnv-compliant FastAPI app.
# This registers the /reset, /step, /state, /close HTTP endpoints and
# wires them to our UrbanPlannerEnvironment.
app = create_app(
    UrbanPlannerEnvironment,
    UrbanPlannerAction,
    UrbanPlannerObservation,
    env_name="openenv-urban-planner",
)
