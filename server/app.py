# =============================================================================
# OpenEnv Urban Planner — FastAPI App Entry Point
# =============================================================================
# Creates the OpenEnv-compliant FastAPI application using `create_app`.
# This is the file referenced in openenv.yaml and the Dockerfile CMD.
#
# Start locally with:
#   uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
# =============================================================================

from openenv.core.env_server import create_app

from models import UrbanPlannerAction, UrbanPlannerObservation
from server.urban_planner_environment import UrbanPlannerEnvironment

# Create the OpenEnv-compliant FastAPI app.
# This registers the /reset, /step, /state, /close HTTP endpoints and
# wires them to our UrbanPlannerEnvironment.
app = create_app(
    UrbanPlannerEnvironment,
    UrbanPlannerAction,
    UrbanPlannerObservation,
    env_name="openenv-urban-planner",
)
