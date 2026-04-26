# =============================================================================
# OpenEnv Urban Planner — MCP Tool Client
# =============================================================================
# Thin client wrapper that connects to the running Urban Planner environment
# server (local or remote HF Space) and exposes a Pythonic interface for
# training scripts and notebooks.
#
# Usage:
#   from openenv_urban_planner import UrbanPlannerEnv
#   env = UrbanPlannerEnv(base_url="http://localhost:8000").sync()
#   obs = env.reset()
#   obs = env.step({"tool_name": "get_city_state", "arguments": {}})
#
# This class inherits from openenv's MCPToolClient which handles:
#   - Session management (start/stop)
#   - Tool call serialization over HTTP
#   - Observation deserialization
# =============================================================================

from __future__ import annotations

from openenv.core.mcp_client import MCPToolClient

# Relative import so the client works whether the package is loaded as
# `openenv_urban_planner.client` (library form) or via direct project-root
# sys.path injection (training scripts).
from .models import (
    UrbanPlannerAction,
    UrbanPlannerObservation,
)


class UrbanPlannerEnv(MCPToolClient):
    """
    Client for the OpenEnv Urban Planner environment.

    Connects to the environment server and provides `reset()` / `step()`
    methods.  Each `step()` call sends a tool-call action and returns
    the resulting observation (including reward and done flag).

    Attributes:
        action_model:      Pydantic model class for serializing actions.
        observation_model: Pydantic model class for deserializing observations.
    """

    # -----------------------------------------------------------------
    # Class-level type annotations — tell the base class which Pydantic
    # models to use for (de)serialization of actions and observations.
    # -----------------------------------------------------------------
    action_model = UrbanPlannerAction
    observation_model = UrbanPlannerObservation

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """
        Initialize the Urban Planner client.

        Args:
            base_url: URL of the running environment server.
                      Defaults to localhost:8000 for local development.
        """
        super().__init__(base_url=base_url)
