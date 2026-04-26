# =============================================================================
# OpenEnv Urban Planner — Package Initializer
# =============================================================================
# Top-level package for the OpenEnv Urban Planner environment.
# Exposes the client class for external consumers (training scripts, notebooks).
#
# We use a *relative* import so that this package works both:
#   - as an installed library (`import openenv_urban_planner`), and
#   - when the project root sits on sys.path (training scripts, Docker).
# =============================================================================

from .client import UrbanPlannerEnv  # noqa: F401

__version__ = "0.1.0"
__all__ = ["UrbanPlannerEnv", "__version__"]
