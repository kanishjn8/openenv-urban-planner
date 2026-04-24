# =============================================================================
# OpenEnv Urban Planner — Pydantic Models
# =============================================================================
# All data models for the environment: zone cells, actions, observations,
# episode configuration, and full server-side state.
#
# Design decisions:
#   - Every model inherits from pydantic.BaseModel with explicit field types.
#   - ZoneCell is a BaseModel (not a dataclass) so it serializes naturally
#     into observation JSON.
#   - Action is a discriminated union via `tool_name` — each MCP tool call
#     maps to one Action instance with the appropriate arguments.
#   - Observation follows the OpenEnv schema: includes tool_result, reward,
#     done flag, and rubric breakdown.
#   - State holds the full 16×16 grid (server-side only; never leaked to the
#     agent via observations thanks to fog-of-war filtering).
# =============================================================================

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Zone Cell — represents one cell in the 16×16 city grid
# =============================================================================

class ZoneCell(BaseModel):
    """
    A single cell in the city grid.

    Attributes:
        zone_type: The land-use designation for this cell.
                   One of: empty, residential, commercial, industrial, green, transit.
        density:   Development intensity (0=empty, 1=low, 2=medium, 3=high).
        infrastructure: List of built infrastructure items in this cell.
                        Valid items: road, metro, hospital, school, flood_barrier.
        population:  Number of residents occupying this cell.
                     Derived from zone_type × density during simulation.
        congestion:  Traffic congestion index (0.0 = free-flowing, 1.0 = gridlock).
                     Recomputed each season from the road network.
        flood_risk:  Probability of flood damage (0.0 = safe, 1.0 = certain).
                     Driven by elevation map + drainage infrastructure.
        school_load: Ratio of students in catchment area to available school capacity.
                     Values > 1.0 mean overcrowding; > 1.3 triggers protests.
        visible:     Whether this cell is visible to the agent (fog-of-war).
                     Hidden cells are excluded from observation payloads.
    """
    zone_type: str = Field(
        default="empty",
        description="Land-use type: empty | residential | commercial | industrial | green | transit",
    )
    density: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Development density level (0=empty, 1=low, 2=medium, 3=high)",
    )
    infrastructure: list[str] = Field(
        default_factory=list,
        description="Infrastructure items present: road, metro, hospital, school, flood_barrier",
    )
    population: int = Field(
        default=0,
        ge=0,
        description="Current population count in this cell",
    )
    congestion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Traffic congestion index (0.0–1.0)",
    )
    flood_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Flood risk probability (0.0–1.0)",
    )
    school_load: float = Field(
        default=0.0,
        ge=0.0,
        description="Student-to-capacity ratio in school catchment",
    )
    visible: bool = Field(
        default=True,
        description="Whether this cell is visible to the agent (fog-of-war)",
    )


# =============================================================================
# Planning Entry — server-maintained action/consequence history
# =============================================================================

class PlanningEntry(BaseModel):
    """
    A single entry in the server-maintained planning log.

    Injected into every observation so the agent has persistent memory of
    its decisions without wasting context on a tool call.  The server trims
    the log to the last 8 entries.

    Attributes:
        season:          Season index when this action was taken.
        action_summary:  Human-readable summary (e.g. "placed metro at (4,7)").
        consequence:     Observed result (e.g. "congestion in zone 3 dropped 0.3").
        reward_delta:    Change in reward from the previous step.
    """
    season: int = Field(
        default=0,
        description="Season when this action occurred",
    )
    action_summary: str = Field(
        default="",
        description="Human-readable action description",
    )
    consequence: str = Field(
        default="",
        description="Observed consequence of the action",
    )
    reward_delta: float = Field(
        default=0.0,
        description="Reward change from the previous step",
    )


# =============================================================================
# Collapse Report — generated on terminal city collapse
# =============================================================================

class CollapseReport(BaseModel):
    """
    Diagnostic report generated when a city collapses (population < 20%
    of start, or budget < 0).  Embedded in the final observation and
    written to assets/collapse_cases/episode_{n}.json.

    Attributes:
        trigger:              The proximate cause of collapse.
        chain:                Sequence of cascade events leading to collapse.
        season_of_no_return:  First season where collapse became inevitable.
        agent_mistake:        The key agent error that initiated the chain.
    """
    trigger: str = Field(
        default="",
        description="Proximate cause of the collapse",
    )
    chain: list[str] = Field(
        default_factory=list,
        description="Cascade chain leading to collapse",
    )
    season_of_no_return: int = Field(
        default=0,
        description="First season where collapse became inevitable",
    )
    agent_mistake: str = Field(
        default="",
        description="Key agent error that started the chain",
    )


# =============================================================================
# Episode Configuration — controls difficulty & modifiers per episode
# =============================================================================

class EpisodeConfig(BaseModel):
    """
    Configuration for a single training episode, produced by the
    CurriculumManager after analyzing rubric scores.

    Attributes:
        difficulty_level: Numeric difficulty tier (1–5).
        modifiers:        List of active difficulty modifiers for this episode
                          (e.g. "add_river_barrier", "trigger_population_surge").
        seed:             Random seed for reproducible city generation.
        starting_budget:  Initial budget (may be reduced by curriculum).
        grid_size:        Width/height of the city grid (always 16 for now).
    """
    difficulty_level: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Difficulty tier (1=easiest, 5=hardest)",
    )
    modifiers: list[str] = Field(
        default_factory=list,
        description="Active curriculum modifiers for this episode",
    )
    seed: int = Field(
        default=42,
        description="Random seed for deterministic city generation",
    )
    starting_budget: int = Field(
        default=10000,
        ge=0,
        description="Initial budget for the episode",
    )
    grid_size: int = Field(
        default=16,
        ge=4,
        description="Width and height of the city grid",
    )


# =============================================================================
# Action — agent's tool-call payload (discriminated by tool_name)
# =============================================================================

class UrbanPlannerAction(BaseModel):
    """
    Represents a single tool call from the agent.

    The `tool_name` field selects which MCP tool to invoke.  The `arguments`
    dict carries the tool-specific keyword arguments.  This is a thin wrapper
    that lets the environment dispatcher route to the correct handler.

    Allowed tool names (none are reserved OpenEnv names):
        get_city_state, get_district_report, place_zone,
        place_infrastructure, allocate_budget, query_residents,
        query_traffic_model, advance_season, get_event_log,
        get_budget_report
    """
    tool_name: str = Field(
        ...,
        description="Name of the MCP tool to invoke",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the selected tool",
    )


# =============================================================================
# Observation — returned to the agent after each step
# =============================================================================

class UrbanPlannerObservation(BaseModel):
    """
    The observation payload sent back to the agent after each tool call.

    This is what the LLM sees — it must never include hidden (fog-of-war)
    cells.  The `visible_grid` dict maps cell-id strings ("row_col") to
    ZoneCell snapshots.

    Attributes:
        season:           Current season index (0–23; 24 steps = 6 in-game years).
        budget_remaining: How much budget the agent has left.
        visible_grid:     Mapping of "row_col" → ZoneCell for all revealed cells.
        event_log:        Last 3 cascade events (floods, protests, overflows).
        tool_result:      Human-readable result string from the tool just called.
        reward:           Rubric score for this step (0.0–1.0 weighted aggregate).
        done:             Whether the episode has ended.
        info:             Extra metadata, including per-rubric breakdown.
    """
    season: int = Field(
        default=0,
        ge=0,
        description="Current season index (0–23)",
    )
    budget_remaining: int = Field(
        default=0,
        description="Remaining budget after this step",
    )
    visible_grid: dict[str, ZoneCell] = Field(
        default_factory=dict,
        description="Visible city cells keyed by 'row_col'",
    )
    event_log: list[str] = Field(
        default_factory=list,
        description="Recent cascade events (last 3)",
    )
    tool_result: str = Field(
        default="",
        description="Direct result of the tool call",
    )
    reward: float = Field(
        default=0.0,
        description="Rubric-computed reward for this step",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode is over",
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra info including rubric_breakdown",
    )
    planning_log: list[PlanningEntry] = Field(
        default_factory=list,
        description="Server-maintained action/consequence history (last 8 entries)",
    )
    policy_constraints: list[str] = Field(
        default_factory=list,
        description="Active charter constraints for this episode",
    )


# =============================================================================
# State — full server-side state (never exposed raw to the agent)
# =============================================================================

class UrbanPlannerState(BaseModel):
    """
    Complete internal state of a running episode.

    This includes the full 16×16 grid (including hidden cells), budget,
    step counts, difficulty level, and active crises.  The `state()` method
    on the environment class returns this, but the agent only ever sees the
    filtered Observation.

    Attributes:
        episode_id:       Unique identifier for this episode.
        season:           Current season index.
        step_count:       Total tool calls made this episode.
        season_count:     Number of completed season advances.
        budget:           Current budget.
        grid:             Full 16×16 grid as a dict of "row_col" → ZoneCell.
        difficulty_level: Current difficulty tier (1–5).
        active_crises:    List of ongoing crisis identifiers.
        initial_population: Population at episode start (for collapse detection).
        initial_budget:   Budget at episode start.
    """
    episode_id: str = Field(
        default="",
        description="Unique episode identifier",
    )
    season: int = Field(
        default=0,
        ge=0,
        description="Current season index",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Total tool calls so far",
    )
    season_count: int = Field(
        default=0,
        ge=0,
        description="Number of completed season advances",
    )
    budget: int = Field(
        default=10000,
        description="Current remaining budget",
    )
    grid: dict[str, ZoneCell] = Field(
        default_factory=dict,
        description="Full 16×16 grid (server-side, includes hidden cells)",
    )
    difficulty_level: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Current difficulty tier",
    )
    active_crises: list[str] = Field(
        default_factory=list,
        description="Ongoing crisis event identifiers",
    )
    planning_log: list[PlanningEntry] = Field(
        default_factory=list,
        description="Full server-side planning history (trimmed to 8 in observations)",
    )
    initial_population: int = Field(
        default=0,
        ge=0,
        description="Population at episode start (for collapse check)",
    )
    initial_budget: int = Field(
        default=10000,
        ge=0,
        description="Budget at episode start",
    )
