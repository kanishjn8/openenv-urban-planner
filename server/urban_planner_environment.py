# =============================================================================
# OpenEnv Urban Planner — MCPEnvironment (Core Environment Logic)
# =============================================================================
# This is the main environment class that wires together:
#   - CitySimulation (physics engine)
#   - UrbanPlannerRubric (reward computation)
#   - CurriculumManager (adaptive difficulty)
#   - MCP tool dispatch (FastMCP server with 10 registered tools)
#
# Implements the OpenEnv Gym-style API: reset(), step(), state property.
# Session-isolated (is_concurrent = True) so multiple agents can train
# simultaneously against independent city instances.
#
# Tool call flow:
#   1. Agent sends an UrbanPlannerAction (tool_name + arguments).
#   2. step() routes to the appropriate CitySimulation method.
#   3. Shaped reward is computed for this tool call (keeps GRPO signal alive).
#   4. Every STEPS_PER_SEASON tool calls, the season advances automatically
#      and the rubric score overrides the shaped reward.
#   5. PlanningEntry appended and trimmed to last 8 entries.
#   6. UrbanPlannerObservation is returned with visible grid + reward + done.
# =============================================================================

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import State

from models import (
    EpisodeConfig,
    PlanningEntry,
    UrbanPlannerObservation,
    UrbanPlannerState,
    ZoneCell,
)
from server.city_simulation import (
    CitySimulation,
    SHAPED_REWARDS,
    SHAPED_PENALTY_INDUSTRIAL_NEAR_RESIDENTIAL,
    SHAPED_PENALTY_OVERSPEND_EARLY,
    SHAPED_PENALTY_POLICY_VIOLATION,
    STEPS_PER_SEASON,
)
from server.curriculum import CurriculumManager
from server.rubric import urban_planner_rubric

# Maximum number of planning log entries to include in observations
PLANNING_LOG_MAX_ENTRIES = 8

# Path for collapse case JSON files
COLLAPSE_CASES_DIR = Path(__file__).resolve().parent.parent / "assets" / "collapse_cases"

# Episode counter for collapse report filenames
_episode_counter = 0


class UrbanPlannerEnvironment(MCPEnvironment):
    """
    OpenEnv-compliant urban planning environment.

    The agent acts as a city planner making sequential spatial decisions
    (zoning, infrastructure, budget allocation) on a 16×16 grid city
    under resource constraints with cascading physical consequences.

    Attributes:
        is_concurrent: True — each session gets its own city instance.
    """

    # Session-isolated: multiple agents can connect simultaneously
    is_concurrent = True

    def __init__(self) -> None:
        """
        Initialize the environment:
          1. Create the simulation engine, rubric, and curriculum manager.
          2. Build the FastMCP server with all 10 agent-facing tools.
          3. Pass the MCP server to the MCPEnvironment base class.
        """
        self._sim = CitySimulation()
        self._rubric = urban_planner_rubric
        self._curriculum = CurriculumManager()
        self._state: UrbanPlannerState | None = None
        self._last_rubric_scores: dict[str, float] = {}
        self._planning_log: list[PlanningEntry] = []
        self._last_reward: float = 0.0

        # Build and register the MCP tool server
        mcp_server = self._build_mcp_server()
        super().__init__(mcp_server=mcp_server)

    # ── MCP Tool Registration ────────────────────────────────────────────

    def _build_mcp_server(self) -> FastMCP:
        """
        Create a FastMCP server and register all 10 agent-facing tools.

        IMPORTANT: None of the tool names use reserved OpenEnv names
        (reset, step, state, close).
        """
        mcp = FastMCP("OpenEnv Urban Planner")

        # --- Tool 1: get_city_state ---
        @mcp.tool()
        def get_city_state(region: str = "all") -> str:
            """Return a JSON snapshot of all visible grid cells."""
            data = self._sim.get_city_state(region)
            return json.dumps(data, indent=2)

        # --- Tool 2: get_district_report ---
        @mcp.tool()
        def get_district_report(district_id: int) -> str:
            """Return detailed statistics for one 4×4 district."""
            report = self._sim.get_district_report(district_id)
            return json.dumps(report, indent=2)

        # --- Tool 3: place_zone ---
        @mcp.tool()
        def place_zone(x: int, y: int, zone_type: str, density: int) -> str:
            """Rezone a cell. Costs budget based on zone type and density."""
            return self._sim.place_zone(row=x, col=y, zone_type=zone_type, density=density)

        # --- Tool 4: place_infrastructure ---
        @mcp.tool()
        def place_infrastructure(x: int, y: int, infra_type: str) -> str:
            """Place infrastructure (road/metro/hospital/school/flood_barrier) in a cell."""
            return self._sim.place_infrastructure(row=x, col=y, infra_type=infra_type)

        # --- Tool 5: allocate_budget ---
        @mcp.tool()
        def allocate_budget(category: str, amount: int) -> str:
            """Shift budget between maintenance/expansion/emergency categories."""
            return self._sim.allocate_budget(category=category, amount=amount)

        # --- Tool 6: query_residents ---
        @mcp.tool()
        def query_residents(district_id: int) -> str:
            """Return natural-language resident feedback for a district."""
            return self._sim.query_residents(district_id)

        # --- Tool 7: query_traffic_model ---
        @mcp.tool()
        def query_traffic_model(origin: int, destination: int) -> str:
            """Return projected congestion for a route between two districts."""
            return self._sim.query_traffic_model(origin, destination)

        # --- Tool 8: advance_season ---
        @mcp.tool()
        def advance_season() -> str:
            """Fast-forward simulation by 1 season without placing anything."""
            return self._sim.advance_season()

        # --- Tool 9: get_event_log ---
        @mcp.tool()
        def get_event_log(last_n: int = 5) -> str:
            """Return the most recent cascade events (floods, protests, etc.)."""
            events = self._sim.get_event_log(last_n)
            return json.dumps(events, indent=2)

        # --- Tool 10: get_budget_report ---
        @mcp.tool()
        def get_budget_report() -> str:
            """Return formatted seasonal revenue/expenditure breakdown with warnings."""
            return self._sim.get_budget_report()

        return mcp

    # ── Gym-style API ────────────────────────────────────────────────────

    def reset(self, config: EpisodeConfig | None = None) -> UrbanPlannerObservation:
        """
        Reset the environment for a new episode.

        If no config is provided, the curriculum manager generates one
        based on the agent's performance in the previous episode.

        Args:
            config: Optional episode configuration. If None, the curriculum
                    manager produces an adaptive config.

        Returns:
            Initial observation with the seed city visible grid and
            policy_constraints injected.
        """
        cfg = config or self._curriculum.next_episode_config(self._last_rubric_scores)
        self._sim.initialize(cfg)

        # Reset planning log
        self._planning_log = []
        self._last_reward = 0.0

        self._state = UrbanPlannerState(
            episode_id=str(uuid4()),
            season=0,
            step_count=0,
            season_count=0,
            budget=self._sim.budget,
            grid={
                key: cell.model_copy()
                for key, cell in self._sim.grid.items()
            },
            difficulty_level=cfg.difficulty_level,
            active_crises=[],
            planning_log=[],
            initial_population=self._sim.initial_population,
            initial_budget=cfg.starting_budget,
        )

        return self._make_observation(
            tool_result="City initialized. Ready to plan. Use get_city_state to see the grid.",
            policy_constraints=self._sim.policy_constraints,
        )

    def _step_impl(
        self,
        action: object,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> UrbanPlannerObservation:
        """
        OpenEnv-core hook for non-MCP actions.

        `openenv-core`'s `MCPEnvironment.step()` routes MCP actions
        (ListToolsAction / CallToolAction) internally. Everything else must be
        handled by subclasses via `_step_impl()`.

        For this environment, we support a dict-based action protocol:
        `{ "tool_name": str, "arguments": dict }`, which mirrors
        `UrbanPlannerAction`.
        """
        if isinstance(action, dict):
            # Delegate to our dict-based step implementation.
            return self.step(action)

        # Anything else is unsupported for this environment.
        return self._make_observation(
            tool_result=(
                "Error: unsupported action type for UrbanPlannerEnvironment. "
                "Expected a dict with 'tool_name'/'arguments'."
            ),
            done=True,
        )

    def step(self, action: dict) -> UrbanPlannerObservation:
        """
        Execute one agent tool call and return the resulting observation.

        Reward logic (per plan §5.3):
          - Between season boundaries: shaped reward based on tool type + penalties.
          - At season boundaries: rubric score overrides shaped reward.
          - The reward field is always non-zero.

        After every STEPS_PER_SEASON calls, the season auto-advances
        and cascades are computed.

        Args:
            action: Dict with 'tool_name' and 'arguments' keys.

        Returns:
            Observation with tool result, reward, done flag, and planning_log.
        """
        if self._state is None:
            return self._make_observation(
                tool_result="Error: environment not initialized. Call reset() first.",
                done=True,
            )

        tool_name = action.get("tool_name", "")
        arguments = action.get("arguments", {})

        # Dispatch the tool call to the simulation
        tool_result = self._dispatch_tool(tool_name, arguments)

        # Increment step counter
        self._state.step_count += 1

        # Compute shaped reward for this tool call
        reward = self._shaped_reward(tool_name, arguments)

        # Auto-advance season every STEPS_PER_SEASON tool calls
        is_season_boundary = self._state.step_count % STEPS_PER_SEASON == 0
        if is_season_boundary:
            season_result = self._sim.advance_season()
            self._state.season_count += 1
            self._state.season = self._sim.season
            tool_result += f"\n[Season advanced: {season_result}]"

            # Rubric score overrides shaped reward at season boundary
            total_reward, breakdown = self._rubric.score(self._sim)
            self._last_rubric_scores = breakdown
            reward = total_reward  # rubric overrides shaped reward
        else:
            breakdown = self._last_rubric_scores

        # Ensure reward is always non-zero (plan rule)
        if reward == 0.0:
            reward = 0.001  # minimal positive signal

        # Update state from simulation
        self._state.budget = self._sim.budget

        # Check terminal conditions
        done = self._sim.is_terminal()

        # If collapsing, generate and log collapse report + ASCII snapshot
        collapse_info = {}
        if done and self._sim.season < 24:
            # City collapsed (not just max seasons)
            self._sim._ascii_snapshots[self._sim.season] = self._sim.generate_ascii_snapshot()
            collapse_report = self._sim.generate_collapse_report()
            collapse_info["collapse_report"] = collapse_report.model_dump()
            self._log_collapse_report(collapse_report)

        # Sync active crises
        self._state.active_crises = [
            e for e in self._sim.event_log[-10:]
            if "FLOOD" in e or "PROTEST" in e or "SHORTFALL" in e
        ]

        # Append to planning log, trim to last 8 entries
        reward_delta = reward - self._last_reward
        self._last_reward = reward
        entry = PlanningEntry(
            season=self._sim.season,
            action_summary=f"{tool_name}({json.dumps(arguments)})" if arguments else tool_name,
            consequence=tool_result[:200],  # truncate long results
            reward_delta=round(reward_delta, 4),
        )
        self._planning_log.append(entry)
        self._planning_log = self._planning_log[-PLANNING_LOG_MAX_ENTRIES:]

        # Sync planning log to state
        self._state.planning_log = list(self._planning_log)

        return self._make_observation(
            tool_result=tool_result,
            reward=reward,
            done=done,
            rubric_breakdown={**breakdown, **collapse_info},
            policy_constraints=self._sim.policy_constraints,
        )

    @property
    def state(self) -> State:
        """Return the full internal state (server-side only)."""
        return self._state

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _shaped_reward(self, tool_name: str, arguments: dict) -> float:
        """
        Compute the shaped intermediate reward for a tool call.

        This keeps GRPO gradients alive between season boundaries.
        Includes base tool reward + penalties for bad placements.
        """
        base = SHAPED_REWARDS.get(tool_name, 0.01)

        # Penalty: placing industrial adjacent to residential
        if tool_name == "place_zone" and arguments.get("zone_type") == "industrial":
            x, y = arguments.get("x", 0), arguments.get("y", 0)
            for nr, nc in self._sim._neighbors(x, y):
                nkey = self._sim._cell_key(nr, nc)
                if self._sim.grid[nkey].zone_type == "residential":
                    base += SHAPED_PENALTY_INDUSTRIAL_NEAR_RESIDENTIAL
                    break

        # Penalty: spending > 40% budget in first 4 seasons
        if self._state and self._state.season_count < 4:
            spent = self._state.initial_budget - self._sim.budget
            if spent > self._state.initial_budget * 0.4:
                base += SHAPED_PENALTY_OVERSPEND_EARLY

        # Penalty: violating policy constraints
        if self._state and self._sim.budget < 500:
            # Check emergency fund constraint
            if any("$500" in c for c in self._sim.policy_constraints):
                base += SHAPED_PENALTY_POLICY_VIOLATION

        return round(base, 4)

    def _dispatch_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Route a tool call to the appropriate CitySimulation method.

        Returns the tool's result string.
        """
        dispatch_map = {
            "get_city_state":        lambda: json.dumps(
                self._sim.get_city_state(arguments.get("region", "all")), indent=2
            ),
            "get_district_report":   lambda: json.dumps(
                self._sim.get_district_report(arguments["district_id"]), indent=2
            ),
            "place_zone":            lambda: self._sim.place_zone(
                arguments["x"], arguments["y"],
                arguments["zone_type"], arguments["density"],
            ),
            "place_infrastructure":  lambda: self._sim.place_infrastructure(
                arguments["x"], arguments["y"], arguments["infra_type"],
            ),
            "allocate_budget":       lambda: self._sim.allocate_budget(
                arguments["category"], arguments["amount"],
            ),
            "query_residents":       lambda: self._sim.query_residents(
                arguments["district_id"],
            ),
            "query_traffic_model":   lambda: self._sim.query_traffic_model(
                arguments["origin"], arguments["destination"],
            ),
            "advance_season":        lambda: self._sim.advance_season(),
            "get_event_log":         lambda: json.dumps(
                self._sim.get_event_log(arguments.get("last_n", 5)), indent=2
            ),
            "get_budget_report":     lambda: self._sim.get_budget_report(),
        }

        handler = dispatch_map.get(tool_name)
        if handler is None:
            return f"Error: unknown tool '{tool_name}'. Available: {list(dispatch_map.keys())}"

        try:
            return handler()
        except (KeyError, TypeError) as exc:
            return f"Error calling {tool_name}: {exc}"

    def _make_observation(
        self,
        tool_result: str = "",
        reward: float = 0.0,
        done: bool = False,
        rubric_breakdown: dict[str, float] | None = None,
        policy_constraints: list[str] | None = None,
    ) -> UrbanPlannerObservation:
        """
        Build an UrbanPlannerObservation from the current simulation state.

        Filters the grid to only include visible cells (fog-of-war enforcement).
        Injects planning_log (trimmed to 8) and policy_constraints.
        """
        # Build visible grid (server-side fog-of-war enforcement)
        visible_grid: dict[str, ZoneCell] = {}
        for key, cell in self._sim.grid.items():
            if cell.visible:
                visible_grid[key] = cell.model_copy()

        return UrbanPlannerObservation(
            season=self._sim.season,
            budget_remaining=self._sim.budget,
            visible_grid=visible_grid,
            event_log=self._sim.get_event_log(3),
            tool_result=tool_result,
            reward=reward,
            done=done,
            info={"rubric_breakdown": rubric_breakdown or {}},
            planning_log=list(self._planning_log),
            policy_constraints=policy_constraints or [],
        )

    def _log_collapse_report(self, report) -> None:
        """
        Write a collapse report to assets/collapse_cases/episode_{n}.json.
        """
        global _episode_counter
        _episode_counter += 1

        try:
            COLLAPSE_CASES_DIR.mkdir(parents=True, exist_ok=True)
            filepath = COLLAPSE_CASES_DIR / f"episode_{_episode_counter}.json"
            with open(filepath, "w") as f:
                json.dump(report.model_dump(), f, indent=2)
        except OSError:
            # Don't crash the environment if file I/O fails
            pass
