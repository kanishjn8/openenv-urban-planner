# =============================================================================
# OpenEnv Urban Planner — Regression Tests for Audit Bug Fixes
# =============================================================================
# Each test pins down a specific bug identified in the project audit so it
# cannot silently re-emerge.
#
#   #1 advance_season double-step          → tests in TestAdvanceSeasonNoDoubleStep
#   #2 BudgetEfficiencyRubric math         → tests in TestBudgetEfficiencyMath
#   #3 ConnectivityRubric residential bridge → TestConnectivityRoadOnly
#   #4 school_load=2.0 auto-collapse       → TestSchoolLoadEarlyGame
#   #5 LongHorizonCoherence dedup hack     → TestCoherenceHighDensityDedup
#   #6 Info-tool shaped reward exploit     → TestShapedRewardsNotExploitable
#   #7 Package import works as library     → TestPackageImports
# =============================================================================

from __future__ import annotations


from openenv_urban_planner.models import EpisodeConfig
from openenv_urban_planner.server.city_simulation import (
    CitySimulation,
    SHAPED_REWARDS,
)
from openenv_urban_planner.server.rubric import (
    BudgetEfficiencyRubric,
    ConnectivityRubric,
    LongHorizonCoherenceRubric,
)
from openenv_urban_planner.server.urban_planner_environment import (
    UrbanPlannerEnvironment,
)


# ── Bug #1 — advance_season must not double-step ────────────────────────────


class TestAdvanceSeasonNoDoubleStep:
    """The agent's `advance_season` tool must advance the season exactly once,
    even when the call lands on the auto-advance boundary."""

    def _make_env(self) -> UrbanPlannerEnvironment:
        env = UrbanPlannerEnvironment()
        env.reset(EpisodeConfig(seed=42, starting_budget=10_000))
        return env

    def test_six_get_states_advances_twice(self) -> None:
        """Six pure-info calls = two season boundaries = season should be 2."""
        env = self._make_env()
        for _ in range(6):
            env.step({"tool_name": "get_city_state", "arguments": {"region": "all"}})
        assert env._sim.season == 2

    def test_advance_season_on_boundary_does_not_double_step(self) -> None:
        """The previous bug was: agent calls advance_season as the 3rd tool,
        season jumps from 0 to 2 in a single step.  After the fix, season
        should be 1."""
        env = self._make_env()
        env.step({"tool_name": "get_city_state", "arguments": {"region": "all"}})
        env.step({"tool_name": "get_city_state", "arguments": {"region": "all"}})
        env.step({"tool_name": "advance_season", "arguments": {}})
        assert env._sim.season == 1, (
            f"Expected season=1 after one explicit advance on boundary, got {env._sim.season}"
        )

    def test_six_steps_with_alternating_advance_advances_twice(self) -> None:
        """Two info + one advance, twice → exactly 2 seasons advance."""
        env = self._make_env()
        for _ in range(2):
            env.step({"tool_name": "get_city_state", "arguments": {"region": "all"}})
            env.step({"tool_name": "get_city_state", "arguments": {"region": "all"}})
            env.step({"tool_name": "advance_season", "arguments": {}})
        assert env._sim.season == 2


# ── Bug #2 — BudgetEfficiencyRubric uses initial_budget ─────────────────────


class TestBudgetEfficiencyMath:
    """Spending budget must monotonically decrease the efficiency score, and
    the formula must reference the actual starting budget, not a population
    proxy."""

    def _sim(self, starting_budget: int = 10_000) -> CitySimulation:
        sim = CitySimulation()
        sim.initialize(EpisodeConfig(seed=42, starting_budget=starting_budget))
        return sim

    def test_initial_budget_is_recorded(self) -> None:
        """`sim.initial_budget` must be set at reset for downstream rubrics."""
        sim = self._sim(starting_budget=12_345)
        assert sim.initial_budget == 12_345

    def test_full_budget_gives_full_welfare_credit(self) -> None:
        sim = self._sim()
        rubric = BudgetEfficiencyRubric()
        score = rubric.score(sim)
        # With nothing spent, score should equal welfare (no spending penalty)
        from openenv_urban_planner.server.rubric import ResidentWelfareRubric
        assert abs(score - ResidentWelfareRubric().score(sim)) < 1e-6

    def test_spending_decreases_score(self) -> None:
        sim = self._sim()
        before = BudgetEfficiencyRubric().score(sim)
        sim.budget = 0  # spent everything
        after = BudgetEfficiencyRubric().score(sim)
        assert after < before

    def test_score_in_bounds_under_overdraft(self) -> None:
        sim = self._sim()
        sim.budget = -500  # over-draft
        score = BudgetEfficiencyRubric().score(sim)
        assert 0.0 <= score <= 1.0


# ── Bug #3 — ConnectivityRubric must require actual roads ───────────────────


class TestConnectivityRoadOnly:
    """A residential cell is connected iff it has road access — a chain of
    residentials with no roads must NOT count as connected."""

    def _blank_sim(self) -> CitySimulation:
        sim = CitySimulation()
        sim.initialize(EpisodeConfig(seed=42, starting_budget=10_000))
        # Wipe everything to a known empty state
        for cell in sim.grid.values():
            cell.zone_type = "empty"
            cell.density = 0
            cell.population = 0
            cell.infrastructure = []
        return sim

    def test_residential_chain_without_roads_is_not_connected(self) -> None:
        sim = self._blank_sim()
        # commercial+road at (0,0); residentials at (0,1)-(0,3) — no roads
        sim.grid["0_0"].zone_type = "commercial"
        sim.grid["0_0"].infrastructure = ["road"]
        for c in (1, 2, 3):
            sim.grid[f"0_{c}"].zone_type = "residential"
            sim.grid[f"0_{c}"].population = 50
        # only (0,1) is adjacent to a road cell ⇒ only 1/3 connected
        score = ConnectivityRubric().score(sim)
        assert abs(score - 1.0 / 3.0) < 1e-6, (
            f"Expected 0.333 (only direct neighbor counts), got {score}"
        )

    def test_residential_with_road_neighbor_counts(self) -> None:
        sim = self._blank_sim()
        sim.grid["0_0"].zone_type = "commercial"
        sim.grid["0_0"].infrastructure = ["road"]
        sim.grid["0_1"].infrastructure = ["road"]  # extension
        sim.grid["0_2"].zone_type = "residential"  # adjacent to (0,1)
        sim.grid["0_2"].population = 50
        score = ConnectivityRubric().score(sim)
        assert score == 1.0

    def test_no_commercial_road_means_zero(self) -> None:
        sim = self._blank_sim()
        sim.grid["0_0"].zone_type = "residential"
        sim.grid["0_0"].population = 50
        score = ConnectivityRubric().score(sim)
        assert score == 0.0


# ── Bug #4 — School-load must not auto-collapse a passive city ──────────────


class TestSchoolLoadEarlyGame:
    """An untouched city should not collapse purely from school-load decline
    in the first ~12 seasons."""

    def test_passive_city_survives_to_season_12(self) -> None:
        sim = CitySimulation()
        sim.initialize(EpisodeConfig(seed=42, starting_budget=10_000))
        for _ in range(12):
            sim.advance_season()
        assert not sim.is_terminal(), (
            f"Passive city collapsed by S{sim.season}: pop={sim._total_population()} "
            f"of {sim.initial_population}"
        )

    def test_no_protests_when_no_school_in_catchment(self) -> None:
        sim = CitySimulation()
        sim.initialize(EpisodeConfig(seed=42, starting_budget=10_000))
        # Make sure there are no schools anywhere
        for cell in sim.grid.values():
            cell.infrastructure = [i for i in cell.infrastructure if i != "school"]
        sim.advance_season()
        assert not any("PROTEST" in e for e in sim.event_log), (
            "Spurious PROTEST events fired even though no school exists to protest about"
        )


# ── Bug #5 — Coherence rule C must not depend on neighbor ordering ──────────


class TestCoherenceHighDensityDedup:
    """High-density residential without road access must count as exactly one
    contradiction per cell, regardless of whether _neighbors returns 2, 3, or
    4 entries (boundary effects)."""

    def _blank_sim(self) -> CitySimulation:
        sim = CitySimulation()
        sim.initialize(EpisodeConfig(seed=42, starting_budget=10_000))
        for cell in sim.grid.values():
            cell.zone_type = "empty"
            cell.density = 0
            cell.population = 0
            cell.infrastructure = []
        return sim

    def test_corner_cell_with_density_3_and_no_roads_is_a_contradiction(self) -> None:
        # Corner cells have only 2 neighbors — the old dedup hack worked but
        # was fragile.  Verify the new code still flags this case.
        sim = self._blank_sim()
        sim.grid["0_0"].zone_type = "residential"
        sim.grid["0_0"].density = 3
        sim.grid["0_0"].population = 400
        score = LongHorizonCoherenceRubric().score(sim)
        assert score < 1.0

    def test_adjacent_road_neighbor_resolves_contradiction(self) -> None:
        sim = self._blank_sim()
        sim.grid["0_0"].zone_type = "residential"
        sim.grid["0_0"].density = 3
        sim.grid["0_0"].population = 400
        sim.grid["0_1"].infrastructure = ["road"]
        score = LongHorizonCoherenceRubric().score(sim)
        # Only contradiction for (0,0) was "no road access"; with a road
        # neighbor it should be resolved → score 1.0 (or near it).
        assert score == 1.0


# ── Bug #6 — Info tools must not give exploitable shaped reward ─────────────


class TestShapedRewardsNotExploitable:
    INFO_TOOLS = {
        "query_residents", "get_city_state", "get_budget_report",
        "get_district_report", "query_traffic_model", "get_event_log",
    }

    def test_all_info_tools_zero(self) -> None:
        for tool in self.INFO_TOOLS:
            assert SHAPED_REWARDS[tool] == 0.0, (
                f"Info tool '{tool}' has nonzero shaped reward "
                f"({SHAPED_REWARDS[tool]}) — exploit risk"
            )

    def test_constructive_tools_still_positive(self) -> None:
        assert SHAPED_REWARDS["place_infrastructure"] > 0
        assert SHAPED_REWARDS["place_zone"] > 0


# ── Bug #7 — Package imports work as a library ──────────────────────────────


class TestPackageImports:
    """Smoke-test the supported library import paths.  If these fail, the
    package layout in pyproject.toml or the relative imports in __init__.py /
    client.py have regressed."""

    def test_top_level_import(self) -> None:
        import openenv_urban_planner  # noqa: F401

    def test_client_reexport(self) -> None:
        from openenv_urban_planner import UrbanPlannerEnv  # noqa: F401

    def test_models_import(self) -> None:
        from openenv_urban_planner.models import EpisodeConfig  # noqa: F401

    def test_environment_import(self) -> None:
        from openenv_urban_planner.server.urban_planner_environment import (  # noqa: F401
            UrbanPlannerEnvironment,
        )

    def test_rubric_import(self) -> None:
        from openenv_urban_planner.server.rubric import urban_planner_rubric  # noqa: F401
