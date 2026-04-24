# =============================================================================
# OpenEnv Urban Planner — Environment Tests
# =============================================================================
# Tests for the core environment lifecycle: initialization, reset, step,
# tool dispatch, season advancement, and terminal conditions.
# =============================================================================

from __future__ import annotations

from openenv_urban_planner.models import EpisodeConfig, UrbanPlannerState
from openenv_urban_planner.server.city_simulation import (
    CitySimulation,
    SHAPED_REWARDS,
)


class TestCitySimulation:
    """Tests for the CitySimulation engine."""

    def _make_sim(self, seed: int = 42) -> CitySimulation:
        """Helper: create and initialize a simulation with default config."""
        sim = CitySimulation()
        config = EpisodeConfig(seed=seed, starting_budget=10000)
        sim.initialize(config)
        return sim

    # ── Initialization ───────────────────────────────────────────────────

    def test_initialize_creates_grid(self) -> None:
        """Grid should be 16×16 = 256 cells after init."""
        sim = self._make_sim()
        assert len(sim.grid) == 16 * 16

    def test_initialize_seeds_center_districts(self) -> None:
        """Center 4×4 block should contain non-empty zones."""
        sim = self._make_sim()
        center = sim.grid_size // 2 - 2
        for r in range(center, center + 4):
            for c in range(center, center + 4):
                cell = sim.grid[f"{r}_{c}"]
                assert cell.zone_type != "empty", f"Seed cell ({r},{c}) should not be empty"

    def test_initialize_sets_budget(self) -> None:
        """Budget should match the config's starting_budget."""
        sim = self._make_sim()
        assert sim.budget == 10000

    def test_initialize_has_fog_of_war(self) -> None:
        """Approximately 30% of cells should be hidden."""
        sim = self._make_sim()
        hidden = sum(1 for c in sim.grid.values() if not c.visible)
        # Allow some variance: 30% of 256 ≈ 77, accept 50–100
        assert 50 <= hidden <= 100, f"Expected ~77 hidden cells, got {hidden}"

    def test_deterministic_seeding(self) -> None:
        """Same seed should produce identical grids."""
        sim_a = self._make_sim(seed=123)
        sim_b = self._make_sim(seed=123)
        for key in sim_a.grid:
            assert sim_a.grid[key].zone_type == sim_b.grid[key].zone_type
            assert sim_a.grid[key].density == sim_b.grid[key].density

    # ── Zone & Infrastructure Placement ──────────────────────────────────

    def test_place_zone_valid(self) -> None:
        """Placing a valid zone should update the cell and deduct budget."""
        sim = self._make_sim()
        initial_budget = sim.budget
        result = sim.place_zone(0, 0, "residential", 2)
        assert "Error" not in result
        assert sim.grid["0_0"].zone_type == "residential"
        assert sim.grid["0_0"].density == 2
        assert sim.budget < initial_budget

    def test_place_zone_invalid_type(self) -> None:
        """Invalid zone type should return an error."""
        sim = self._make_sim()
        result = sim.place_zone(0, 0, "lava_pit", 1)
        assert "Error" in result

    def test_place_infrastructure_valid(self) -> None:
        """Placing valid infrastructure should update the cell."""
        sim = self._make_sim()
        result = sim.place_infrastructure(0, 0, "road")
        assert "Error" not in result
        assert "road" in sim.grid["0_0"].infrastructure

    def test_place_infrastructure_reveals_neighbors(self) -> None:
        """Placing infrastructure should reveal adjacent cells."""
        sim = self._make_sim()
        # Force neighbors to be hidden
        sim.grid["0_1"].visible = False
        sim.grid["1_0"].visible = False
        sim.place_infrastructure(0, 0, "road")
        assert sim.grid["0_1"].visible is True
        assert sim.grid["1_0"].visible is True

    def test_insufficient_budget_rejects_placement(self) -> None:
        """Placement should fail if budget is too low."""
        sim = self._make_sim()
        sim.budget = 0
        result = sim.place_zone(0, 0, "commercial", 3)
        assert "Error" in result

    # ── Season Advancement ───────────────────────────────────────────────

    def test_advance_season_increments_counter(self) -> None:
        """Season counter should increase by 1."""
        sim = self._make_sim()
        sim.advance_season()
        assert sim.season == 1

    def test_advance_season_generates_events(self) -> None:
        """Season advance should populate the event log."""
        sim = self._make_sim()
        # Force a flood-prone cell
        sim.grid["0_0"].flood_risk = 0.9
        sim.grid["0_0"].infrastructure = []  # no barrier
        sim.advance_season()
        # Should have at least one flood event
        assert any("FLOOD" in e for e in sim.event_log)

    # ── Terminal Conditions ──────────────────────────────────────────────

    def test_terminal_on_max_seasons(self) -> None:
        """Episode ends after 24 seasons."""
        sim = self._make_sim()
        sim.season = 24
        assert sim.is_terminal() is True

    def test_terminal_on_negative_budget(self) -> None:
        """Episode ends if budget goes negative."""
        sim = self._make_sim()
        sim.budget = -1
        assert sim.is_terminal() is True

    def test_terminal_on_population_collapse(self) -> None:
        """Episode ends if population drops below 20% of initial."""
        sim = self._make_sim()
        # Zero out all population
        for cell in sim.grid.values():
            cell.population = 0
        assert sim.is_terminal() is True

    def test_not_terminal_on_healthy_city(self) -> None:
        """A healthy city in early seasons should not be terminal."""
        sim = self._make_sim()
        assert sim.is_terminal() is False

    # ── Query Tools ──────────────────────────────────────────────────────

    def test_get_city_state_only_visible(self) -> None:
        """get_city_state should exclude hidden cells."""
        sim = self._make_sim()
        visible_state = sim.get_city_state()
        for key, data in visible_state.items():
            assert sim.grid[key].visible is True

    def test_get_district_report_reveals_cells(self) -> None:
        """Querying a district should reveal all its cells."""
        sim = self._make_sim()
        # Hide all cells in district 0
        for r in range(4):
            for c in range(4):
                sim.grid[f"{r}_{c}"].visible = False
        sim.get_district_report(0)
        for r in range(4):
            for c in range(4):
                assert sim.grid[f"{r}_{c}"].visible is True

    # ── Budget Report ────────────────────────────────────────────────────

    def test_get_budget_report_returns_formatted_string(self) -> None:
        """get_budget_report should return a multi-line string with revenue and expenditure."""
        sim = self._make_sim()
        report = sim.get_budget_report()
        assert "Budget Report" in report
        assert "Revenue" in report
        assert "Expenditure" in report
        assert "Net" in report

    def test_get_budget_report_includes_warnings_when_budget_low(self) -> None:
        """Budget report should include a warning when budget is critically low."""
        sim = self._make_sim()
        sim.budget = 500
        report = sim.get_budget_report()
        assert "Warning" in report

    # ── Collapse Report ──────────────────────────────────────────────────

    def test_generate_collapse_report_on_budget_exhaustion(self) -> None:
        """Collapse report should identify budget exhaustion as trigger."""
        sim = self._make_sim()
        sim.budget = -100
        report = sim.generate_collapse_report()
        assert "Budget exhausted" in report.trigger
        assert report.season_of_no_return >= 0

    def test_generate_collapse_report_on_population_collapse(self) -> None:
        """Collapse report should identify population collapse."""
        sim = self._make_sim()
        for cell in sim.grid.values():
            cell.population = 0
        report = sim.generate_collapse_report()
        assert "Population collapsed" in report.trigger

    def test_collapse_report_has_required_fields(self) -> None:
        """CollapseReport should have all four required fields."""
        sim = self._make_sim()
        sim.budget = -1
        report = sim.generate_collapse_report()
        assert hasattr(report, 'trigger')
        assert hasattr(report, 'chain')
        assert hasattr(report, 'season_of_no_return')
        assert hasattr(report, 'agent_mistake')

    # ── ASCII Snapshot ───────────────────────────────────────────────────

    def test_ascii_snapshot_generated_at_init(self) -> None:
        """ASCII snapshot should be generated at season 0 on init."""
        sim = self._make_sim()
        assert 0 in sim._ascii_snapshots
        assert "S0:" in sim._ascii_snapshots[0]

    def test_ascii_snapshot_contains_zone_symbols(self) -> None:
        """ASCII snapshot should contain zone type symbols."""
        sim = self._make_sim()
        snapshot = sim.generate_ascii_snapshot()
        # Should have at least some residential (R) and commercial (C) from seed
        assert "R" in snapshot
        assert "C" in snapshot

    def test_ascii_snapshot_logged_at_season_12(self) -> None:
        """ASCII snapshot should be logged at season 12."""
        sim = self._make_sim()
        for _ in range(12):
            sim.advance_season()
        assert 12 in sim._ascii_snapshots

    # ── Policy Constraints ───────────────────────────────────────────────

    def test_policy_constraints_set_on_init(self) -> None:
        """Policy constraints should be set based on difficulty level."""
        sim = CitySimulation()
        config = EpisodeConfig(seed=42, starting_budget=10000, difficulty_level=3)
        sim.initialize(config)
        assert len(sim.policy_constraints) > 0
        assert any("industrial" in c.lower() for c in sim.policy_constraints)

    def test_policy_constraints_scale_with_difficulty(self) -> None:
        """Higher difficulty should have more constraints."""
        sim1 = CitySimulation()
        sim1.initialize(EpisodeConfig(seed=42, difficulty_level=1))
        sim5 = CitySimulation()
        sim5.initialize(EpisodeConfig(seed=42, difficulty_level=5))
        assert len(sim5.policy_constraints) > len(sim1.policy_constraints)

    # ── Shaped Rewards ───────────────────────────────────────────────────

    def test_shaped_rewards_defined_for_all_tools(self) -> None:
        """SHAPED_REWARDS should have entries for all 10 tools."""
        expected_tools = {
            "place_infrastructure", "place_zone", "query_residents",
            "get_city_state", "get_budget_report", "get_district_report",
            "query_traffic_model", "get_event_log", "allocate_budget",
            "advance_season",
        }
        assert set(SHAPED_REWARDS.keys()) == expected_tools

    def test_shaped_rewards_are_non_negative(self) -> None:
        """Base shaped rewards should be non-negative."""
        for tool, reward in SHAPED_REWARDS.items():
            assert reward >= 0.0, f"Shaped reward for {tool} should be >= 0"
