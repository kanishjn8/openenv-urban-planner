# =============================================================================
# OpenEnv Urban Planner — Rubric Tests
# =============================================================================
# Tests for all five rubric sub-components and the composite scorer.
# Validates scoring bounds (0.0–1.0) and directional correctness.
# =============================================================================

from __future__ import annotations

from openenv_urban_planner.models import EpisodeConfig
from openenv_urban_planner.server.city_simulation import CitySimulation
from openenv_urban_planner.server.rubric import (
    ConnectivityRubric,
    ResidentWelfareRubric,
    EconomicViabilityRubric,
    BudgetEfficiencyRubric,
    LongHorizonCoherenceRubric,
    urban_planner_rubric,
)


class TestRubricScores:
    """Tests for individual rubric components and the composite scorer."""

    def _make_sim(self, seed: int = 42) -> CitySimulation:
        """Helper: create and initialize a simulation."""
        sim = CitySimulation()
        config = EpisodeConfig(seed=seed, starting_budget=10000)
        sim.initialize(config)
        return sim

    # ── Bounds check: every rubric should return [0, 1] ──────────────────

    def test_connectivity_bounds(self) -> None:
        """Connectivity score should be in [0, 1]."""
        sim = self._make_sim()
        score = ConnectivityRubric().score(sim)
        assert 0.0 <= score <= 1.0

    def test_welfare_bounds(self) -> None:
        """Welfare score should be in [0, 1]."""
        sim = self._make_sim()
        score = ResidentWelfareRubric().score(sim)
        assert 0.0 <= score <= 1.0

    def test_economic_bounds(self) -> None:
        """Economic score should be in [0, 1]."""
        sim = self._make_sim()
        score = EconomicViabilityRubric().score(sim)
        assert 0.0 <= score <= 1.0

    def test_efficiency_bounds(self) -> None:
        """Efficiency score should be in [0, 1]."""
        sim = self._make_sim()
        score = BudgetEfficiencyRubric().score(sim)
        assert 0.0 <= score <= 1.0

    def test_coherence_bounds(self) -> None:
        """Coherence score should be in [0, 1]."""
        sim = self._make_sim()
        score = LongHorizonCoherenceRubric().score(sim)
        assert 0.0 <= score <= 1.0

    # ── Directional correctness ──────────────────────────────────────────

    def test_connectivity_improves_with_roads(self) -> None:
        """Adding roads should improve connectivity score."""
        sim = self._make_sim()
        rubric = ConnectivityRubric()
        score_before = rubric.score(sim)

        # Add roads connecting residential cells to commercial zones
        for r in range(sim.grid_size):
            for c in range(sim.grid_size):
                sim.place_infrastructure(r, c, "road")

        score_after = rubric.score(sim)
        assert score_after >= score_before

    def test_welfare_decreases_with_congestion(self) -> None:
        """High congestion should lower welfare score."""
        sim = self._make_sim()
        rubric = ResidentWelfareRubric()
        score_before = rubric.score(sim)

        # Max out congestion on all residential cells
        for cell in sim.grid.values():
            if cell.zone_type == "residential":
                cell.congestion = 1.0

        score_after = rubric.score(sim)
        assert score_after < score_before

    def test_coherence_penalizes_industrial_near_school(self) -> None:
        """Industrial zones adjacent to schools should lower coherence."""
        sim = self._make_sim()
        rubric = LongHorizonCoherenceRubric()

        # Place industrial zone next to a cell with a school
        sim.grid["0_0"].zone_type = "industrial"
        sim.grid["0_0"].density = 2
        sim.grid["0_1"].infrastructure = ["school"]
        sim.grid["0_1"].zone_type = "residential"
        sim.grid["0_1"].density = 1

        score = rubric.score(sim)
        # Should be below 1.0 due to the contradiction
        assert score < 1.0

    # ── Composite rubric ─────────────────────────────────────────────────

    def test_composite_returns_total_and_breakdown(self) -> None:
        """Composite rubric should return (float, dict) tuple."""
        sim = self._make_sim()
        total, breakdown = urban_planner_rubric.score(sim)
        assert isinstance(total, float)
        assert isinstance(breakdown, dict)
        assert 0.0 <= total <= 1.0
        assert set(breakdown.keys()) == {
            "connectivity", "welfare", "economic", "efficiency", "coherence"
        }

    def test_composite_weights_sum_to_one(self) -> None:
        """Rubric weights should sum to 1.0."""
        total_weight = sum(
            w for _, w in urban_planner_rubric.components.values()
        )
        assert abs(total_weight - 1.0) < 1e-6
