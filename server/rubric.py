# =============================================================================
# OpenEnv Urban Planner — Rubric Scoring System
# =============================================================================
# Five composable sub-rubrics that evaluate the agent's city-planning quality.
# Each returns 0.0–1.0.  Weighted aggregation produces the final reward.
#
# Rubric tension (why this is hard to game):
#   - Connectivity demands roads → costs budget → hurts efficiency
#   - Economic viability wants commercial zones → creates traffic → hurts welfare
#   - Long-horizon coherence prevents greedy local optimization
# =============================================================================

from __future__ import annotations

from openenv_urban_planner.server.city_simulation import CitySimulation


# ── Base class (lightweight, no openenv dependency needed for scoring) ───────

class Rubric:
    """Base class for a single scoring rubric.  Returns 0.0–1.0."""

    def score(self, sim: CitySimulation) -> float:
        raise NotImplementedError


# ── Sub-rubric 1: Connectivity ──────────────────────────────────────────────

class ConnectivityRubric(Rubric):
    """
    Fraction of residential cells reachable from any commercial zone via
    a connected road network.

    Score = connected_residential / total_residential

    Uses BFS from every commercial cell that has a road, traversing only
    road-connected cells, and marks which residential cells are reached.
    """

    def score(self, sim: CitySimulation) -> float:
        # Gather residential and commercial cells
        residential_keys: set[str] = set()
        commercial_with_road: list[tuple[int, int]] = []

        for r in range(sim.grid_size):
            for c in range(sim.grid_size):
                key = sim._cell_key(r, c)
                cell = sim.grid[key]
                if cell.zone_type == "residential":
                    residential_keys.add(key)
                if cell.zone_type == "commercial" and "road" in cell.infrastructure:
                    commercial_with_road.append((r, c))

        if not residential_keys:
            return 1.0  # no residential → trivially connected

        # BFS from all commercial road cells through the road network
        visited: set[str] = set()
        queue: list[tuple[int, int]] = list(commercial_with_road)
        for r, c in queue:
            visited.add(sim._cell_key(r, c))

        while queue:
            r, c = queue.pop(0)
            for nr, nc in sim._neighbors(r, c):
                nkey = sim._cell_key(nr, nc)
                if nkey in visited:
                    continue
                ncell = sim.grid[nkey]
                # Can traverse if cell has a road or is a residential endpoint
                if "road" in ncell.infrastructure or ncell.zone_type == "residential":
                    visited.add(nkey)
                    queue.append((nr, nc))

        connected = residential_keys & visited
        return len(connected) / len(residential_keys)


# ── Sub-rubric 2: Resident Welfare ──────────────────────────────────────────

class ResidentWelfareRubric(Rubric):
    """
    Composite welfare metric across all residential cells:
      mean(1 - congestion, 1 - school_load_capped, 1 - flood_risk)

    Higher is better.  A perfectly healthy city scores 1.0.
    """

    def score(self, sim: CitySimulation) -> float:
        scores: list[float] = []
        for cell in sim.grid.values():
            if cell.zone_type != "residential":
                continue
            # Cap school_load at 1.0 for scoring (anything above is equally bad)
            s_load = min(cell.school_load, 1.0)
            welfare = (
                (1.0 - cell.congestion)
                + (1.0 - s_load)
                + (1.0 - cell.flood_risk)
            ) / 3.0
            scores.append(welfare)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)


# ── Sub-rubric 3: Economic Viability ────────────────────────────────────────

class EconomicViabilityRubric(Rubric):
    """
    Measures tax-base potential: commercial density × proximity to residential.

    For each commercial cell, check 4-connected neighbors for residential
    cells.  Score = normalized sum of (density × adjacency_bonus).
    """

    def score(self, sim: CitySimulation) -> float:
        raw_score = 0.0
        max_possible = 0.0  # theoretical max if all cells were commercial d=3

        for r in range(sim.grid_size):
            for c in range(sim.grid_size):
                key = sim._cell_key(r, c)
                cell = sim.grid[key]
                max_possible += 3.0  # max density

                if cell.zone_type != "commercial":
                    continue

                # Count residential neighbors → adjacency bonus
                adj_bonus = 0
                for nr, nc in sim._neighbors(r, c):
                    nkey = sim._cell_key(nr, nc)
                    if sim.grid[nkey].zone_type == "residential":
                        adj_bonus += 1

                raw_score += cell.density * (1 + adj_bonus)

        if max_possible == 0:
            return 0.0
        return min(1.0, raw_score / (max_possible * 0.1))


# ── Sub-rubric 4: Budget Efficiency ─────────────────────────────────────────

class BudgetEfficiencyRubric(Rubric):
    """
    Reward for achieving welfare gains without overspending.

    Score = welfare_score / (initial_budget - remaining_budget + 1)
    Normalized to 0–1 range.
    """

    def __init__(self) -> None:
        self._welfare = ResidentWelfareRubric()

    def score(self, sim: CitySimulation) -> float:
        welfare = self._welfare.score(sim)
        spent = max(sim.initial_population, 1)  # proxy: use initial_pop as normalizer
        # Use actual budget spend
        budget_spent = max(1, (sim.initial_population * 10) - sim.budget + 1)
        # Normalize: high welfare with low spend → high score
        raw = welfare / (1.0 + budget_spent / 10000.0)
        return min(1.0, raw)


# ── Sub-rubric 5: Long-Horizon Coherence ────────────────────────────────────

class LongHorizonCoherenceRubric(Rubric):
    """
    Penalizes contradictory decisions — e.g. industrial zone adjacent to a
    school, or residential next to heavy industrial without a green buffer.

    Score = 1 - (contradiction_count / max(total_placements, 1))

    Contradiction rules (adjacency constraints):
      - industrial adjacent to school → contradiction
      - industrial adjacent to residential (no green buffer) → contradiction
      - high-density residential with no road access → contradiction
    """

    # Adjacency pairs that count as contradictions
    CONTRADICTION_RULES: list[tuple[str, str]] = [
        ("industrial", "school"),      # industrial cell next to cell with school
        ("industrial", "residential"), # industrial next to residential (no buffer)
    ]

    def score(self, sim: CitySimulation) -> float:
        contradictions = 0
        total_placements = 0

        for r in range(sim.grid_size):
            for c in range(sim.grid_size):
                key = sim._cell_key(r, c)
                cell = sim.grid[key]

                if cell.zone_type == "empty":
                    continue
                total_placements += 1

                for nr, nc in sim._neighbors(r, c):
                    nkey = sim._cell_key(nr, nc)
                    ncell = sim.grid[nkey]

                    # Check industrial-school contradiction
                    if (
                        cell.zone_type == "industrial"
                        and "school" in ncell.infrastructure
                    ):
                        contradictions += 1

                    # Check industrial-residential without green buffer
                    if (
                        cell.zone_type == "industrial"
                        and ncell.zone_type == "residential"
                    ):
                        # Check if any neighbor of the residential cell is green
                        has_buffer = any(
                            sim.grid[sim._cell_key(br, bc)].zone_type == "green"
                            for br, bc in sim._neighbors(nr, nc)
                        )
                        if not has_buffer:
                            contradictions += 1

                    # High-density residential without road
                    if (
                        cell.zone_type == "residential"
                        and cell.density >= 3
                        and "road" not in cell.infrastructure
                    ):
                        # Only count once per cell, not per neighbor
                        if (nr, nc) == sim._neighbors(r, c)[0]:
                            contradictions += 1

        if total_placements == 0:
            return 1.0
        return max(0.0, 1.0 - (contradictions / total_placements))


# =============================================================================
# Composite Rubric — weighted aggregation of all five sub-rubrics
# =============================================================================

class UrbanPlannerRubric:
    """
    Composite rubric that combines five sub-rubrics with weights.

    Weights (sum to 1.0):
      - connectivity:  0.25
      - welfare:       0.30
      - economic:      0.20
      - efficiency:    0.10
      - coherence:     0.15
    """

    def __init__(self) -> None:
        self.components: dict[str, tuple[Rubric, float]] = {
            "connectivity": (ConnectivityRubric(), 0.25),
            "welfare":      (ResidentWelfareRubric(), 0.30),
            "economic":     (EconomicViabilityRubric(), 0.20),
            "efficiency":   (BudgetEfficiencyRubric(), 0.10),
            "coherence":    (LongHorizonCoherenceRubric(), 0.15),
        }

    def score(self, sim: CitySimulation) -> tuple[float, dict[str, float]]:
        """
        Compute the weighted total reward and per-component breakdown.

        Returns:
            (total_reward, breakdown_dict) where breakdown_dict maps
            component name → individual score (0.0–1.0).
        """
        breakdown: dict[str, float] = {}
        total = 0.0

        for name, (rubric, weight) in self.components.items():
            component_score = rubric.score(sim)
            breakdown[name] = round(component_score, 4)
            total += weight * component_score

        return round(total, 4), breakdown


# Singleton instance for the environment to use
urban_planner_rubric = UrbanPlannerRubric()
