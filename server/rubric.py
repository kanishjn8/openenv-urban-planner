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

from server.city_simulation import CitySimulation


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

    Implementation: BFS from every commercial cell that has a road,
    traversing ONLY through cells that themselves have a road.  A residential
    cell is "connected" iff at least one of its 4-neighbors is on this road
    network (i.e. it has direct road access).

    BUG-FIX #3: previous implementation let residential cells act as bridge
    nodes during BFS.  That meant a chain of residential cells with NO roads
    counted as connected, which let an agent score 0.25 of total reward by
    spamming residential alone — completely defeating the metric.
    """

    def score(self, sim: CitySimulation) -> float:
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
        if not commercial_with_road:
            return 0.0  # no commercial-with-road source → nothing reachable

        # BFS through the road network only
        road_visited: set[str] = set()
        queue: list[tuple[int, int]] = list(commercial_with_road)
        for r, c in queue:
            road_visited.add(sim._cell_key(r, c))

        while queue:
            r, c = queue.pop(0)
            for nr, nc in sim._neighbors(r, c):
                nkey = sim._cell_key(nr, nc)
                if nkey in road_visited:
                    continue
                ncell = sim.grid[nkey]
                # Strict: traverse ONLY through road cells
                if "road" in ncell.infrastructure:
                    road_visited.add(nkey)
                    queue.append((nr, nc))

        # A residential cell is "connected" iff any of its neighbors is on
        # the road network (or it itself has a road).
        connected = 0
        for rkey in residential_keys:
            r, c = (int(x) for x in rkey.split("_"))
            if "road" in sim.grid[rkey].infrastructure and rkey in road_visited:
                connected += 1
                continue
            for nr, nc in sim._neighbors(r, c):
                if sim._cell_key(nr, nc) in road_visited:
                    connected += 1
                    break

        return connected / len(residential_keys)


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

    Formal definition:
      spent = max(0, initial_budget - current_budget)
      score = welfare / (1 + spent / initial_budget)

    Bounded to [0, 1].  An untouched budget gives full welfare credit; spending
    the whole budget halves it.  Negative budget (over-draft) is clamped.

    BUG-FIX #2: previous implementation used `initial_population * 10` as the
    spend baseline (a leftover proxy from an earlier version).  That made
    efficiency depend on **population size** rather than **budget consumed**,
    which is the wrong semantics.  We now use the actual `initial_budget`
    recorded by the simulation at reset.
    """

    def __init__(self) -> None:
        self._welfare = ResidentWelfareRubric()

    def score(self, sim: CitySimulation) -> float:
        welfare = self._welfare.score(sim)
        initial_budget = max(1, getattr(sim, "initial_budget", 10000) or 10000)
        spent = max(0, initial_budget - sim.budget)
        raw = welfare / (1.0 + spent / initial_budget)
        return float(max(0.0, min(1.0, raw)))


# ── Sub-rubric 5: Long-Horizon Coherence ────────────────────────────────────

class LongHorizonCoherenceRubric(Rubric):
    """
    Penalizes contradictory decisions — e.g. industrial zone adjacent to a
    school, or residential next to heavy industrial without a green buffer.

    Score = 1 - (contradiction_count / max(total_placements, 1))

    Contradiction rules:
      A. industrial adjacent to a cell with a school → contradiction (per-edge)
      B. industrial adjacent to residential without a green buffer → contradiction (per-edge)
      C. high-density residential with no road access (at the cell itself or
         any neighbor) → contradiction (per-cell)

    BUG-FIX #5: rule C used a fragile dedup trick — `if (nr,nc) == _neighbors(r,c)[0]`
    inside the inner neighbor loop — which only worked while `_neighbors`
    returned a stable order and never skipped boundaries.  We now evaluate
    rule C once per cell, outside the inner loop, where it semantically
    belongs.
    """

    # Adjacency pairs that count as contradictions (kept for reference / tests)
    CONTRADICTION_RULES: list[tuple[str, str]] = [
        ("industrial", "school"),
        ("industrial", "residential"),
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

                # Rule C — per-cell, evaluated exactly once.
                # A high-density residential cell needs *some* road access:
                # either on the cell itself or on one of its 4-neighbors.
                if cell.zone_type == "residential" and cell.density >= 3:
                    has_road_access = "road" in cell.infrastructure or any(
                        "road" in sim.grid[sim._cell_key(nr, nc)].infrastructure
                        for nr, nc in sim._neighbors(r, c)
                    )
                    if not has_road_access:
                        contradictions += 1

                # Rules A & B — per neighbor.
                for nr, nc in sim._neighbors(r, c):
                    nkey = sim._cell_key(nr, nc)
                    ncell = sim.grid[nkey]

                    # A: industrial adjacent to school
                    if (
                        cell.zone_type == "industrial"
                        and "school" in ncell.infrastructure
                    ):
                        contradictions += 1

                    # B: industrial adjacent to residential without green buffer
                    if (
                        cell.zone_type == "industrial"
                        and ncell.zone_type == "residential"
                    ):
                        has_buffer = any(
                            sim.grid[sim._cell_key(br, bc)].zone_type == "green"
                            for br, bc in sim._neighbors(nr, nc)
                        )
                        if not has_buffer:
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
