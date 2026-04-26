# =============================================================================
# OpenEnv Urban Planner — City Simulation Engine
# =============================================================================
# Pure Python + numpy simulation of the 16×16 city grid.
# Handles: grid initialization, zone/infrastructure placement, seasonal
# cascade computation (traffic, population, floods, school overflow, budget),
# fog-of-war visibility, and terminal-state detection.
#
# Design principle: deterministic given a seed.  Same seed → same city.
# =============================================================================

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ..models import CollapseReport, EpisodeConfig, ZoneCell


# ── Constants ────────────────────────────────────────────────────────────────

VALID_ZONE_TYPES = {"empty", "residential", "commercial", "industrial", "green", "transit"}
VALID_INFRA_TYPES = {"road", "metro", "hospital", "school", "flood_barrier"}

# Cost tables (budget units)
ZONE_COSTS: dict[str, int] = {
    "residential": 200, "commercial": 300, "industrial": 250,
    "green": 100, "transit": 350, "empty": 0,
}
INFRA_COSTS: dict[str, int] = {
    "road": 150, "metro": 500, "hospital": 800,
    "school": 600, "flood_barrier": 400,
}

# Population per density level for residential zones
POP_PER_DENSITY = {0: 0, 1: 50, 2: 150, 3: 400}

# Cascade thresholds
CONGESTION_GROWTH_THRESHOLD = 0.4
SCHOOL_LOAD_GROWTH_THRESHOLD = 0.8
FLOOD_RISK_GROWTH_THRESHOLD = 0.3
FLOOD_EVENT_THRESHOLD = 0.7
SCHOOL_OVERFLOW_THRESHOLD = 1.0
SCHOOL_PROTEST_THRESHOLD = 1.3

# Maintenance cost per infrastructure item per season
MAINTENANCE_COST_PER_INFRA = 10

# Number of tool calls that constitute one season advance
STEPS_PER_SEASON = 3

# Maximum seasons per episode (6 in-game years × 4 seasons)
MAX_SEASONS = 24

# Fog-of-war: fraction of cells hidden at reset
FOG_FRACTION = 0.30

# School catchment radius (Manhattan distance)
SCHOOL_CATCHMENT_RADIUS = 3

# Emergency budget cost when a flood cascade fires
FLOOD_EMERGENCY_COST = 300

# Shaped reward values per tool call (used between season boundaries by the
# environment to keep GRPO gradient signal alive).

SHAPED_REWARDS: dict[str, float] = {
    "place_infrastructure": 0.05,
    "place_zone":           0.04,
    "allocate_budget":      0.01,
    "advance_season":       0.00,
    # info / query tools — neutral, no exploit surface
    "query_residents":      0.00,
    "get_city_state":       0.00,
    "get_budget_report":    0.00,
    "get_district_report":  0.00,
    "query_traffic_model":  0.00,
    "get_event_log":        0.00,
}

# Shaped reward penalties
SHAPED_PENALTY_INDUSTRIAL_NEAR_RESIDENTIAL = -0.03
SHAPED_PENALTY_OVERSPEND_EARLY = -0.05
SHAPED_PENALTY_POLICY_VIOLATION = -0.04

# Policy constraints by difficulty level
POLICY_CONSTRAINTS: dict[int, list[str]] = {
    1: [
        "Emergency fund must maintain >= $500 at all times",
    ],
    2: [
        "No industrial zones within 2 cells of residential",
        "Emergency fund must maintain >= $500 at all times",
    ],
    3: [
        "No industrial zones within 2 cells of residential",
        "Green space must cover >= 10% of total area by season 12",
        "Emergency fund must maintain >= $500 at all times",
    ],
    4: [
        "No industrial zones within 2 cells of residential",
        "Green space must cover >= 10% of total area by season 12",
        "Emergency fund must maintain >= $500 at all times",
        "All residential districts must have at least one school",
    ],
    5: [
        "No industrial zones within 2 cells of residential",
        "Green space must cover >= 10% of total area by season 12",
        "Emergency fund must maintain >= $500 at all times",
        "All residential districts must have at least one school",
        "Commercial zones must not exceed 30% of total area",
    ],
}


class CitySimulation:
    """
    Core simulation engine for the OpenEnv Urban Planner environment.

    Manages the 16×16 city grid, computes seasonal cascades, enforces
    fog-of-war visibility, and tracks episode-level metrics.
    """

    def __init__(self) -> None:
        self.grid_size: int = 16
        self.grid: dict[str, ZoneCell] = {}
        self.elevation: np.ndarray = np.zeros((16, 16))
        self.event_log: list[str] = []
        self.budget: int = 0
        # initial_budget is recorded at reset and used by BudgetEfficiencyRubric
        self.initial_budget: int = 0
        self.season: int = 0
        self.initial_population: int = 0
        self.rng: random.Random = random.Random(42)
        self.collapse_reason: str | None = None
        self._ascii_snapshots: dict[int, str] = {}
        self._policy_constraints: list[str] = []
        self._revenue_last_season: int = 0
        self._expenditure_last_season: int = 0

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _cell_key(row: int, col: int) -> str:
        """Convert (row, col) to the canonical grid key string."""
        return f"{row}_{col}"

    def _neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Return valid 4-connected neighbors of (row, col)."""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                result.append((nr, nc))
        return result

    def _total_population(self) -> int:
        """Sum population across all cells."""
        return sum(cell.population for cell in self.grid.values())

    def _total_infra_count(self) -> int:
        """Count total infrastructure items for maintenance cost."""
        return sum(len(cell.infrastructure) for cell in self.grid.values())

    # ── Initialization ───────────────────────────────────────────────────

    def initialize(self, config: EpisodeConfig) -> None:
        """
        Set up a fresh city from the given episode configuration.

        Steps:
          1. Seed the RNG for determinism.
          2. Generate an elevation map (Perlin-like noise via numpy).
          3. Create a blank 16×16 grid.
          4. Place 4×4 seed districts in the center with mixed zones.
          5. Apply curriculum modifiers (river barriers, legacy infra, etc.).
          6. Apply fog-of-war (hide ~30% of cells).
          7. Compute initial population and record it for collapse detection.
        """
        self.grid_size = config.grid_size
        self.rng = random.Random(config.seed)
        self.budget = config.starting_budget
        # Keep a copy of the starting budget for downstream rubric math.
        self.initial_budget = config.starting_budget
        self.season = 0
        self.event_log = []

        # --- Step 2: elevation map (seeded noise) ---
        np_rng = np.random.default_rng(config.seed)
        # Simple smoothed noise to simulate terrain elevation
        raw_noise = np_rng.random((self.grid_size, self.grid_size))
        # Smooth by averaging with neighbors (poor-man's blur)
        kernel = np.ones((3, 3)) / 9.0
        padded = np.pad(raw_noise, 1, mode="wrap")
        smoothed = np.zeros_like(raw_noise)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                smoothed[r, c] = np.sum(padded[r:r + 3, c:c + 3] * kernel)
        self.elevation = smoothed

        # --- Step 3: blank grid ---
        self.grid = {}
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                self.grid[self._cell_key(r, c)] = ZoneCell(
                    flood_risk=float(max(0.0, 0.5 - self.elevation[r, c])),
                )

        # --- Step 4: seed 4×4 districts in center ---
        center = self.grid_size // 2 - 2  # top-left of 4×4 block
        seed_types = ["residential", "commercial", "residential", "industrial"]
        idx = 0
        for r in range(center, center + 4):
            for c in range(center, center + 4):
                key = self._cell_key(r, c)
                zone = seed_types[idx % len(seed_types)]
                density = self.rng.choice([1, 2])
                self.grid[key].zone_type = zone
                self.grid[key].density = density
                self.grid[key].infrastructure = ["road"]
                if zone == "residential":
                    self.grid[key].population = POP_PER_DENSITY[density]
                idx += 1

        # --- Step 5: curriculum modifiers ---
        self._apply_modifiers(config.modifiers)

        # --- Step 6: fog-of-war ---
        all_keys = list(self.grid.keys())
        num_hidden = int(len(all_keys) * FOG_FRACTION)
        hidden_keys = self.rng.sample(all_keys, num_hidden)
        for key in hidden_keys:
            self.grid[key].visible = False

        # --- Step 7: record initial population ---
        self.initial_population = max(self._total_population(), 1)
        self.collapse_reason = None
        self._ascii_snapshots = {}

        # --- Step 8: set policy constraints based on difficulty ---
        self._policy_constraints = list(
            POLICY_CONSTRAINTS.get(config.difficulty_level, POLICY_CONSTRAINTS[1])
        )

        # --- Step 9: log initial ASCII snapshot ---
        self._ascii_snapshots[0] = self.generate_ascii_snapshot()

    def _apply_modifiers(self, modifiers: list[str]) -> None:
        """
        Apply curriculum-driven modifiers to the freshly initialized grid.

        Each modifier changes the grid in a specific way to increase
        difficulty along one rubric dimension.
        """
        for mod in modifiers:
            if mod == "add_river_barrier":
                # Place a vertical river of empty/high-flood cells down column 8
                for r in range(self.grid_size):
                    key = self._cell_key(r, 8)
                    self.grid[key].zone_type = "empty"
                    self.grid[key].density = 0
                    self.grid[key].flood_risk = 0.9
                    self.grid[key].infrastructure = []

            elif mod == "trigger_population_surge":
                # Boost population +50% in all residential cells
                for cell in self.grid.values():
                    if cell.zone_type == "residential":
                        cell.population = int(cell.population * 1.5)

            elif mod == "add_competing_district":
                # Spawn a rival commercial zone in the top-left corner
                for r in range(3):
                    for c in range(3):
                        key = self._cell_key(r, c)
                        self.grid[key].zone_type = "commercial"
                        self.grid[key].density = 3
                        self.grid[key].infrastructure = ["road"]

            elif mod == "reduce_starting_budget":
                # Cut 20% of the budget
                self.budget = int(self.budget * 0.8)

            elif mod == "inject_legacy_constraints":
                # Pre-place bad infrastructure (industrial next to schools)
                spots = [(0, 15), (15, 0), (15, 15)]
                for r, c in spots:
                    key = self._cell_key(r, c)
                    self.grid[key].zone_type = "industrial"
                    self.grid[key].density = 2
                    self.grid[key].infrastructure = ["school"]

    # ── Zone & Infrastructure Placement ──────────────────────────────────

    def place_zone(self, row: int, col: int, zone_type: str, density: int) -> str:
        """
        Rezone a cell. Deducts cost from budget.

        Returns a human-readable result string for the observation.
        """
        if zone_type not in VALID_ZONE_TYPES:
            return f"Error: invalid zone_type '{zone_type}'. Must be one of {VALID_ZONE_TYPES}."
        if not (0 <= density <= 3):
            return "Error: density must be 0–3."
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return f"Error: ({row},{col}) is out of grid bounds."

        cost = ZONE_COSTS.get(zone_type, 0) * max(density, 1)
        if cost > self.budget:
            return f"Error: insufficient budget. Need {cost}, have {self.budget}."

        key = self._cell_key(row, col)
        self.grid[key].zone_type = zone_type
        self.grid[key].density = density
        if zone_type == "residential":
            self.grid[key].population = POP_PER_DENSITY[density]
        else:
            self.grid[key].population = 0
        self.budget -= cost

        return f"Zoned ({row},{col}) as {zone_type} density {density}. Cost: {cost}. Budget: {self.budget}."

    def place_infrastructure(self, row: int, col: int, infra_type: str) -> str:
        """
        Place infrastructure in a cell. Deducts cost from budget.
        Reveals adjacent fog-of-war cells.

        Returns a human-readable result string.
        """
        if infra_type not in VALID_INFRA_TYPES:
            return f"Error: invalid infra_type '{infra_type}'. Must be one of {VALID_INFRA_TYPES}."
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return f"Error: ({row},{col}) is out of grid bounds."

        cost = INFRA_COSTS[infra_type]
        if cost > self.budget:
            return f"Error: insufficient budget. Need {cost}, have {self.budget}."

        key = self._cell_key(row, col)
        cell = self.grid[key]
        if infra_type in cell.infrastructure:
            return f"Error: {infra_type} already exists at ({row},{col})."

        cell.infrastructure.append(infra_type)
        self.budget -= cost

        # Reveal adjacent cells (fog-of-war mechanic)
        for nr, nc in self._neighbors(row, col):
            self.grid[self._cell_key(nr, nc)].visible = True

        return f"Placed {infra_type} at ({row},{col}). Cost: {cost}. Budget: {self.budget}."

    def allocate_budget(self, category: str, amount: int) -> str:
        """
        Shift budget between categories.
        Currently simplified: just validates the allocation is possible.
        """
        valid_categories = {"maintenance", "expansion", "emergency"}
        if category not in valid_categories:
            return f"Error: category must be one of {valid_categories}."
        if amount > self.budget:
            return f"Error: cannot allocate {amount}, only {self.budget} available."
        # In a full implementation this would track per-category budgets.
        return f"Allocated {amount} to {category}. Remaining budget: {self.budget}."

    # ── Query Tools (read-only, no budget cost) ──────────────────────────

    def get_city_state(self, region: str = "all") -> dict[str, Any]:
        """
        Return a JSON-serializable snapshot of all visible grid cells.
        """
        result = {}
        for key, cell in self.grid.items():
            if cell.visible:
                result[key] = cell.model_dump()
        return result

    def get_district_report(self, district_id: int) -> dict[str, Any]:
        """
        Return detailed stats for one 4×4 district.

        Districts are numbered 0–15 in row-major order (4 districts per row
        for a 16×16 grid).  Calling this reveals all cells in the district.
        """
        districts_per_row = self.grid_size // 4
        if not (0 <= district_id < districts_per_row * districts_per_row):
            return {"error": f"Invalid district_id {district_id}."}

        dr = (district_id // districts_per_row) * 4
        dc = (district_id % districts_per_row) * 4

        cells: dict[str, Any] = {}
        total_pop = 0
        total_congestion = 0.0
        total_flood = 0.0
        count = 0

        for r in range(dr, dr + 4):
            for c in range(dc, dc + 4):
                key = self._cell_key(r, c)
                cell = self.grid[key]
                # Reveal cells when district is queried
                cell.visible = True
                cells[key] = cell.model_dump()
                total_pop += cell.population
                total_congestion += cell.congestion
                total_flood += cell.flood_risk
                count += 1

        return {
            "district_id": district_id,
            "cells": cells,
            "total_population": total_pop,
            "avg_congestion": round(total_congestion / max(count, 1), 3),
            "avg_flood_risk": round(total_flood / max(count, 1), 3),
        }

    def query_residents(self, district_id: int) -> str:
        """
        Return a natural-language complaint/approval string from residents
        based on district conditions.
        """
        report = self.get_district_report(district_id)
        if "error" in report:
            return report["error"]

        messages = []
        if report["avg_congestion"] > 0.5:
            messages.append("Residents complain about heavy traffic.")
        if report["avg_flood_risk"] > 0.4:
            messages.append("People are worried about flooding.")
        if report["total_population"] == 0:
            messages.append("Nobody lives here yet.")

        if not messages:
            messages.append("Residents are generally satisfied.")

        return " ".join(messages)

    def query_traffic_model(self, origin: int, destination: int) -> str:
        """
        Return projected congestion for a route between two districts.
        Simplified model: averages congestion along the path.
        """
        districts_per_row = self.grid_size // 4
        max_id = districts_per_row * districts_per_row - 1
        if not (0 <= origin <= max_id and 0 <= destination <= max_id):
            return f"Error: district IDs must be 0–{max_id}."

        # Average congestion of both district centers
        or_row = (origin // districts_per_row) * 4 + 2
        or_col = (origin % districts_per_row) * 4 + 2
        de_row = (destination // districts_per_row) * 4 + 2
        de_col = (destination % districts_per_row) * 4 + 2

        c1 = self.grid[self._cell_key(or_row, or_col)].congestion
        c2 = self.grid[self._cell_key(de_row, de_col)].congestion
        avg = round((c1 + c2) / 2, 3)

        return f"Projected congestion from district {origin} to {destination}: {avg}"

    def get_event_log(self, last_n: int = 5) -> list[str]:
        """Return the most recent cascade events."""
        return self.event_log[-last_n:]

    # ── Season Advance & Cascade Computation ─────────────────────────────

    def advance_season(self) -> str:
        """
        Advance the simulation by one season.  Computes all cascade rules
        in order: traffic → population growth → flood events → school
        overflow → budget drain.

        Returns a summary string.
        """
        self.season += 1
        events: list[str] = []

        self._cascade_traffic()
        self._cascade_population(events)
        self._cascade_floods(events)
        self._cascade_school_overflow(events)
        self._cascade_budget_drain(events)

        self.event_log.extend(events)

        # Log ASCII snapshot at key seasons
        if self.season in (12, 24):
            self._ascii_snapshots[self.season] = self.generate_ascii_snapshot()

        summary = f"Season {self.season} advanced. Events: {events if events else 'none'}."
        return summary

    def _cascade_traffic(self) -> None:
        """
        Cascade rule 1 — TRAFFIC.
        - Residential cells generate traffic proportional to density.
        - Cells without adjacent roads gain +0.2 congestion.
        - Metro adjacency reduces congestion by 0.15.
        """
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                key = self._cell_key(r, c)
                cell = self.grid[key]
                if cell.zone_type == "empty":
                    continue

                # Check for adjacent road / metro
                has_road = "road" in cell.infrastructure
                has_metro = "metro" in cell.infrastructure
                for nr, nc in self._neighbors(r, c):
                    nkey = self._cell_key(nr, nc)
                    if "road" in self.grid[nkey].infrastructure:
                        has_road = True
                    if "metro" in self.grid[nkey].infrastructure:
                        has_metro = True

                # Residential cells generate traffic
                if cell.zone_type == "residential":
                    traffic_gen = cell.density * 0.05
                    cell.congestion = min(1.0, cell.congestion + traffic_gen)

                # No road access → congestion penalty
                if not has_road:
                    cell.congestion = min(1.0, cell.congestion + 0.2)

                # Metro reduces congestion
                if has_metro:
                    cell.congestion = max(0.0, cell.congestion - 0.15)

    def _cascade_population(self, events: list[str]) -> None:
        """
        Cascade rule 2 — POPULATION GROWTH.

        - Grow +5% if congestion < 0.4 AND school_load < 0.8 AND flood_risk < 0.3.
        - Decline -8% if any threshold exceeded.

        BUG-FIX #4: pre-school edge case.  If the city has no schools yet
        (early game), the school_load constraint would otherwise force every
        residential cell into permanent decline (school_load=1.0 sentinel for
        "no school in catchment" > 0.8 threshold).  That made the city auto-
        collapse around season 20 with zero agent action.  We now treat the
        school constraint as inactive while no schools exist anywhere in the
        city; once any school is built, normal load semantics kick in.
        """
        any_schools_built = any(
            "school" in cell.infrastructure for cell in self.grid.values()
        )

        for cell in self.grid.values():
            if cell.zone_type != "residential" or cell.population == 0:
                continue

            school_ok = (
                cell.school_load < SCHOOL_LOAD_GROWTH_THRESHOLD
                or not any_schools_built
            )
            conditions_good = (
                cell.congestion < CONGESTION_GROWTH_THRESHOLD
                and school_ok
                and cell.flood_risk < FLOOD_RISK_GROWTH_THRESHOLD
            )
            if conditions_good:
                cell.population = int(cell.population * 1.05)
            else:
                old_pop = cell.population
                cell.population = int(cell.population * 0.92)
                if old_pop > 0 and cell.population < old_pop:
                    events.append(
                        f"Population decline at cell: congestion={cell.congestion:.2f}, "
                        f"school_load={cell.school_load:.2f}, flood_risk={cell.flood_risk:.2f}"
                    )

    def _cascade_floods(self, events: list[str]) -> None:
        """
        Cascade rule 3 — FLOOD EVENTS.
        - If flood_risk > 0.7 and no flood_barrier: cascade fires.
        - Destroys 1 adjacent cell's infrastructure.
        - Adds crisis, costs emergency budget.
        - Reveals the flooded cell (fog-of-war).
        """
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                key = self._cell_key(r, c)
                cell = self.grid[key]

                if cell.flood_risk <= FLOOD_EVENT_THRESHOLD:
                    continue
                if "flood_barrier" in cell.infrastructure:
                    continue

                # Flood cascade fires!
                crisis_id = f"flood_zone_{r}_{c}"
                events.append(f"FLOOD at ({r},{c})! Infrastructure damaged.")

                # Reveal the cell
                cell.visible = True

                # Destroy one piece of infrastructure in a random adjacent cell
                neighbors = self._neighbors(r, c)
                self.rng.shuffle(neighbors)
                for nr, nc in neighbors:
                    nkey = self._cell_key(nr, nc)
                    ncell = self.grid[nkey]
                    if ncell.infrastructure:
                        removed = ncell.infrastructure.pop(0)
                        events.append(f"  → Destroyed {removed} at ({nr},{nc}).")
                        ncell.visible = True  # revealed by cascade
                        break

                # Emergency budget cost
                self.budget -= FLOOD_EMERGENCY_COST

    def _cascade_school_overflow(self, events: list[str]) -> None:
        """
        Cascade rule 4 — SCHOOL OVERFLOW.
        - Compute school_load for each cell based on catchment radius.
        - If > 1.0: halt population growth (handled in population cascade).
        - If > 1.3: protest event → budget cost + satisfaction penalty.
        """
        # First, find all school locations and their capacities
        schools: list[tuple[int, int]] = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if "school" in self.grid[self._cell_key(r, c)].infrastructure:
                    schools.append((r, c))

        # For each cell, compute school_load based on nearest school
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                key = self._cell_key(r, c)
                cell = self.grid[key]
                if cell.zone_type != "residential" or cell.population == 0:
                    cell.school_load = 0.0
                    continue

                # Find students in this cell (assume 20% of pop are students)
                students = cell.population * 0.2

                # Find nearest school within catchment
                min_dist = float("inf")
                in_catchment = False
                for sr, sc in schools:
                    dist = abs(r - sr) + abs(c - sc)  # Manhattan distance
                    if dist <= SCHOOL_CATCHMENT_RADIUS:
                        in_catchment = True
                        min_dist = min(min_dist, dist)

                if not in_catchment or not schools:
                    # No school nearby — high load (welfare drops to zero on
                    # the school axis since it's capped at 1.0 in welfare
                    # scoring), but we DO NOT trigger protests here: residents
                    # can't protest a school that doesn't exist.
                    cell.school_load = 1.0
                    continue

                # Capacity per school = 100 students; load = students / capacity
                capacity = 100.0
                cell.school_load = round(students / capacity, 2)

                # Protest only when an actual school is in catchment and is
                # overcrowded — this is a real, agent-attributable failure.
                if cell.school_load > SCHOOL_PROTEST_THRESHOLD:
                    events.append(
                        f"PROTEST at ({r},{c}): school overcrowding "
                        f"(load={cell.school_load:.1f}x capacity)."
                    )
                    self.budget -= 150  # protest budget penalty

    def _cascade_budget_drain(self, events: list[str]) -> None:
        """
        Cascade rule 5 — BUDGET DRAIN.
        - Maintenance = total_infra_count × 10 per season.
        - If budget < maintenance: infrastructure degrades (density -= 1).
        """
        maintenance = self._total_infra_count() * MAINTENANCE_COST_PER_INFRA
        if self.budget >= maintenance:
            self.budget -= maintenance
        else:
            # Can't afford maintenance — degrade infrastructure
            events.append(
                f"BUDGET SHORTFALL: maintenance costs {maintenance}, "
                f"budget is {self.budget}. Infrastructure degrading."
            )
            self.budget = 0
            # Reduce density of random cells
            for cell in self.grid.values():
                if cell.density > 0 and cell.infrastructure:
                    cell.density = max(0, cell.density - 1)

    # ── Terminal State Detection ─────────────────────────────────────────

    def is_terminal(self) -> bool:
        """
        Check if the episode should end.

        Terminal conditions:
          - 24 seasons completed (6 in-game years).
          - Population collapsed below 20% of starting population.
          - Budget went negative.
        """
        if self.season >= MAX_SEASONS:
            return True
        if self._total_population() < self.initial_population * 0.2:
            return True
        if self.budget < 0:
            return True
        return False

    # ── Visibility Helpers ───────────────────────────────────────────────

    def get_visible_grid(self) -> dict[str, dict[str, Any]]:
        """Return only the visible cells as serializable dicts."""
        return {
            key: cell.model_dump()
            for key, cell in self.grid.items()
            if cell.visible
        }

    # ── Budget Report Tool ───────────────────────────────────────────────────

    def get_budget_report(self) -> str:
        """
        Return a formatted revenue/expenditure breakdown for the current
        season.  Includes contextual warnings when costs are rising.
        """
        # Compute revenue from zones
        residential_tax = 0
        commercial_tax = 0
        for cell in self.grid.values():
            if cell.zone_type == "residential":
                residential_tax += cell.population * 1  # $1 per resident
            elif cell.zone_type == "commercial":
                commercial_tax += cell.density * 300

        total_revenue = residential_tax + commercial_tax
        maintenance = self._total_infra_count() * MAINTENANCE_COST_PER_INFRA
        emergency_costs = sum(
            FLOOD_EMERGENCY_COST
            for e in self.event_log[-5:]
            if "FLOOD" in e
        )
        total_expenditure = maintenance + emergency_costs
        net = total_revenue - total_expenditure

        # Store for reference
        self._revenue_last_season = total_revenue
        self._expenditure_last_season = total_expenditure

        report_lines = [
            f"Season {self.season} Budget Report:",
            f"  Revenue:     ${total_revenue:,}  (residential tax: ${residential_tax:,} | commercial: ${commercial_tax:,})",
            f"  Expenditure: ${total_expenditure:,}  (maintenance: ${maintenance:,} | emergency: ${emergency_costs:,})",
            f"  Net:         {'+'if net >= 0 else ''}{net:,}",
        ]

        # Contextual warnings
        if maintenance > total_revenue * 0.7:
            report_lines.append(
                f"  Warning: Maintenance costs ({maintenance}) consuming >70% of revenue. "
                "Consider reducing infrastructure or increasing tax base."
            )
        if emergency_costs > 0:
            report_lines.append(
                f"  Warning: Emergency flood response cost ${emergency_costs:,} this season."
            )
        if self.budget < 1000:
            report_lines.append(
                "  Warning: Budget critically low. Consider allocating to emergency fund."
            )

        return "\n".join(report_lines)

    # ── Collapse Reporting ──────────────────────────────────────────────────

    def generate_collapse_report(self) -> CollapseReport:
        """
        Generate a diagnostic report when the city collapses.

        Analyzes the event log to determine the proximate trigger,
        the cascade chain, and the agent's key mistake.
        """
        # Determine trigger
        if self.budget < 0:
            trigger = "Budget exhausted"
        elif self._total_population() < self.initial_population * 0.2:
            trigger = f"Population collapsed to {self._total_population()} (< 20% of {self.initial_population})"
        else:
            trigger = "Maximum seasons reached (not a collapse)"

        # Build chain from recent critical events
        chain = [
            e for e in self.event_log
            if any(kw in e for kw in ("FLOOD", "PROTEST", "SHORTFALL", "decline"))
        ][-5:]  # last 5 critical events

        # Find season of no return (first season with critical events)
        season_of_no_return = self.season
        for i, event in enumerate(self.event_log):
            if any(kw in event for kw in ("FLOOD", "SHORTFALL")):
                # Rough estimate: events are added during advance_season
                season_of_no_return = max(1, i // 2)
                break

        # Identify key mistake from early events
        agent_mistake = "No specific mistake identified"
        for event in self.event_log[:5]:
            if "industrial" in event.lower() and ("school" in event.lower() or "residential" in event.lower()):
                agent_mistake = event
                break
            if "FLOOD" in event:
                agent_mistake = f"Failed to protect against flooding: {event}"
                break

        return CollapseReport(
            trigger=trigger,
            chain=chain,
            season_of_no_return=season_of_no_return,
            agent_mistake=agent_mistake,
        )

    # ── ASCII Grid Snapshot ─────────────────────────────────────────────────

    def generate_ascii_snapshot(self) -> str:
        """
        Generate an ASCII representation of the current city grid.

        Legend:
          R=residential, C=commercial, I=industrial, G=green,
          T=transit, M=metro, F=flood, .=empty
        """
        symbol_map = {
            "residential": "R",
            "commercial":  "C",
            "industrial":  "I",
            "green":       "G",
            "transit":     "T",
            "empty":       ".",
        }
        lines = [f"S{self.season}:"]
        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                cell = self.grid[self._cell_key(r, c)]
                # Priority: flood > metro > zone type
                if cell.flood_risk > FLOOD_EVENT_THRESHOLD and "flood_barrier" not in cell.infrastructure:
                    row_chars.append("F")
                elif "metro" in cell.infrastructure:
                    row_chars.append("M")
                else:
                    row_chars.append(symbol_map.get(cell.zone_type, "?"))
            lines.append("  " + " ".join(row_chars))
        return "\n".join(lines)

    @property
    def policy_constraints(self) -> list[str]:
        """Return the active policy constraints for the current episode."""
        return list(self._policy_constraints)
