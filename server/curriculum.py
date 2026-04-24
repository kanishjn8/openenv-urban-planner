# =============================================================================
# OpenEnv Urban Planner — Adaptive Curriculum Manager
# =============================================================================
# After each episode, the curriculum manager analyzes which rubric dimensions
# the agent scored above 0.7 and escalates those dimensions for the next
# episode.  This means the city evolves to challenge whatever the agent has
# already learned — preventing the agent from converging on a single strategy.
#
# Escalation rules:
#   connectivity → add river barrier (splits grid, disrupting road networks)
#   welfare      → trigger population surge (+50% mid-episode)
#   economic     → add competing district (rival commercial zone spawns)
#   efficiency   → reduce starting budget (20% less to work with)
#   coherence    → inject legacy constraints (pre-placed bad infrastructure)
# =============================================================================

from __future__ import annotations

import random

from models import EpisodeConfig


class CurriculumManager:
    """
    Adaptive difficulty escalator for the OpenEnv Urban Planner.

    Tracks difficulty level (1–5) and generates episode configurations
    that target the agent's current strengths.

    Attributes:
        difficulty_level: Current difficulty tier (1=easy, 5=hardest).
                          Increases when the agent consistently scores well.
        ESCALATION_RULES: Maps rubric dimension → modifier function name.
        HIGH_SCORE_THRESHOLD: Score above which a dimension is considered
                              "mastered" and should be escalated.
    """

    # Threshold above which a rubric dimension triggers escalation
    HIGH_SCORE_THRESHOLD = 0.7

    # Maps rubric component → city modifier to apply next episode
    ESCALATION_RULES: dict[str, str] = {
        "connectivity": "add_river_barrier",       # river splits the grid
        "welfare":      "trigger_population_surge", # +50% population mid-episode
        "economic":     "add_competing_district",   # rival commercial zone spawns
        "efficiency":   "reduce_starting_budget",   # 20% less budget
        "coherence":    "inject_legacy_constraints", # pre-placed bad infrastructure
    }

    def __init__(self) -> None:
        self.difficulty_level: int = 1

    def next_episode_config(self, rubric_scores: dict[str, float]) -> EpisodeConfig:
        """
        Generate the configuration for the next training episode.

        Analyzes rubric_scores to find dimensions the agent has mastered
        (score > 0.7) and activates the corresponding escalation modifiers.
        Also increments difficulty level when many dimensions are strong.

        Args:
            rubric_scores: Dict of {rubric_name: score} from the last episode.
                           Empty dict on the very first episode.

        Returns:
            EpisodeConfig with appropriate difficulty, modifiers, and seed.
        """
        # Identify dimensions the agent is strong at
        strong_dims = [
            name
            for name, score in rubric_scores.items()
            if score > self.HIGH_SCORE_THRESHOLD
        ]

        # Collect modifiers for strong dimensions
        modifiers = [
            self.ESCALATION_RULES[dim]
            for dim in strong_dims
            if dim in self.ESCALATION_RULES
        ]

        # Increase difficulty when agent masters 3+ dimensions
        if len(strong_dims) >= 3 and self.difficulty_level < 5:
            self.difficulty_level += 1

        # Compute starting budget (may be reduced by modifier)
        base_budget = 10000
        if "reduce_starting_budget" in modifiers:
            base_budget = int(base_budget * 0.8)

        return EpisodeConfig(
            difficulty_level=self.difficulty_level,
            modifiers=modifiers,
            seed=random.randint(0, 9999),
            starting_budget=base_budget,
        )
