---
title: OpenEnv Urban Planner
emoji: 🏙️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# OpenEnv Urban Planner 🏙️

> An OpenEnv environment for training LLMs on long-horizon urban planning

## The Problem

Modern LLMs struggle with long-horizon spatial reasoning — making decision A, observing consequence B ten steps later, and correcting course. Urban planning is the perfect stress-test: every infrastructure placement cascades into traffic load, school capacity, flood risk, and economic activity across an entire district. No shortcut exists. You can't fake planning a city.

**Capability gap being trained:** Causal, multi-step spatial reasoning under delayed, multi-objective reward signals.

## The Environment

The agent acts as a city planner for a **16×16 grid** city, making sequential decisions across **24 seasons (6 in-game years)**:

- **Zoning**: Designate cells as residential, commercial, industrial, green, or transit
- **Infrastructure**: Place roads, metro lines, hospitals, schools, and flood barriers
- **Budget Management**: Allocate limited funds across maintenance, expansion, and emergencies

### Agent Tools (MCP)

| Tool | Purpose |
|---|---|
| `get_city_state` | View visible grid cells |
| `get_district_report` | Detailed stats for a 4×4 district |
| `place_zone` | Rezone a cell |
| `place_infrastructure` | Build infrastructure |
| `allocate_budget` | Shift budget between categories |
| `query_residents` | Get resident feedback |
| `query_traffic_model` | Project route congestion |
| `advance_season` | Fast-forward 1 season |
| `get_event_log` | View recent cascade events |
| `get_budget_report` | View revenue/expenditure breakdown |

### Cascade System

Every season, the city computes cascading consequences:
1. **Traffic** — residential density generates congestion; roads/metro mitigate it
2. **Population** — grows in healthy areas, declines in stressed ones
3. **Floods** — high-risk areas without barriers destroy infrastructure
4. **School Overflow** — overcrowding halts growth and triggers protests
5. **Budget Drain** — maintenance costs degrade infrastructure if unpaid

### Adaptive Curriculum

The city evolves to challenge whatever the agent has learned — mastering connectivity triggers river barriers, strong welfare management triggers population surges.

## Rubric Scoring

Five weighted sub-rubrics (sum to 1.0):

| Component | Weight | Measures |
|---|---|---|
| Connectivity | 0.25 | Road network coverage |
| Welfare | 0.30 | Low congestion + flood risk + school load |
| Economic | 0.20 | Commercial viability near residents |
| Efficiency | 0.10 | Welfare gains per budget spent |
| Coherence | 0.15 | No contradictory placements |

## Quick Start

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run server locally
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run tests
uv run pytest tests/ -q

# Validate OpenEnv compliance
uv run openenv validate --url http://localhost:8000

# Push to HF Spaces
uv run openenv push --enable-interface
```

## Results

*Training plots will be saved to `assets/plots/` after running the GRPO training notebook.*

## Why It Matters

- **Urban AI Research**: First environment framing full urban planning as an LLM tool-call trajectory
- **Long-Horizon Benchmarks**: Tests 10+ step causal reasoning chains
- **Planning Tools**: Potential real-world applications in urban development AI

## Links

- 📓 Training notebook: `notebooks/train_grpo.ipynb`

## License

MIT
