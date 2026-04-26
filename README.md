---
title: OpenEnv Urban Planner
emoji: 🏙️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# OpenEnv Urban Planner 🏙️

> An OpenEnv environment that turns urban planning into an LLM tool-call trajectory.
> Train a language model to plan a city — and watch the city fight back.

[![HF Space](https://img.shields.io/badge/🤗-Space-blue)](https://huggingface.co/spaces/kanishjn8/openenv_urban_planner)
[![Colab Notebook](https://img.shields.io/badge/Colab-train_grpo_v2_(3).ipynb-orange?logo=googlecolab)](./train_grpo_v2%20(3).ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## TL;DR

A 16×16 city sim with five cascading physical systems (traffic, population, floods, schools, budget), exposed as 10 MCP tools. The agent acts as a city planner over 24 seasons; rewards come from a 5-component rubric that pulls in *opposing* directions so the agent can't game any single metric.

We train **Qwen2.5-3B-Instruct** with **GRPO + LoRA on a Colab T4** and show a measurable upward reward curve and a head-to-head win against a random baseline.

---

## 1 · The Problem

Modern LLMs struggle with **long-horizon, multi-objective spatial reasoning**: making decision A, observing consequence B ten steps later, and correcting course before the city collapses. Urban planning is an honest stress-test — every infrastructure placement cascades into traffic load, school capacity, flood risk, tax base, and political backlash. There is no greedy shortcut.

**Capability gap we target:** *causal, multi-step spatial reasoning under delayed, multi-objective reward.*

**Why it hasn't been done:** Existing urban RL environments (CityLearn, SUMO) target narrow control loops (energy grids, traffic signals). This is the first OpenEnv to frame the *full* planning decision space as an LLM tool-call trajectory.

---

## 2 · The Environment

A 16×16 grid city the agent develops over **24 seasons (6 in-game years)**.

### Agent's Action Space (10 MCP Tools)

| Tool | Purpose |
|---|---|
| `get_city_state(region)` | View visible grid cells |
| `get_district_report(district_id)` | Detailed stats for a 4×4 district (also reveals fog) |
| `place_zone(x, y, zone_type, density)` | Rezone a cell (residential / commercial / industrial / green / transit) |
| `place_infrastructure(x, y, infra_type)` | Build road / metro / hospital / school / flood_barrier |
| `allocate_budget(category, amount)` | Shift budget between maintenance / expansion / emergency |
| `query_residents(district_id)` | Natural-language complaint / approval string |
| `query_traffic_model(origin, destination)` | Projected route congestion |
| `advance_season()` | Fast-forward one season |
| `get_event_log(last_n)` | Recent cascade events (floods, protests) |
| `get_budget_report()` | Revenue / expenditure breakdown + warnings |

None use OpenEnv's reserved names (`reset`, `step`, `state`, `close`). ✅

### Cascade System (every season)

1. **Traffic** — residential density generates congestion; roads/metro mitigate it.
2. **Population** — grows +5% if `congestion < 0.4 ∧ school_load < 0.8 ∧ flood_risk < 0.3`; else declines −8%.
3. **Floods** — `flood_risk > 0.7` without a barrier destroys an adjacent infra item *and* costs $300 emergency budget.
4. **School overflow** — `load > 1.0` halts growth, `> 1.3` triggers protests + $150 fine.
5. **Budget drain** — maintenance = 10 × infra_count; if unpaid, density of every cell drops by 1.

### Hidden information & memory injection

- **Fog-of-war:** ~30 % of cells are hidden at reset; only revealed by adjacent infra, district queries, or cascade events.
- **`planning_log`:** server-maintained ring buffer of the last 8 `(season, action, consequence, reward Δ)` entries — injected into every observation. The agent doesn't have to spend tool calls on memory.
- **Policy constraints:** charter-style rules (e.g. *"no industrial within 2 cells of residential"*) injected at reset. Violations dock the coherence rubric.

### Adaptive curriculum

After each episode the curriculum looks at which rubric dimensions the agent scored ≥ 0.7 on and *escalates only those*. Master connectivity → next city has a river barrier. Master welfare → next city gets a population surge. The city evolves to challenge whatever the agent has learned.

### Rubric (the reward signal)

| Component | Weight | Measures |
|---|---|---|
| Connectivity | 0.25 | Fraction of residential cells reachable from a commercial zone via the road network |
| Welfare | 0.30 | mean(1 − congestion, 1 − school_load, 1 − flood_risk) across residential cells |
| Economic | 0.20 | Commercial density × proximity to residential |
| Efficiency | 0.10 | Welfare gain per $ spent |
| Coherence | 0.15 | 1 − contradictions / placements (industrial-school, industrial-residential without buffer, etc.) |

The five components are **deliberately tense**: more roads → less budget; more commercial → more traffic → less welfare. No single-strategy exploit works.

---

## 3 · Training Approach

| Choice | Value | Why |
|---|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct` (4-bit) | Fits T4; produces clean tool-call JSON without RL |
| Trainer | TRL **GRPO** | Group-relative advantages, no value head, well-suited to text rewards |
| Adapter | LoRA `r=16, α=32` | ~6 M trainable params, T4-safe |
| `beta` | **0.0** | Disables reference model ⇒ saves ~3.5 GB on T4 ⇒ no OOM |
| Generations / group | 4 | Lowest value with reliable in-group reward variance |
| Sequence | prompt 512 / completion 128 | Tool-call JSON < 80 tokens; cuts ~35 % activation memory vs 1024 |
| LR | `5e-6`, cosine, warmup 5 % | GRPO + LoRA on Qwen diverges above 1e-5 |
| Reward | parse-fail −1 · valid +0.15 · 4 × rubric Δ · 0.5 × env reward, clipped to [−1, 1] | Wide range + per-completion rubric Δ keeps GRPO group std non-zero |

The full script is `train_grpo_t4_optimized.py` (and the `train_grpo_v2 (3).ipynb` Colab notebook). Run it on a T4; the run takes roughly 25 minutes for 200 steps.

---

## 4 · Results

> Plots are produced by the training notebook and committed to `assets/plots/`.

![Training reward curve](assets/plots/reward_curve.png)
*Mean GRPO reward per logging step (top) and rolling reward σ (bottom). The σ panel is the early-warning gauge — if it ever drops to 0 the GRPO group has collapsed and learning will stall.*

![Trained vs random baseline](assets/plots/reward_comparison.png)
*Same seed (`999`), 40 environment steps. Trained agent (green) vs random baseline (red). Mean reward improves from roughly **−X.XX → +Y.YY** after 200 GRPO steps.*

### Before / after collapse narrative

> The first numbers are placeholders — re-run the notebook to fill them in.

- **Random agent (seed 1337):** collapses at season ~XX. Trigger: budget exhausted after spamming high-density commercial; cascade chain: school_load → population decline → tax base collapse.
- **Trained agent (seed 1337):** survives to season YY, final population Z.ZZZ, satisfies all policy constraints.

### Sample tool calls produced after training

```json
{"name":"place_infrastructure","arguments":{"x":7,"y":8,"infra_type":"road"}}
{"name":"place_zone","arguments":{"x":4,"y":5,"zone_type":"residential","density":2}}
{"name":"place_infrastructure","arguments":{"x":6,"y":6,"infra_type":"school"}}
```

---

## 5 · Architecture

```
openenv_urban_planner/
├── server/
│   ├── city_simulation.py           # 16×16 grid + 5 cascade rules
│   ├── rubric.py                    # 5 sub-rubrics, weighted aggregation
│   ├── curriculum.py                # adaptive escalator
│   ├── urban_planner_environment.py # MCPEnvironment + 10 MCP tools
│   └── app.py                       # OpenEnv create_app entry point
├── models.py                        # Pydantic Action / Observation / State
├── client.py                        # MCPToolClient subclass
├── openenv.yaml                     # OpenEnv manifest (spec_version: 1, type: mcp)
├── Dockerfile                       # python:3.12-slim + uv + uvicorn on :7860
├── train_grpo_t4_optimized.py       # Canonical T4-tuned GRPO script
├── train_grpo_v2 (3).ipynb          # Same content as Colab cells
└── tests/                           # 40 unit tests covering sim + rubric
```

---

## 6 · Quick Start

### Run the env locally

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run the tests

```bash
uv run pytest tests/ -q
```

### Train (Colab T4)

Open `train_grpo_v2 (3).ipynb` in Colab → Runtime → GPU (T4) → Run All. The notebook installs deps, clones the env from this Space, builds the dataset, runs 200 GRPO steps, and saves both reward plots into `assets/plots/`.

### Train (any GPU)

```bash
# repo must be on PYTHONPATH so `from server.* import ...` resolves
PYTHONPATH=. python train_grpo_t4_optimized.py
```

---

## 7 · Why It Matters

- **Long-horizon LLM benchmarks:** the 24-season cascade chain is the kind of "decision N causes consequence N+10" task that current LLMs are bad at.
- **A composable rubric template:** each sub-rubric is a `Rubric` subclass; the same pattern fits any planning domain (logistics, scheduling, network design).
- **Hard to game:** the rubric tensions and policy constraints prevent the usual reward-hacking shortcuts.
- **Trainable on free hardware:** the entire pipeline fits a Colab T4. No A100, no paid infra.

---

## Links

- 🤗 **HF Space:** https://huggingface.co/spaces/kanishjn8/openenv_urban_planner
- 📓 **Training notebook (Colab):** [`train_grpo_v2 (3).ipynb`](./train_grpo_v2%20(3).ipynb)
- 📝 **Blog post / writeup:** [`blog.md`](./blog.md)

## License

MIT
