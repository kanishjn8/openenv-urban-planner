# A City That Fights Back: Training an LLM to Plan Cities with GRPO on a Colab T4

> *How we built `openenv_urban_planner` — an OpenEnv environment that turns urban planning into an LLM tool-call trajectory — and trained Qwen2.5-3B against it on a free Colab GPU.*

---

## 1 · Why urban planning?

Pick any benchmark on the LLM-eval leaderboard right now and you'll find one common gap: **causal reasoning over long horizons under delayed, multi-objective reward**. Models can answer trivia, write code, even navigate a single web form, but they fall apart when "the right move now" depends on a consequence that won't show up for ten steps.

Urban planning is the cleanest stress-test for that capability we could think of. Every decision — *zone this block residential, run a road through here, build a school over there* — sets off cascades:

- traffic load on adjacent cells,
- population growth or decline,
- flood damage when the rains come,
- protests when the schools fill up,
- a tax base that may or may not pay the maintenance bill in three seasons.

There is no shortcut. You can't fake planning a city. So we built one.

## 2 · The environment in one sentence

`openenv_urban_planner` is a **16×16 grid city** with **5 cascading physical systems**, a **5-component rubric**, and **10 MCP tools** that let an LLM act as the city planner over **24 seasons (6 in-game years)**.

What the agent sees each step:

```
Season 6 | Budget $4,200
Events: FLOOD at (0,12)! Infrastructure damaged. | PROTEST at (5,4): school overcrowding (load=1.4x capacity).
Policy: No industrial zones within 2 cells of residential; Emergency fund must maintain >= $500 at all times
Log: S5 place_zone({"x":3,"y":4,"zone_type":"residential","density":2}) dr=+0.04 || S5 place_infrastructure({"x":3,"y":4,"infra_type":"road"}) dr=+0.02 || S6 advance_season() dr=-0.18
Tool result: Zoned (3,4) as residential density 2. Cost: 400. Budget: 4200.
```

What it can do (the action space):

```python
{"name":"place_zone","arguments":{"x":7,"y":8,"zone_type":"residential","density":2}}
{"name":"place_infrastructure","arguments":{"x":7,"y":8,"infra_type":"school"}}
{"name":"advance_season","arguments":{}}
# … 7 more tools (city/district queries, budget, residents, traffic, events)
```

It must output **one JSON tool call per step**.

### The five cascades

Each `advance_season` runs five rules in order:

1. **Traffic.** Residential cells generate `density × 0.05` congestion; cells without road adjacency gain `+0.2`; metro adjacency gives `−0.15`.
2. **Population.** Cells with `congestion<0.4 ∧ school_load<0.8 ∧ flood_risk<0.3` grow +5 %; otherwise they shed −8 %.
3. **Floods.** `flood_risk > 0.7` without a `flood_barrier` destroys an adjacent infra item and costs **$300** emergency.
4. **School overflow.** `load > 1.0` halts growth, `> 1.3` triggers a protest + $150 fine.
5. **Budget drain.** Maintenance = 10 × infra_count; if unpaid, every cell's density drops by 1.

### The five rubrics

The reward signal is **deliberately tense** — every component pulls against another:

| Component | What it rewards | What it costs |
|---|---|---|
| **Connectivity** (0.25) | residential cells reachable via roads from a commercial zone | road budget, contradicts efficiency |
| **Welfare** (0.30) | low congestion + low school load + low flood risk | requires defensive spend, contradicts economic |
| **Economic** (0.20) | commercial density adjacent to residential | adds traffic, contradicts welfare |
| **Efficiency** (0.10) | welfare gain per $ spent | contradicts connectivity / coherence |
| **Coherence** (0.15) | no contradictory adjacencies (industrial-school, etc.) + policy compliance | constrains the action space |

This makes the rubric **hard to game**. Any monomania (spam commercial / spam roads / spam green) tanks at least one other dimension.

### Two design tricks worth stealing

- **`planning_log` injected into every observation.** A server-maintained ring of the last 8 `(season, action, consequence, reward Δ)` entries. The agent never has to call a `get_history()` tool — its own recent past is just *there*. This directly addresses the long-horizon context problem: instead of forcing the model to remember things across hundreds of input tokens, we hand it a compact, structured past.
- **Adaptive curriculum.** After each episode, whichever rubric components scored ≥ 0.7 get *escalated* in the next episode. Mastering connectivity adds a river barrier. Mastering welfare triggers a population surge. The city actually fights back.

## 3 · Training: GRPO + LoRA on a Colab T4

The training pipeline is **Unsloth + TRL GRPO + LoRA**, all on a free Colab T4 (15.6 GB VRAM). No A100. No paid infra.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=640, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
```

The interesting part is the **GRPO config**:

```python
grpo_config = GRPOConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations             = 4,
    beta                        = 0.0,        # ← see §4
    epsilon                     = 0.2,
    learning_rate               = 5e-6,
    max_grad_norm               = 1.0,
    max_prompt_length           = 512,
    max_completion_length       = 128,
    temperature                 = 1.0,
    top_p                       = 0.95,
)
```

And the **reward function** — the part that actually decides whether learning happens:

```python
def reward_fn(completions, seed=None, history=None, **kw):
    rewards = []
    for completion, s, hist_json in zip(completions, seed, history):
        parsed = parse_tool_call(completion)
        if parsed is None:
            rewards.append(-1.0); continue                    # decisive parse penalty
        env.reset(EpisodeConfig(seed=int(s), starting_budget=10_000))
        for a in json.loads(hist_json)[-3:]:
            env.step(a)                                       # replay prefix
        score_before, _ = urban_planner_rubric.score(env._sim)
        obs = env.step({"tool_name": parsed[0], "arguments": parsed[1]})
        if _is_tool_error(obs.tool_result):
            rewards.append(-0.4); continue
        score_after, _  = urban_planner_rubric.score(env._sim)
        r = 0.15 + 4.0 * (score_after - score_before) + 0.5 * obs.reward
        rewards.append(float(np.clip(r, -1.0, 1.0)))
    return rewards
```

Two design points to note:

- **The main signal is the rubric Δ.** The same starting state, scored before vs after each completion's action. *This is the term that varies across the GRPO group* (each completion picks a different action and changes the city differently), so the group's reward σ stays healthy and GRPO has a non-zero advantage to learn from.
- **One shared env reused across reward calls.** Building a fresh `UrbanPlannerEnvironment` (FastMCP server included) for every completion was a silent ~3× slowdown in our first attempt.

## 4 · The four hard problems we hit (and how we fixed them)

### Problem 1 — CUDA OOM on every run

The first version of the script tried to load Qwen2.5-7B + GRPO + 6 generations + 1024 seq_len. T4 had ~30 MB free at peak. Three things fixed this:

| Fix | VRAM saved |
|---|---|
| Drop to **Qwen2.5-3B** | ~3 GB |
| `MAX_SEQ_LENGTH = 640` (was 1024) | ~1 GB activations |
| `beta = 0.0` ⇒ TRL skips the reference model | **~3.5 GB** |

`beta = 0.0` was the single biggest win. By default GRPO loads a frozen copy of your base model to compute a KL penalty — for a 3B model in 4-bit that's ~3.5 GB of dead weight on a T4. With `beta = 0` you opt out of the KL term entirely and rely on PPO-style epsilon clipping for stability. We also added `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` which recovers another ~0.5 GB by letting the allocator reuse fragmented blocks.

### Problem 2 — Reward plateaued around 0

The first reward function looked sensible but produced almost no learning. We logged the per-step reward distribution and found:

- The reward range was a tight `[-0.30, +0.45]` band.
- The per-completion variance was almost entirely *format* variance — the rubric was the same for every completion in the group because the rubric only fires at season boundaries, every 3 tool calls. So 2 out of every 3 completions got essentially the same reward.
- `reward_std` collapsed to ≈ 0 and GRPO advantages went with it.

Three changes turned this around:

1. **Wider reward range:** `[-1, +1]` instead of a thin shaped band.
2. **Per-completion rubric Δ:** score the rubric *before and after* this completion's action. Different completions ⇒ different cities ⇒ different Δs ⇒ healthy group σ.
3. **Decisive parse penalty (-1.0):** the model learned the JSON schema in the first ~30 steps instead of half-learning it for 200 steps.

### Problem 3 — Training was glacial (~0.01 it/s)

Two silent killers:

- The reward function rebuilt `UrbanPlannerEnvironment` (which spins up a FastMCP server) for every completion — *that's tens of milliseconds per call, called ~16 times per training step.*
- `max_grad_norm = 0.05` (we inherited from a previous attempt) was clipping basically every gradient to zero, which made each step do almost nothing useful.

Reusing one env (`reset()` + replay prefix) and bumping `max_grad_norm` to `1.0` together pushed wall-clock throughput from ~0.01 it/s to ~0.10 it/s — 200 steps in under 30 minutes.

### Problem 4 — The environment had real bugs

Building the training stack uncovered a handful of environment bugs that would have invalidated any reward we measured. The biggest one:

- **The `advance_season` tool double-stepped.** When the agent called `advance_season()` on the 3rd tool call, the environment dispatched it (advancing the season once) and then *also* applied the auto-advance for the season-boundary check (advancing the season *again*). The fix is a one-liner in `_step_impl`: skip the auto-advance if the tool the agent just called is already `advance_season`.
- The `BudgetEfficiencyRubric` used `initial_population × 10` instead of `initial_budget` as its spending baseline (a leftover proxy from an earlier version) — fixed to use the actual `state.initial_budget`.
- The `ConnectivityRubric` BFS treated residential cells as bridge nodes, so a string of residential cells with no roads counted as "connected." Restricting traversal to road cells produces the score the design intended.

These fixes go in *before* the next training run; they're documented in the audit at the top of the codebase.

## 5 · Results

After 200 GRPO steps on a T4 the trained agent shows:

- **Reward curve** climbs from roughly −0.2 to +0.2 with a healthy reward σ throughout (no group collapse).
- **Head-to-head vs random baseline** (same seed, same horizon): trained agent gets roughly **3-5× higher mean reward**.
- **Format compliance:** parse-failure rate drops from > 40 % at step 0 to < 5 % by step 100.

The qualitative behavior is the satisfying part. The trained model:

- Places **roads adjacent to residential cells** before increasing density.
- **Builds schools before** density gets above 2.
- **Avoids industrial placements** near residential when policy constraints say so.
- **Calls `get_budget_report` after expensive moves** instead of at random.

(Plots: `assets/plots/reward_curve.png`, `assets/plots/reward_comparison.png`. Re-run the notebook to regenerate.)

## 6 · What we'd do next

- **Replace random teacher rollouts** with [a small SFT warm-start](./train_lora_local_small.py) using domain heuristics — a cheap way to lift the starting reward off the floor.
- **Verifier-shaped rewards.** Pull `query_residents` and `get_budget_report` outputs into the reward as feature signals (e.g. *did the agent's action remove a "Residents complain about traffic" message?*).
- **Cross-difficulty curriculum eval.** Right now we eval at fixed difficulty 1; the curriculum manager already supports difficulty 2-5 for the next experiment.
- **Train a 7B model** on a single A100 to see how far the architecture scales.

## 7 · Try it yourself

- 🤗 **HF Space:** [huggingface.co/spaces/kanishjn8/openenv_urban_planner](https://huggingface.co/spaces/kanishjn8/openenv_urban_planner)
- 📓 **Colab notebook:** [`train_grpo_v2 (3).ipynb`](./train_grpo_v2%20(3).ipynb) — open in Colab, set runtime to T4 GPU, Run All.
- 📁 **Code:** see `server/`, `models.py`, and `train_grpo_t4_optimized.py` in this repo.

We deliberately kept the environment self-contained (pure Python + numpy, no external services) so it's easy to fork and modify. If you build a new sub-rubric or cascade rule we'd love to see it.

---

*Built for the OpenEnv Hackathon (India 2026), Themes 2 (Long-Horizon Planning) and 3.1 (Professional World Modeling).*
