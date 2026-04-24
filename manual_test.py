import sys
import os
import json

# Add the parent directory to sys.path so python can find 'openenv_urban_planner'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv_urban_planner.server.urban_planner_environment import UrbanPlannerEnvironment

def main():
    env = UrbanPlannerEnvironment()
    obs = env.reset()

    print("="*60)
    print("Environment Reset")
    print(f"Initial Season: {obs.season}, Budget: {obs.budget_remaining}")
    print("="*60)

    def run_tool(tool_name, **kwargs):
        print(f"\n[ RUNNING TOOL ] {tool_name}")
        print(f"Args: {kwargs}")
        action = {"tool_name": tool_name, "arguments": kwargs}
        obs = env.step(action)
        
        result_str = obs.tool_result
        if len(result_str) > 300:
            result_str = result_str[:300] + "... [TRUNCATED]"
            
        print(f"Result : {result_str}")
        print(f"Reward : {obs.reward:.4f}")
        print(f"Done   : {obs.done}")
        return obs

    run_tool("get_city_state", region="all")
    run_tool("get_district_report", district_id=0)
    run_tool("place_zone", x=0, y=0, zone_type="residential", density=2)
    run_tool("place_infrastructure", x=0, y=0, infra_type="road")
    run_tool("allocate_budget", category="maintenance", amount=100)
    run_tool("query_residents", district_id=0)
    run_tool("query_traffic_model", origin=0, destination=1)
    run_tool("get_event_log", last_n=5)
    run_tool("get_budget_report")
    run_tool("advance_season")

    print("\n" + "="*60)
    print("Manual Testing Complete")
    print("="*60)

if __name__ == "__main__":
    main()
