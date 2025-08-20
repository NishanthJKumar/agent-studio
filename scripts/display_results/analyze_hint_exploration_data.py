"""Analyzes a folder of JSON files corresponding to data collected during exploration."""

import argparse
from pathlib import Path
from agent_studio.utils.json_utils import read_json


def main():
    parser = argparse.ArgumentParser(description="Analyze a folder of JSON files corresponding to data collected during exploration")
    parser.add_argument("path", help="Path to the folder containing JSON files to analyze")
    args = parser.parse_args()
    folder_path = Path(args.path)
    # Main results loop.
    task_config_to_successes = {}
    task_config_and_plan_to_outcome = {}
    for json_file in folder_path.glob("*.json"):
        filename = json_file.stem
        task_name = filename.split("_traj")[0]
        data = read_json(json_file)
        outcome = data["outcome"]
        if task_config_to_successes.get(task_name) is None:
            task_config_to_successes[task_name] = []
        task_config_to_successes[task_name].append(outcome)
        plan = data["hint_string"]
        if task_config_and_plan_to_outcome.get((task_name, plan)) is None:
            task_config_and_plan_to_outcome[(task_name, plan)] = []
        task_config_and_plan_to_outcome[(task_name, plan)].append((outcome, json_file))


    # Results display.
    for task_name, successes in task_config_to_successes.items():
        print(f"{task_name} has {sum(successes)} successes out of {len(successes)} trials")
    print(f"Total number of successes: {sum(sum(successes) for successes in task_config_to_successes.values())}")
    print(f"Total number of trials: {sum(len(successes) for successes in task_config_to_successes.values())}")

    for (task_name, plan), outcomes in task_config_and_plan_to_outcome.items():
        outcomes_set = set(outcome[0] for outcome in outcomes)
        if len(outcomes_set) > 1:
            print()
            print(f"{task_name} has different outcomes for the same plan! {list(outcome[1] for outcome in outcomes)}")
            print()


if __name__ == "__main__":
    main()
