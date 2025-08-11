"""Analyzes a folder of JSON files corresponding to data collected during exploration."""


from pathlib import Path
from agent_studio.utils.json_utils import read_json


folder_path = Path("exploration_for_finetuning_vscode")
task_config_to_successes = {}
for json_file in folder_path.glob("*.json"):
    filename = json_file.stem
    task_name = filename.split("_traj")[0]
    data = read_json(json_file)
    outcome = data["outcome"]
    if task_config_to_successes.get(task_name) is None:
        task_config_to_successes[task_name] = []
    task_config_to_successes[task_name].append(outcome)

for task_name, successes in task_config_to_successes.items():
    print(f"{task_name} has {sum(successes)} successes out of {len(successes)} trials")

print(f"Total number of successes: {sum(sum(successes) for successes in task_config_to_successes.values())}")
print(f"Total number of trials: {sum(len(successes) for successes in task_config_to_successes.values())}")



