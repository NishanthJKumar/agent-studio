"""Analyzes a folder of JSON files corresponding to data collected during exploration."""

import argparse
from pathlib import Path
from agent_studio.utils.json_utils import read_json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict


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

    # Plotting loop.
    tasks = list(task_config_to_successes.keys())
    short_labels = sorted([t[:3] for t in tasks])  # only first 3 chars
    success_counts = [sum(results) for results in task_config_to_successes.values()]
    failure_counts = [len(results) - sum(results) for results in task_config_to_successes.values()]

    x = np.arange(len(tasks))  # task positions
    bar_width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bolder but not gaudy colors
    success_color = "#2E8B57"  # sea green
    failure_color = "#C94C4C"  # muted brick red

    # Stacked bars with borders
    bars_success = ax.bar(
        x, success_counts, bar_width, 
        label="Success", color=success_color,
        edgecolor="black", linewidth=0.8
    )
    bars_failure = ax.bar(
        x, failure_counts, bar_width, bottom=success_counts,
        label="Failure", color=failure_color,
        edgecolor="black", linewidth=0.8
    )

    # Labels and formatting
    ax.set_ylabel("Count")
    ax.set_xlabel("Task")
    ax.set_title("Task Outcomes (stacked)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=0)
    ax.set_ylim(0, 50)  # freeze y-axis scale
    ax.legend()

    # Add count labels inside their respective segments
    for i, (s, f) in enumerate(zip(success_counts, failure_counts)):
        if s > 0:
            ax.text(x[i], s / 2, f"{s}", ha="center", va="center", 
                    color="white", fontsize=9, fontweight="bold")
        if f > 0:
            ax.text(x[i], s + f / 2, f"{f}", ha="center", va="center", 
                    color="white", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
