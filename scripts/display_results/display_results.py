#!/usr/bin/env python3

import argparse
from pathlib import Path

from agent_studio.utils.json_utils import make_report2


def main():
    """
    Main function that parses command line arguments and runs the make_report2 function.
    Takes in a directory location for results, a task config directory, and a depth int.
    """
    parser = argparse.ArgumentParser(description="Generate a report from task results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing the results",
    )
    parser.add_argument(
        "--task_config_dir",
        type=str,
        required=True,
        help="Directory containing task configurations",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Depth for indentation in the output (default: 0)",
    )

    args = parser.parse_args()

    # Convert string paths to Path objects
    results_dir = Path(args.results_dir)
    task_config_dir = Path(args.task_config_dir)

    # Run the make_report2 function
    report = make_report2(task_config_dir, results_dir, args.depth)

    # Print the final report summary
    print("\nFinal Report Summary:")
    print(f"Average Score: {report['average_score']:.2f}")
    print(f"Total Tasks: {report['total_task_count']}")
    print(f"Finished Tasks: {report['finished_task_count']}")
    print(f"Unfinished Tasks: {report['unfinished_task_count']}")
    print(f"Successful Tasks: {report['succ_task_count']}")
    print(f"Failed Tasks: {report['fail_task_count']}")


if __name__ == "__main__":
    main()
