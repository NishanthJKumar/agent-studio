import logging
import os
import re
import subprocess
import time

import psutil

from agent_studio.envs.desktop_env.evaluators.evaluator import (
    Evaluator,
    FeedbackException,
    evaluation_handler,
    reset_handler,
)
from agent_studio.utils.human_utils import confirm_action

logger = logging.getLogger(__name__)


def find_procs_by_name(name: str) -> list[psutil.Process]:
    ls = []
    template = re.compile(name)
    for p in psutil.process_iter():
        name_, exe = "", ""
        try:
            name_ = p.name()
            exe = p.exe()
            logging.info(f"Found process: {name_} {exe}")
        except (psutil.AccessDenied, psutil.ZombieProcess, psutil.NoSuchProcess):
            continue
        if template.match(name_) or template.match(os.path.basename(exe)):
            ls.append(p)
    return ls


class ProcessEvaluator(Evaluator):
    name: str = "process"

    @evaluation_handler("match_process")
    def match_process(self, name: str) -> None:
        """
        Check if a process with the given name is running. \
                Can be a regex pattern.

        Args:
            name (str): Name of the process to check.

        Raises:
            FeedbackException: If the process is not found.
        """
        procs = find_procs_by_name(name)
        if len(procs) == 0:
            raise FeedbackException(f"Process with name {name} not found.")

    @reset_handler("create_process")
    def create_process(self, cmd: list[str], wait_for: str | None) -> None:
        """
        Create a process with the given command.

        Args:
            cmd (list[str]): Command to create the process.
            wait_for (str | None): Name of the process to wait for. \
                Can be a regex pattern. \
                If None, the function will not wait for any process.

        Raises:
            FeedbackException: If the process creation fails.
        """
        logging.info(f"Creating process with command: {cmd}")
        try:
            process = subprocess.Popen(cmd)
        except Exception as e:
            logging.error(f"Failed to create process: {e}")
            raise FeedbackException(f"Failed to create process: {e}")
        logging.info(f"Process successfully created with id: {process}")
        if wait_for is not None:
            logging.info(f"Waiting for process matching: {wait_for}")
            for process_matching_attempt_idx in range(10):
                if len(find_procs_by_name(wait_for)) > 0:
                    break
                logging.info("No matching process found, sleeping...")
                time.sleep(0.5)
            logging.info("Matching process found.")
        logging.info("Exiting process creation!")

    @reset_handler("pkill_by_name")
    def pkill_by_name(self, name: str) -> None:
        """
        Kill all processes with the given name.

        Args:
            name (str): Name pattern of the process to kill. \
                Can be a regex pattern.
        """

        def _kill_processes(procs: list[psutil.Process]) -> None:
            for proc in procs:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass

        proc_list = find_procs_by_name(name)
        message: str = "Killing processes: \n"
        for proc in proc_list:
            message += f"{proc}\n"
        if len(proc_list) > 0:
            confirm_action(message)(_kill_processes)(proc_list)
