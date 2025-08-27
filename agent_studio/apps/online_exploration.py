from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import os
import sys
import threading
import time
from pathlib import Path
import copy

import jsonpickle
import numpy as np
import requests
from PyQt6.QtCore import (
    QEvent,
    QMutex,
    QObject,
    QSize,
    Qt,
    QThread,
    QTimer,
    QWaitCondition,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QImage
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

# Import Xvfb and start the display before anything else!
# isort: off
SKIP_XVFB = os.getenv("SKIP_XVFB", "false").lower() == "true"
if not SKIP_XVFB:
    from xvfbwrapper import Xvfb

    vdisplay = Xvfb()
    vdisplay.start()
else:
    vdisplay = None
# isort: on

from agent_studio.agent import setup_agent
from agent_studio.agent.base_agent import BaseAgent
from agent_studio.config.config import Config
from agent_studio.envs.desktop_env.evaluators.evaluator_helper import evaluator_router
from agent_studio.envs.desktop_env.vnc_client import (
    LocalStreamer,
    VNCFrame,
    VNCStreamer,
)
from agent_studio.utils.communication import (
    AgentStudioEvalRequest,
    AgentStudioResetRequest,
    AgentStudioStatusResponse,
    AgentStudioTextRequest,
)
from agent_studio.utils.gui import (
    ChoiceDialog,
    ChoiceDialogPython,
    InputDialog,
    JSONEditor,
    TimerLabel,
)
from agent_studio.utils.json_utils import (
    apply_env_vars,
    read_task_jsons,
    read_unfinished_tasks,
)
from agent_studio.utils.types import StepInfo, TaskConfig, VideoMeta, Message, MessageList
from agent_studio.apps.online_benchmark import FrameBuffer, WorkerSignals, TaskThread, GUI, NonGUI
from agent_studio.llm import ModelManager

config = Config()

logger = logging.getLogger(__name__)
REMOTE_SERVER_ADDR = f"http://{config.env_server_addr}:{config.env_server_port}"


def reset_task(args, task_config: TaskConfig) -> TaskConfig:
    # Get remote env_vars
    if args.remote:
        response_raw = requests.get(f"{REMOTE_SERVER_ADDR}/env_vars")
        response = AgentStudioStatusResponse(**response_raw.json())
        assert (
            response.status == "success"
        ), f"Fail to reset task: {response.message}"
        env_vars = response.message
        assert isinstance(env_vars, dict), "Invalid env_vars"
    else:
        env_vars = config.env_vars
    # logger.info(f"Env vars: {env_vars}")
    # logger.info(f"Task config before: {task_config}")
    env_vars["AS_ROOT"] = "/home/ubuntu/agent_studio"
    task_config = apply_env_vars(task_config, env_vars)
    # logger.info(f"Task config after: {task_config}")
    # Reset
    try:
        if task_config.reset_procedure is not None:
            if args.remote:
                response_raw = requests.post(
                    f"{REMOTE_SERVER_ADDR}/task/reset",
                    json=AgentStudioResetRequest(
                        procedures=task_config.reset_procedure
                    ).model_dump(),
                )
                response = AgentStudioStatusResponse(**response_raw.json())
                response = wait_finish(is_eval=False, response=response)
                logger.info(f"Reset response: {response}")
                assert (
                    response.status == "finished"
                    and response.content == "success"
                ), f"Fail to reset task: {response.message}"
            else:
                evaluators = evaluator_router(task_config)
                evaluators.reset(task_config.reset_procedure)
    except AssertionError:
        logger.error(f"Failed to reset task: {task_config.task_id}")
        raise AssertionError
    return task_config


# Code duplication due to use of global variable here.
def wait_finish(is_eval: bool, response: AgentStudioStatusResponse):
    if response.status == "finished":
        return response
    elif response.status == "wait_for_input":
        # Can't override in eval mode
        if config.need_human_confirmation and not is_eval:
            user_input = input(response.content)
        else:
            user_input = "y"
        response_raw = requests.post(
            url=f"{REMOTE_SERVER_ADDR}/task/confirm",
            json=AgentStudioTextRequest(message=user_input).model_dump(),
        )
        assert response_raw.status_code == 200
        response = AgentStudioStatusResponse(**response_raw.json())
        return wait_finish(is_eval, response)
    else:
        raise ValueError(f"Unknown status: {response.status}, {response.content}")
# End code dupe

def run_exploration(args, interface: NonGUI | None = None) -> None:
    try:
        # Setup agent
        results_dir = Path(
            f"{args.log_dir}/{args.model}/{args.agent}/{args.prompting_approach}"
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.log_model_outputs:
            (results_dir / timestamp).mkdir(parents=True, exist_ok=True)
        if args.agent == "feedback":
            agent = setup_agent(
                agent_name=args.agent,
                model=args.model,
                remote=args.remote,
                runtime_server_addr=config.env_server_addr,
                runtime_server_port=config.env_server_port,
                results_dir=results_dir / timestamp,
                feedback_model=args.feedback_model,
                prompt_approach=args.prompting_approach,
                feedback_prompt_approach=args.feedback_prompting_approach,
                restrict_to_one_step=config.restrict_to_one_step,
                model_server=args.model_server,
                extra_args={"scoring_approach": args.plan_scoring_approach, 
                    "scoring_model_name": args.plan_scoring_model_name,
                "num_unique_plan_candidates": args.exp_episodes}
            )
        else:
            agent = setup_agent(
                agent_name=args.agent,
                model=args.model,
                remote=args.remote,
                runtime_server_addr=config.env_server_addr,
                runtime_server_port=config.env_server_port,
                results_dir=results_dir / timestamp,
                restrict_to_one_step=config.restrict_to_one_step,
                prompt_approach=args.prompting_approach,
                model_server=args.model_server,
                extra_args={"scoring_approach": args.plan_scoring_approach, 
                    "scoring_model_name": args.plan_scoring_model_name,
                    "num_unique_plan_candidates": args.exp_episodes,
                }
            )

        # Setup tasks
        if args.ignore_finished:
            task_configs_json = read_unfinished_tasks(
                Path(args.task_configs_path), results_dir
            )
        else:
            task_configs_json = read_task_jsons(Path(args.task_configs_path))
        task_configs: list[TaskConfig] = []
        for task_config in task_configs_json:
            task_configs.append(TaskConfig.model_validate(task_config))

        # Exploration loop.
        for task_config in tqdm(task_configs, desc="Collecting data across tasks"):
            episode_results = []
            for episode in range(args.exp_episodes):
                logger.info(f"Start task: {task_config.task_id}, episode: {episode}")

                # Run a single episode of exploration.
                try:
                    task_config = reset_task(args, task_config)
                    instruction = task_config.instruction
                    logger.info(f"Task instruction: {instruction}")

                    # Reset the agent
                    agent.reset(task_config=task_config)
                    if task_config.visual:
                        assert (
                            interface is not None
                        ), "Interface has to be open for visual tasks."
                        interface.start_recording()

                    # Loop until the task is done or the max step is reached.
                    start_time = time.time()
                    current_step = 0
                    action_memory = []
                    init_obs = None
                    while True:
                        logger.info(f"Step {current_step}")
                        if task_config.visual:
                            assert (
                                interface is not None
                            ), "Interface has to be open for visual tasks."
                            obs = interface.get_screenshot()
                        else:
                            obs = None
                        if init_obs is None and obs is not None:
                            init_obs = copy.deepcopy(obs)
                        try:
                            step_info = agent.generate_action(
                                obs=obs, model_name=args.model
                            )
                            action = step_info.action
                            action_memory.append(action)
                        except Exception as e:
                            logger.error(f"Failed to generate action: {e}")
                            step_info = StepInfo(
                                obs=obs,
                                action="",
                                prompt=[],
                                response="",
                                unexecuted_code="",
                                info={},
                                result={},
                                timestamp=0.0,
                            )
                            action = ""

                        failure_msg: None | str = None
                        if config.need_human_confirmation and (
                            input(f"Action:\n{action}\nConfirm action (y/n): ")
                            .strip()
                            .lower()
                            != "y"
                        ):
                            failure_msg = "Cancelled by human."
                        # If the max step is reached.
                        elif current_step >= task_config.max_steps:
                            failure_msg = "Max step reached."
                        # If the time limit is reached, the action is not confirmed.
                        elif (
                            args.use_time_limit
                            and time.time() - start_time > task_config.max_time
                        ):
                            failure_msg = "Time limit reached."
                        # If the action is empty.
                        elif action == "":
                            failure_msg = "Failed to generate action."
                        # If the action is the same as the previous nine actions.
                        elif (
                            len(action_memory) >= 20
                            and action_memory[-1] == action_memory[-2] == action_memory[-3]
                        ):
                            failure_msg = "Repeated action."
                        result, done = agent.step_action(
                            failure_msg=failure_msg, step_info=step_info
                        )
                        time.sleep(config.min_action_interval)
                        if done:
                            break
                        current_step += 1
                    stop_time = time.time()

                    if not args.no_log:
                        results_json_dir = results_dir / "json_results"
                        if not results_json_dir.exists():
                            results_json_dir.mkdir(parents=True, exist_ok=True)
                        task_trajectory_path = results_json_dir / task_config.task_id
                        if not task_trajectory_path.exists():
                            task_trajectory_path.mkdir(parents=True, exist_ok=True)
                        video_meta: VideoMeta | None = None
                        if task_config.visual:
                            task_trajectory_path.mkdir(parents=True, exist_ok=True)
                            video_path = task_trajectory_path / "video.mp4"
                            assert interface is not None
                            video_meta = interface.save_video(video_path)
                            logger.info(f"Video saved to {video_path}")

                    # Evaluate
                    error_in_eval = False
                    logger.info(f"\n\n config: {task_config} \n\n")
                    if args.remote:
                        response_raw = requests.post(
                            f"{REMOTE_SERVER_ADDR}/task/eval",
                            json=AgentStudioEvalRequest(
                                procedures=task_config.eval_procedure,
                                as_kwargs=str(
                                    jsonpickle.encode({"trajectory": agent.trajectory})
                                ),
                            ).model_dump(),
                        )
                        response = AgentStudioStatusResponse(**response_raw.json())
                        response = wait_finish(is_eval=True, response=response)
                        if not (
                            response.status == "finished"
                            and isinstance(response.message, dict)  # noqa: E501
                        ):
                            logger.error(
                                f"[Caught Unhandled Error in Eval] {str(response.message)}]"
                            )
                            score = 0.0
                            feedback = "Evaluator broke for reason: " + str(
                                response.message
                            )
                            error_in_eval = True
                        else:
                            score, feedback = (
                                response.message["score"],
                                response.message["feedback"],
                            )
                    else:
                        logger.info("Start evaluation")
                        try:
                            score, feedback = evaluators(task_config.eval_procedure)
                        except Exception as e:
                            logger.error(f"[Caught Unhandled Error in Eval] {str(e)}]")
                            score = 0.0
                            feedback = "Evaluator broke for reason: " + str(e)
                            error_in_eval = True

                    if score == 1.0:
                        logger.info("[Result] (PASS)")
                    else:
                        logger.info(f"[Result] (FAIL): {feedback}")
                    episode_results.append(score == 1.0)
                    logger.info(f"Episode {episode} summary:"
                                f"\n\tScore: {score}"
                                f"\n\tFeedback: {feedback}"
                                f"\tSteps: {current_step}"
                                f"\n\tTime: {stop_time - start_time:.2f} seconds")
                    if args.save_finetuning_data:
                        agent.save_finetuning_data(score == 1.0, len(agent.trajectory), init_obs, args.finetuning_data_path)

                except Exception as e:
                    import traceback

                    logger.error(f"[Unhandled Error] {repr(e)}]")
                    traceback.print_exc()
                finally:
                    # Clean up
                    if task_config.cleanup_procedure is not None:
                        if args.remote:
                            response_raw = requests.post(
                                f"{REMOTE_SERVER_ADDR}/task/reset",
                                json=AgentStudioResetRequest(
                                    procedures=task_config.cleanup_procedure
                                ).model_dump(),
                            )
                            response = AgentStudioStatusResponse(**response_raw.json())
                            response = wait_finish(is_eval=False, response=response)
                            assert (
                                response.status == "finished"
                                and response.content == "success"
                            ), f"Fail to reset task: {response.message}"
                        else:
                            evaluators = evaluator_router(task_config)
                            evaluators.reset(task_config.cleanup_procedure)

                # Make the agent aware of this attempt at solving the task.
                if args.use_reflexion:
                    agent.prev_attempt_summaries.append(agent.construct_traj_summary(args.model, score == 1.0, feedback))
        
            logger.info(f"\n\nFinal Results Successes for task {task_config.task_id}: {episode_results}\n\n")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if interface is not None:
            interface.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--agent", type=str, default="direct", help="Agent type")
    parser.add_argument(
        "--prompting_approach", type=str, default="naive", help="Prompting approach"
    )
    parser.add_argument(
        "--feedback_prompting_approach",
        type=str,
        default="direct",
        help="Feedback prompting approach",
    )
    parser.add_argument("--task_configs_path", type=str, help="Path to the task config")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Path to save the logs",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Run with rendering (should be enabled for visual tasks)",
    )
    parser.add_argument("--remote", action="store_true", help="Run in remote mode")
    parser.add_argument(
        "--window_width", type=int, default=800, help="Width of the window"
    )
    parser.add_argument(
        "--window_height", type=int, default=600, help="Height of the window"
    )
    parser.add_argument(
        "--need_human_confirmation",
        action="store_true",
        help="Need human confirmation for actions",
    )
    parser.add_argument(
        "--use_time_limit", action="store_true", help="Use time limit for tasks"
    )
    parser.add_argument(
        "--ignore_finished", action="store_true", help="Only evaluate unfinished tasks"
    )
    parser.add_argument("--no_log", action="store_true", help="Do not log the results")
    parser.add_argument(
        "--feedback_model", type=str, help="Feedback model name", required=False
    )
    parser.add_argument(
        "--env_server_addr",
        type=str,
        default="127.0.0.1",
        help="Environment server address",
    )
    parser.add_argument(
        "--env_server_port", type=int, default=8000, help="Environment server port"
    )
    parser.add_argument("--vnc_port", type=int, default=5900, help="VNC port")
    parser.add_argument(
        "--vnc_password", type=str, default="123456", help="VNC password"
    )
    parser.add_argument(
        "--model_server", type=str, help="Model server address for RemoteProvider"
    )
    parser.add_argument(
        "--exp_episodes", type=int, default=3, help="Number of episodes for exploration"
    )
    parser.add_argument(
        "--use_reflexion", action="store_true", help="Use reflexion for exploration"
    )
    parser.add_argument(
        "--save_finetuning_data", action="store_true", help="Save trajectory info used for downstream finetuning"
    )
    parser.add_argument(
        "--finetuning_data_path",  type=str, default="finetuning_data", help="Location at which to save finetuning data"
    )
    parser.add_argument(
        "--plan_scoring_approach", type=str, default="uniform", help="Plan scoring approach for the bilevel planning approach"
    )
    parser.add_argument(
        "--plan_scoring_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Plan scoring model name"
    )
    parser.add_argument(
        "--log_model_outputs", action="store_true", help="Log model outputs"
    )
    args = parser.parse_args()
    logger.info(f"Running with args: {args}")
    assert args.task_configs_path is not None, "Task config is not set."

    config.remote = args.remote
    config.headless = not args.render
    config.need_human_confirmation = args.need_human_confirmation
    if config.remote:
        config.env_server_addr = args.env_server_addr
        config.env_server_port = args.env_server_port
        config.vnc_port = args.vnc_port
        config.vnc_password = args.vnc_password

    # Update the REMOTE_SERVER_ADDR
    global REMOTE_SERVER_ADDR
    REMOTE_SERVER_ADDR = f"http://{config.env_server_addr}:{config.env_server_port}"

    # Ensure a second screen is available.
    app = QApplication(sys.argv)
    screens = QApplication.screens()

    if not args.render:
        interface = NonGUI(
            args=args,
            remote=args.remote,
            window_width=args.window_width,
            window_height=args.window_height,
        )
        run_exploration(args, interface)
    else:
        try:
            # Create the main interface.
            interface = GUI(
                args=args,
                remote=args.remote,
                window_width=args.window_width,
                window_height=args.window_height,
            )
            interface.resize(args.window_width, args.window_height)

            if not args.remote:
                # Move window to the second screen
                second_screen = screens[1]
                geometry = second_screen.geometry()
                interface.move(geometry.topLeft())
            interface.show()

            sys.exit(app.exec())
        except asyncio.exceptions.CancelledError:
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Stop the virtual Xvfb display if it was started
        if vdisplay is not None:
            vdisplay.stop()
