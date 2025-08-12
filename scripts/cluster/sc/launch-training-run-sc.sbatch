#!/bin/bash
#SBATCH --job-name=critic-oneround
#SBATCH --output=critic_oneround-vscode-train-200.log
#SBATCH --error=critic_oneround-vscode-train-200.err
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --account=comem
#SBATCH --qos=h100_comem_high
#SBATCH --ntasks-per-node=1

# Define variables
USER="njkmr"
TASK_PATH="eval_online_benchmarks/tasks/single_gui/vscode/train"
ACTOR_MODEL_NAME="gpt-4o-2024-11-20"
CRITIC_ACTOR_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
AGENT="bilevel_planning"
OUTPUT_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
DATA_COLLECTION_DIR="exploration_for_finetuning_vscode" # TODO: change this to be more meaningful/unique later!
SAVED_WEIGHTS_DIR="test_vscode_trained_weights" # TODO: once we turn this to run multiple rounds, we need to change this with every roudn
FINETUNING_SCRIPT_PATH="../agentic-value-function-finetuning/agent-studio-hint-finetuning.py"
FINETUNING_LOGGING_PATH="test-training-vscode.log"
NUM_TRAINING_EPOCHS="200"

# Ensure there's no buffering and we actually print training details to the 
# logging files immediately.
export PYTHONUNBUFFERED=1

ENV_SERVER_PORT=$(generate_random_port)
VNC_PORT=$(generate_random_port)
API_WEB_SOCKET=$(generate_random_port)
API_SOCKET=$(generate_random_port)
SERVER_SOCKET=$(generate_random_port)
HF_SERVER_PORT=$(generate_random_port)
MODEL_SERVER_PORT=$(generate_random_port)

echo "Starting model training on data from $DATA_COLLECTION_DIR"
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate agent-studio
output=$(python -u agent-studio-hint-finetuning.py --data_folder $DATA_COLLECTION_DIR --output_folder $SAVED_WEIGHTS_DIR --log_file $FINETUNING_LOGGING_PATH --num_train_epochs $NUM_TRAINING_EPOCHS)
best_epoch=$(echo "$output" | grep "BEST_EPOCH=" | cut -d'=' -f2)
echo "Model training completed!"
echo "Best epoch: $best_epoch"
