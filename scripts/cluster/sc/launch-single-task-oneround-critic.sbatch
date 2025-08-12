#!/bin/bash
#SBATCH --job-name=critic-oneround
#SBATCH --output=critic_oneround-vscode-trainset-eval-200.log
#SBATCH --error=critic_oneround-vscode-trainset-eval-200.err
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --account=comem
#SBATCH --qos=h100_lowest
#SBATCH --ntasks-per-node=1

# Define variables
USER="njkmr"
TASK_PATH="eval_online_benchmarks/tasks/single_gui/vscode/train"
ACTOR_MODEL_NAME="gpt-4o-2024-11-20"
CRITIC_ACTOR_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
AGENT="bilevel_planning"
PROMPTING_APPROACH="structured_planning_decoupled_bilevel"
EXP_EPISODES=20
OUTPUT_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
DATA_COLLECTION_DIR="exploration_for_finetuning_vscode_200" # TODO: change this to be more meaningful/unique later!
SAVED_WEIGHTS_DIR="exploration_for_finetuning_vscode_trained_weights" # TODO: once we turn this to run multiple rounds, we need to change this with every round
FINETUNING_SCRIPT_PATH="../agentic-value-function-finetuning/agent-studio-hint-finetuning.py"
FINETUNING_LOGGING_PATH="direct_training-vscode.log"
NUM_TRAINING_EPOCHS="646"

# Ensure there's no buffering and we actually print training details to the 
# logging files immediately.
export PYTHONUNBUFFERED=1


# Utility function to generate random port numbers
generate_random_port() {
  while :; do
    PORT=$((RANDOM % 64512 + 1024))
    if ! netstat -tuln | grep -q ":$PORT "; then
      echo $PORT
      return
    fi
  done
}

ENV_SERVER_PORT=$(generate_random_port)
VNC_PORT=$(generate_random_port)
API_WEB_SOCKET=$(generate_random_port)
API_SOCKET=$(generate_random_port)
SERVER_SOCKET=$(generate_random_port)
HF_SERVER_PORT=$(generate_random_port)
MODEL_SERVER_PORT=$(generate_random_port)

# Launch the server (will keep running throughout the workflow)
echo "Launching server..."
enroot remove agent-studio-server
enroot create -n agent-studio-server agent-studio-server.sqsh
enroot start --env VNC_PASSWORD=123456 \
    --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
    --env VNC_PORT=$VNC_PORT \
    --env SERVER_SOCKET=$SERVER_SOCKET \
    --env API_WEB_SOCKET=$API_WEB_SOCKET \
    --env API_SOCKET=$API_SOCKET \
    --mount /dev/shm:/dev/shm  \
    --mount ${PWD}/agent_studio:/home/ubuntu/agent_studio/agent_studio:rbind,ro \
    --mount ${PWD}/eval_online_benchmarks/files:/home/ubuntu/agent_studio/data:rbind,ro \
    --mount ${PWD}/supervisor_logs/:/var/log \
    --root --rw agent-studio-server &
SERVER_PID=$!
echo "Server launched with PID $SERVER_PID"

# # Launch data collection client.
# echo "Starting data collection..."
# RETRIES=5
# for ((i=1; i<=RETRIES; i++)); do
#   enroot remove agent-studio-client
#   enroot create -n agent-studio-client agent-studio-client.sqsh
#   enroot start --env VNC_PASSWORD=123456 \
#   --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
#   --env VNC_PORT=$VNC_PORT \
#   --env SERVER_SOCKET=$SERVER_SOCKET \
#   --env API_WEB_SOCKET=$API_WEB_SOCKET \
#   --env API_SOCKET=$API_SOCKET \
#   --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
#   --mount "${PWD}:/home/ubuntu/agent_studio" \
#   --root --rw agent-studio-client -c "
#       cd /home/ubuntu/agent_studio
#       python agent_studio/apps/online_exploration.py \
#         --task_configs_path $TASK_PATH \
#         --agent $AGENT \
#         --prompting_approach $PROMPTING_APPROACH \
#         --model $ACTOR_MODEL_NAME \
#         --vnc_port $VNC_PORT \
#         --env_server_port $ENV_SERVER_PORT \
#         --model_server 0.0.0.0:$MODEL_SERVER_PORT \
#         --remote \
#         --exp_episodes $EXP_EPISODES \
#         --plan_scoring_approach uniform \
#         --finetuning_data_path $DATA_COLLECTION_DIR \
#         --save_finetuning_data
#     "

#   if [ $? -eq 0 ]; then
#     echo "Data collection successfully connected and completed its task."
#     break
#   fi
#   echo "Data collection attempt $i failed, retrying..."
#   sleep 5  # Wait before retrying
# done

# echo "Starting model training on data from $DATA_COLLECTION_DIR"
# source /home/$USER/miniconda3/etc/profile.d/conda.sh
# conda activate agent-studio
# output=$(python -u $FINETUNING_SCRIPT_PATH --data_folder $DATA_COLLECTION_DIR --output_folder $SAVED_WEIGHTS_DIR --log_file $FINETUNING_LOGGING_PATH --num_train_epochs $NUM_TRAINING_EPOCHS)
# best_epoch=$(echo "$output" | grep "BEST_EPOCH=" | cut -d'=' -f2)
# echo "Model training completed!"
# echo "Best epoch: $best_epoch"

# Determine which checkpoint to use
if [ -z "$best_epoch" ]; then
  CHECKPOINT_TO_USE=$NUM_TRAINING_EPOCHS
  echo "Using default checkpoint: $CHECKPOINT_TO_USE"
else
  CHECKPOINT_TO_USE=$best_epoch
  echo "Using best epoch checkpoint: $CHECKPOINT_TO_USE"
fi

# Now that training is done, launch a server with these loaded weights.
echo "Launching server with trained model weights!"
# Launch the huggingface model server in the background
# Activate the conda environment
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate agent-studio
python scripts/huggingface_model_server.py --model $CRITIC_ACTOR_MODEL_NAME --port $HF_SERVER_PORT --model_weights_path $SAVED_WEIGHTS_DIR/checkpoint-$CHECKPOINT_TO_USE/ 2>&1 | tee hugging_server_output.log &
HUGGINGFACE_SERVER_PID=$!
echo "Huggingface model server launched with PID $HUGGINGFACE_SERVER_PID"
# Wait for the server to initialize
MAX_ATTEMPTS=100
ATTEMPT=0
SERVER_READY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  echo "Checking if the server is ready (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)..."
  # Check the readiness endpoint
  if curl -s http://127.0.0.1:$HF_SERVER_PORT/ready | grep -q '"status":"ready"'; then
    echo "Model loaded; server is ready!"
    SERVER_READY=true
    break
  fi
  ATTEMPT=$((ATTEMPT+1))
  sleep 5  # Wait before retrying
done

if [ "$SERVER_READY" = false ]; then
  echo "Server did not become ready in time. Exiting."
  exit 1
fi

# Finally, we eval these weights on the same task we collected data on!
echo "Starting client..."
RETRIES=5
for ((i=1; i<=RETRIES; i++)); do
  enroot remove agent-studio-client
  enroot create -n agent-studio-client agent-studio-client.sqsh
  enroot start --env VNC_PASSWORD=123456 \
  --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
  --env VNC_PORT=$VNC_PORT \
  --env SERVER_SOCKET=$SERVER_SOCKET \
  --env API_WEB_SOCKET=$API_WEB_SOCKET \
  --env API_SOCKET=$API_SOCKET \
  --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
  --mount ${PWD}/:/home/ubuntu/agent_studio/ \
  --root --rw agent-studio-client -c "
      cd /home/ubuntu/agent_studio
      mkdir -p data
      cp -r eval_online_benchmarks/files/* data/
      as-online-benchmark --task_configs_path $TASK_PATH \
          --model $ACTOR_MODEL_NAME --env_server_port $ENV_SERVER_PORT \
          --agent bilevel_planning --prompting_approach structured_planning_decoupled_bilevel \
          --vnc_port $VNC_PORT --model_server http://0.0.0.0:$HF_SERVER_PORT \
          --plan_scoring_approach critic --remote
    "

  if [ $? -eq 0 ]; then
    echo "Client successfully connected and completed its task."
    break
  fi
  echo "Client connection attempt $i failed, retrying..."
  sleep 5  # Wait before retrying
done

# Step 4: Clean up
echo "Killing server process with PID $SERVER_PID"
kill $SERVER_PID
echo "Experiment completed."