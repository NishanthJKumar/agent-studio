#!/bin/bash
#SBATCH --array=1-20
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --account=comem
#SBATCH --qos=h100_lowest
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1

# Define variables
USER="njkmr"
ACTOR_MODEL_NAME="gpt-4o-2024-11-20"
CRITIC_ACTOR_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
AGENT="bilevel_planning"
PROMPTING_APPROACH="structured_planning_decoupled_bilevel"
EXP_EPISODES=200
DATA_COLLECTION_DIR="exploration_for_finetuning_vscode_maxxxxed"

# Create array of JSON files with full paths (in sorted order for consistency)
JSON_FILES=(
  "eval_online_benchmarks/tasks/single_gui/vscode/train/083bee5b-24c1-43f9-9644-d1d04926000b.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/08d6db0f-8b06-41ae-bb20-26250e0a760f.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/0d38e311-29b5-4925-a480-14a6a82836c8.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/1c0ab6a9-2bde-49e7-9b9c-ab579ab3eab7.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/3b3098d8-6626-4d8d-a291-2ad19b73d0f2.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/3dcc6db1-7ca5-412d-b519-142724d41ef2.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/5d020bcd-acb7-46ee-a21b-3a261cede5ce.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/88f3f498-7cd0-4f3b-b7ae-40cb8e951aa7.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/8a48df09-4141-4cf9-89c6-e193e0f42451.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/8bbe5ae1-611f-474e-a334-ad56c875e4bc.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/1d3861fa-d605-48be-9337-b5188c351663.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/6034aa6c-2892-4ebd-bdfb-a7cfdddd0cbf.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/760b4347-bf5d-4633-ad6d-8047e9271fac.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/93b4281d-c6e7-4b39-8b3b-0132f1dd8615.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/a1f7f045-7938-4042-a445-fcd7d84aa2a4.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/b5b54c18-fddd-42f1-8299-00ab0a1397ae.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/c4e9503c-822e-4297-a3c5-d4e0e5ad596d.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/d2c5244a-d32b-4bc5-9cf8-616da006ee7a.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/d7a47fa4-5e5b-4028-8d4e-a031f96c67b0.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/test/df81e919-58eb-4dea-87ba-4b6a6321a9c2.json"
)

# Get the JSON file for this array task
if [ "$SLURM_ARRAY_TASK_ID" -ge 1 ] && [ "$SLURM_ARRAY_TASK_ID" -le 20 ]; then
  TASK_CONFIG_PATH="${JSON_FILES[$((SLURM_ARRAY_TASK_ID-1))]}"
  # Extract base name for job identification
  BASE_NAME=$(basename "$TASK_CONFIG_PATH" .json)
  JOB_NAME="critic-${BASE_NAME}"
else
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

# Set output and error file paths using JOB_NAME
OUTPUT_FILE="critic-datacollect-enmasse-${BASE_NAME}.log"
ERROR_FILE="critic-datacollect-enmasse-${BASE_NAME}.err"
exec 1>$OUTPUT_FILE
exec 2>$ERROR_FILE

echo "Processing task: $TASK_CONFIG_PATH"
echo "Job name: $JOB_NAME"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"

# Ensure there's no buffering and we actually print training details to the
# logging files immediately.
export PYTHONUNBUFFERED=1

# Generate random port numbers
generate_random_port() {
  while :; do
    PORT=$((RANDOM % 64512 + 1024))  # Ports between 1024 and 65535
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

echo "Ports assigned:"
echo "  ENV_SERVER_PORT: $ENV_SERVER_PORT"
echo "  VNC_PORT: $VNC_PORT"
echo "  API_WEB_SOCKET: $API_WEB_SOCKET"
echo "  API_SOCKET: $API_SOCKET"
echo "  SERVER_SOCKET: $SERVER_SOCKET"
echo "  MODEL_SERVER_PORT: $MODEL_SERVER_PORT"

# Step 1: Launch the server
# Use unique container names based on job ID and array task ID
SERVER_CONTAINER="agent-studio-server-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
CLIENT_CONTAINER="agent-studio-client-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

echo "Starting server container: $SERVER_CONTAINER"

# Remove containers if they exist (shouldn't fail if they don't)
enroot remove -f $SERVER_CONTAINER 2>/dev/null || true
enroot create -n $SERVER_CONTAINER agent-studio-server.sqsh
enroot start --env VNC_PASSWORD=123456 \
    --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
    --env VNC_PORT=$VNC_PORT \
    --env SERVER_SOCKET=$SERVER_SOCKET \
    --env API_WEB_SOCKET=$API_WEB_SOCKET \
    --env API_SOCKET=$API_SOCKET \
    --mount /dev/shm:/dev/shm  \
    --mount ${PWD}/agent_studio:/home/ubuntu/agent_studio/agent_studio:rbind,ro \
    --mount ${PWD}/eval_online_benchmarks/files:/home/ubuntu/agent_studio/data:rbind,ro \
    --root --rw $SERVER_CONTAINER &
SERVER_PID=$!
echo "Server launched with PID $SERVER_PID"

# Step 2: Wait for the server to be ready
echo "Waiting for the server to be ready..."
sleep 10  # Adjust the sleep duration as needed

# Step 3: Launch data collection client with retry logic
echo "Starting data collection client for: $TASK_CONFIG_PATH"
RETRIES=5
for ((i=1; i<=RETRIES; i++)); do
  enroot remove -f $CLIENT_CONTAINER 2>/dev/null || true
  enroot create -n $CLIENT_CONTAINER agent-studio-client.sqsh
  enroot start --env VNC_PASSWORD=123456 \
  --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
  --env VNC_PORT=$VNC_PORT \
  --env SERVER_SOCKET=$SERVER_SOCKET \
  --env API_WEB_SOCKET=$API_WEB_SOCKET \
  --env API_SOCKET=$API_SOCKET \
  --mount "${PWD}:/home/ubuntu/agent_studio" \
  --root --rw $CLIENT_CONTAINER -c "
      cd /home/ubuntu/agent_studio
      python agent_studio/apps/online_exploration.py \
        --task_configs_path $TASK_CONFIG_PATH \
        --agent $AGENT \
        --prompting_approach $PROMPTING_APPROACH \
        --model $ACTOR_MODEL_NAME \
        --vnc_port $VNC_PORT \
        --env_server_port $ENV_SERVER_PORT \
        --model_server 0.0.0.0:$MODEL_SERVER_PORT \
        --remote \
        --exp_episodes $EXP_EPISODES \
        --plan_scoring_approach uniform \
        --finetuning_data_path ${DATA_COLLECTION_DIR} \
        --save_finetuning_data
    "

  if [ $? -eq 0 ]; then
    echo "Data collection successfully completed for: $TASK_CONFIG_PATH"
    break
  fi
  echo "Data collection attempt $i failed for $TASK_CONFIG_PATH, retrying..."
  sleep 5  # Wait before retrying
done

# Step 4: Clean up
echo "Cleaning up for task: $BASE_NAME"
echo "Killing server process with PID $SERVER_PID"
kill $SERVER_PID 2>/dev/null || true

# Clean up containers
echo "Cleaning up containers..."
enroot remove -f $SERVER_CONTAINER 2>/dev/null || true
enroot remove -f $CLIENT_CONTAINER 2>/dev/null || true

# Kill Huggingface server if it was started
if [[ -n "$HUGGINGFACE_SERVER_PID" ]]; then
  echo "Killing Huggingface server with PID $HUGGINGFACE_SERVER_PID"
  kill $HUGGINGFACE_SERVER_PID 2>/dev/null || true
fi

echo "Task completed: $TASK_CONFIG_PATH"
