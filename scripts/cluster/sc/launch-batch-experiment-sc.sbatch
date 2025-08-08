#!/bin/bash
#SBATCH --array=9
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --account=comem
#SBATCH --qos=h100_comem_high
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1

# Define variables
USER="njkmr"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Set PARTICULAR_PATH and JOB_NAME based on the SLURM_ARRAY_TASK_ID
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/single_api"
  JOB_NAME="single_api"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/docs"
  JOB_NAME="single-gui-docs"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/gimp"
  JOB_NAME="single-gui-gimp"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/os"
  JOB_NAME="single-gui-os"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/sheets"
  JOB_NAME="single-gui-sheets"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/slides"
  JOB_NAME="single-gui-slides"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/vscode"
  JOB_NAME="single-gui-vscode"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/compositional/split0"
  JOB_NAME="compositional-split0"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/compositional/split1"
  JOB_NAME="compositional-split1"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/compositional/split2"
  JOB_NAME="compositional-split2"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 11 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/compositional/split3"
  JOB_NAME="compositional-split3"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 12 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/compositional/split4"
  JOB_NAME="compositional-split4"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 13 ]; then
  PARTICULAR_PATH="eval_online_benchmarks/tasks/compositional/split5"
  JOB_NAME="compositional-split5"
else
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

# Set output and error file paths using JOB_NAME
OUTPUT_FILE="qwen-nevergiveup-${JOB_NAME}.log"
ERROR_FILE="qwen-nevergiveup-${JOB_NAME}.err"
exec 1>$OUTPUT_FILE
exec 2>$ERROR_FILE

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

echo "Starting server on node $(hostname)..."
echo "VNC PORT: $VNC_PORT"
echo "ENV_SERVER_PORT: $ENV_SERVER_PORT"

# Step 0: Launch the huggingface server depending on the model to be used.
if [[ "$MODEL_NAME" == *"gemma"* || "$MODEL_NAME" == *"Qwen"* ]]; then
  echo "LAUNCHING PRIVATE MODEL SERVER"
  # Activate the conda environment
  source /home/$USER/miniconda3/etc/profile.d/conda.sh
  conda activate agent-studio
  # Launch the huggingface model server in the background
  python scripts/huggingface_model_server.py --model $MODEL_NAME --port $HF_SERVER_PORT 2>&1 & #| tee hugging_server_output.log &
  HUGGINGFACE_SERVER_PID=$!
  echo "Huggingface model server launched with PID $HUGGINGFACE_SERVER_PID"

  # Wait for the server to initialize
  MAX_ATTEMPTS=45
  ATTEMPT=0
  SERVER_READY=false

  while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    echo "Checking if the server is ready (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)..."
    # Check the readiness endpoint
    if curl -s http://127.0.0.1:$HF_SERVER_PORT/ready | grep -q '"status":"ready"'; then
      echo "Server is ready!"
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
fi

# Step 1: Launch the server
# Use unique container names based on job ID and array task ID
SERVER_CONTAINER="agent-studio-server-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
CLIENT_CONTAINER="agent-studio-client-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

# Remove containers if they exist (shouldn't fail if they don't)
enroot remove -f $SERVER_CONTAINER 2>/dev/null || true
enroot create -n $SERVER_CONTAINER agent-studio-server.sqsh
enroot start --env VNC_PASSWORD=123456 \
    --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
    --env VNC_PORT=$VNC_PORT \
    --env SERVER_SOCKET=$SERVER_SOCKET \
    --env API_WEB_SOCKET=$API_WEB_SOCKET \
    --env API_SOCKET=$API_SOCKET \
    --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
    --mount /dev/shm:/dev/shm  \
    --mount ${PWD}/agent_studio:/home/ubuntu/agent_studio/agent_studio:rbind,ro \
    --mount ${PWD}/eval_online_benchmarks/files:/home/ubuntu/agent_studio/data:rbind,ro \
    --mount ${PWD}/supervisor_logs/:/var/log \
    --root --rw agent-studio-server &
SERVER_PID=$!
echo "Server launched with PID $SERVER_PID"

# Step 2: Wait for the server to be ready
echo "Waiting for the server to be ready..."
sleep 10  # Adjust the sleep duration as needed

# Step 3: Launch the client with retry logic
echo "Starting client..."
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
  --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
  --mount ${PWD}/:/home/ubuntu/agent_studio/ \
  --root --rw agent-studio-client -c "
      cd /home/ubuntu/agent_studio
      mkdir -p data
      cp -r eval_online_benchmarks/files/* data/
      as-online-benchmark --task_configs_path $PARTICULAR_PATH --model $MODEL_NAME --env_server_port $ENV_SERVER_PORT --prompting_approach never_giveup_check_output --vnc_port $VNC_PORT --model_server http://0.0.0.0:$HF_SERVER_PORT --remote
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

# Clean up containers
echo "Cleaning up containers..."
enroot remove -f $SERVER_CONTAINER 2>/dev/null || true
enroot remove -f $CLIENT_CONTAINER 2>/dev/null || true

# Kill Huggingface server if it was started
if [[ -n "$HUGGINGFACE_SERVER_PID" ]]; then
  echo "Killing Huggingface server with PID $HUGGINGFACE_SERVER_PID"
  kill $HUGGINGFACE_SERVER_PID 2>/dev/null || true
fi

echo "Experiment completed."
