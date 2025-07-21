#!/bin/bash
#SBATCH --array=1-13  # Adjust the array size to match the number of tasks
#SBATCH --time=10:00:00  # Adjust the time as needed
#SBATCH --mem=500G
#SBATCH --account=siro
#SBATCH --qos=siro_high
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50

# Define variables
USER="njkmr"

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
OUTPUT_FILE="structuredplanning-withfailures-${JOB_NAME}-gpt-4o-2024-11-20.log"
ERROR_FILE="structuredplanning-withfailures-${JOB_NAME}-gpt-4o-2024-11-20.err"

# Redirect stdout and stderr
exec 1>$OUTPUT_FILE
exec 2>$ERROR_FILE


MODEL_NAME="gpt-4o-2024-11-20"

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

# Step 0: Launch the huggingface server depending on the model to be used.
if [[ "$MODEL_NAME" == *"gemma"* ]]; then
  echo "LAUNCHING PRIVATE MODEL SERVER"
  # Activate the conda environment
  source /home/$USER/miniconda3/etc/profile.d/conda.sh
  conda activate agent-studio
  # Launch the huggingface model server in the background
  python scripts/huggingface_model_server.py --port $HF_SERVER_PORT 2>&1 &
  HUGGINGFACE_SERVER_PID=$!
  echo "Huggingface model server launched with PID $HUGGINGFACE_SERVER_PID"
  # Wait for the server to initialize.
  # TODO: find a smarter way of doing this!
  sleep 200
fi

echo "Starting server on node $(hostname)..."
echo "VNC PORT: $VNC_PORT"
echo "ENV_SERVER_PORT: $ENV_SERVER_PORT"

# Step 1: Launch the server
apptainer exec --no-home --bind /dev/shm:/dev/shm --writable-tmpfs --fakeroot \
  --env VNC_PORT=$VNC_PORT \
  --env SERVER_SOCKET=$SERVER_SOCKET\
  --env API_WEB_SOCKET=$API_WEB_SOCKET\
  --env API_SOCKET=$API_SOCKET\
  --env ENV_SERVER_PORT=$ENV_SERVER_PORT\
  --bind /home/$USER/agent-studio/scripts/agent_server.py:/home/ubuntu/agent_studio/scripts/agent_server.py:ro \
  --bind /home/$USER/agent-studio/agent_studio/envs:/home/ubuntu/agent_studio/agent_studio/envs:ro \
  --bind /home/$USER/agent-studio/agent_studio/utils:/home/ubuntu/agent_studio/agent_studio/utils:ro \
  --bind /home/$USER/agent-studio/agent_studio/agent:/home/ubuntu/agent_studio/agent_studio/agent:ro \
  --bind /home/$USER/agent-studio/agent_studio/config:/home/ubuntu/agent_studio/agent_studio/config \
  --bind /home/$USER/agent-studio/eval_online_benchmarks/files:/home/ubuntu/agent_studio/data:ro \
  --bind supervisor_logs/:/var/log \
  agent-studio-server-dynamicportrouting1.sif /home/ubuntu/agent_studio/scripts/docker_startup.sh &
SERVER_PID=$!
echo "Server launched with PID $SERVER_PID"

# Step 2: Wait for the server to be ready
echo "Waiting for the server to be ready..."
sleep 10  # Adjust the sleep duration as needed

# Step 3: Launch the client with retry logic
echo "Starting client..."
RETRIES=5
for ((i=1; i<=RETRIES; i++)); do
  apptainer exec --no-home --bind /dev/shm:/dev/shm --writable-tmpfs --fakeroot \
    --bind /home/$USER/agent-studio/:/home/ubuntu/agent_studio/ \
    agent-studio-client.sif /bin/bash -c "
      cd /home/ubuntu/agent_studio
      as-online-benchmark --task_configs_path $PARTICULAR_PATH --model $MODEL_NAME --agent structured_planning --prompting_approach structured_planning_advanced_failure_examples --env_server_port $ENV_SERVER_PORT --vnc_port $VNC_PORT --remote
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
