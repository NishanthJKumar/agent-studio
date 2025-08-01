#!/bin/bash
#SBATCH --job-name=agent_experiment
#SBATCH --output=test-gemma.log
#SBATCH --error=test-gemma.err
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00  # Adjust the time as needed
#SBATCH --mem=500G
#SBATCH --account=siro
#SBATCH --qos=siro_high
#SBATCH --ntasks-per-node 1

# Define variables
USER="njkmr"
PARTICULAR_PATH="eval_online_benchmarks/tasks/single_api/os/0d5f7fef-3e79-4f86-87a8-d3435de04d9e.json"
MODEL_NAME="google/gemma-3n-e4b-it"

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
  python scripts/huggingface_model_server.py --model $MODEL_NAME --port $HF_SERVER_PORT 2>&1 | tee hugging_server_output.log &
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
echo "Experiment completed."
