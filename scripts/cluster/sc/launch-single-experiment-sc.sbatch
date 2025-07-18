#!/bin/bash
#SBATCH --job-name=agent_experiment
#SBATCH --output=test-qwen.log
#SBATCH --error=test-qwen.err
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00  # Adjust the time as needed
#SBATCH --mem=500G
#SBATCH --account=comem
#SBATCH --qos=h100_comem_high
#SBATCH --ntasks-per-node 1

# Define variables
USER="njkmr"
PARTICULAR_PATH="eval_online_benchmarks/tasks/single_api/os/0d5f7fef-3e79-4f86-87a8-d3435de04d9e.json"
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

echo "Starting server on node $(hostname)..."
echo "VNC PORT: $VNC_PORT"
echo "ENV_SERVER_PORT: $ENV_SERVER_PORT"

# # Step 0: Launch the huggingface server depending on the model to be used.
# if [[ "$MODEL_NAME" == *"gemma"* ]]; then
#   echo "LAUNCHING PRIVATE MODEL SERVER"
#   # Activate the conda environment
#   source /home/$USER/miniconda3/etc/profile.d/conda.sh
#   conda activate agent-studio
#   # Launch the huggingface model server in the background
#   python scripts/huggingface_model_server.py --port $HF_SERVER_PORT 2>&1 | tee hugging_server_output.log &
#   HUGGINGFACE_SERVER_PID=$!
#   echo "Huggingface model server launched with PID $HUGGINGFACE_SERVER_PID"
#   # Wait for the server to initialize
#   sleep 65
# fi

# Step 1: Launch the server
enroot create -n agent-studio-server agent-studio-server.sqsh
enroot start --env VNC_PASSWORD=123456 \
    --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
    --env VNC_PORT=$VNC_PORT \
    --env SERVER_SOCKET=$SERVER_SOCKET \
    --env API_WEB_SOCKET=$API_WEB_SOCKET \
    --env API_SOCKET=$API_SOCKET \
    --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
    --mount /dev/shm:/dev/shm  \
    --mount ${PWD}/agent_studio:/home/ubuntu/agent_studio/agent_studio \
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
enroot create -n agent-studio-client agent-studio-client.sqsh
for ((i=1; i<=RETRIES; i++)); do
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

