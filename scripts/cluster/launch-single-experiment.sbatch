#!/bin/bash
#SBATCH --job-name=agent_experiment
#SBATCH --output=experiment.log
#SBATCH --error=experiment.err
#SBATCH --time=10:00:00  # Adjust the time as needed

# Define variables
USER="njkmr"
PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/os/1d6765b3-b744-4aa4-8287-d14e6d3cddac.json"
MODEL_NAME="gpt-4o-2024-08-06"

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

# Modify the config file
CONFIG_FILE="/home/$USER/agent-studio/agent_studio/config/config.py"

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
      as-online-benchmark --task_configs_path $PARTICULAR_PATH --model $MODEL_NAME --env_server_port $ENV_SERVER_PORT --vnc_port $VNC_PORT --remote
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
