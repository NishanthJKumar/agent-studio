#!/bin/bash
#SBATCH --array=1-13
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --account=comem
#SBATCH --qos=h100_comem_high
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1

# Enhanced logging function
log() {
  local level=$1
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
}

# Error handling function
handle_error() {
  log "ERROR" "$1"
  log "ERROR" "Exit code: $2"
  # Print stack trace
  if [ -n "$BASH_VERSION" ]; then
    log "ERROR" "Stack trace:"
    local i=0
    while caller $i; do
      ((i++))
    done | while read line func file; do
      log "ERROR" "[$i] $file:$line $func"
    done
  fi
  exit $2
}

# Set trap for unexpected errors
trap 'handle_error "Unexpected error occurred" $?' ERR

# Define variables
USER="njkmr"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Log script start and configuration
log "INFO" "Starting batch experiment script"
log "INFO" "SLURM_JOB_ID: $SLURM_JOB_ID"
log "INFO" "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
log "INFO" "HOSTNAME: $(hostname)"
log "INFO" "USER: $USER"
log "INFO" "MODEL_NAME: $MODEL_NAME"

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

# Create a debug log file
DEBUG_FILE="qwen-nevergiveup-${JOB_NAME}-debug.log"
touch $DEBUG_FILE

# Function to log to debug file
debug_log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEBUG] $*" >> $DEBUG_FILE
}

# Redirect stdout and stderr
log "INFO" "Redirecting stdout to $OUTPUT_FILE and stderr to $ERROR_FILE"
debug_log "Script started with JOB_NAME: $JOB_NAME"
exec 1>$OUTPUT_FILE
exec 2>$ERROR_FILE

# Continue logging after redirection
log "INFO" "Output redirection complete"

# Generate random port numbers
generate_random_port() {
  local attempts=0
  local max_attempts=10

  while [ $attempts -lt $max_attempts ]; do
    PORT=$((RANDOM % 64512 + 1024))  # Ports between 1024 and 65535
    debug_log "Trying port: $PORT (attempt $((attempts+1))/$max_attempts)"

    if ! netstat -tuln | grep -q ":$PORT "; then
      debug_log "Port $PORT is available"
      echo $PORT
      return 0
    fi

    attempts=$((attempts+1))
  done

  log "ERROR" "Failed to find an available port after $max_attempts attempts"
  return 1
}

# Generate and validate all required ports
log "INFO" "Generating random ports for services"
ENV_SERVER_PORT=$(generate_random_port) || handle_error "Failed to generate ENV_SERVER_PORT" 1
VNC_PORT=$(generate_random_port) || handle_error "Failed to generate VNC_PORT" 1
API_WEB_SOCKET=$(generate_random_port) || handle_error "Failed to generate API_WEB_SOCKET" 1
API_SOCKET=$(generate_random_port) || handle_error "Failed to generate API_SOCKET" 1
SERVER_SOCKET=$(generate_random_port) || handle_error "Failed to generate SERVER_SOCKET" 1
HF_SERVER_PORT=$(generate_random_port) || handle_error "Failed to generate HF_SERVER_PORT" 1

# Log all port assignments
log "INFO" "Starting server on node $(hostname)"
log "INFO" "VNC PORT: $VNC_PORT"
log "INFO" "ENV_SERVER_PORT: $ENV_SERVER_PORT"
log "INFO" "API_WEB_SOCKET: $API_WEB_SOCKET"
log "INFO" "API_SOCKET: $API_SOCKET"
log "INFO" "SERVER_SOCKET: $SERVER_SOCKET"
log "INFO" "HF_SERVER_PORT: $HF_SERVER_PORT"

# Check system resources
log "INFO" "System resources:"
log "INFO" "Available memory: $(free -h | grep Mem | awk '{print $7}')"
log "INFO" "CPU load: $(uptime | awk -F'load average:' '{print $2}')"
log "INFO" "Disk space: $(df -h . | tail -1 | awk '{print $4}') available"

# Step 0: Launch the huggingface server depending on the model to be used.
if [[ "$MODEL_NAME" == *"gemma"* || "$MODEL_NAME" == *"Qwen"* ]]; then
  log "INFO" "LAUNCHING PRIVATE MODEL SERVER for $MODEL_NAME"

  # Check if conda is available
  if [ ! -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    handle_error "Conda initialization script not found at /home/$USER/miniconda3/etc/profile.d/conda.sh" 1
  fi

  # Activate the conda environment
  log "INFO" "Activating conda environment 'agent-studio'"
  source /home/$USER/miniconda3/etc/profile.d/conda.sh
  conda activate agent-studio

  # Check if activation was successful
  if [ $? -ne 0 ]; then
    handle_error "Failed to activate conda environment 'agent-studio'" 1
  fi

  # Check if the script exists
  if [ ! -f "scripts/huggingface_model_server.py" ]; then
    handle_error "Huggingface model server script not found at scripts/huggingface_model_server.py" 1
  fi

  # Launch the huggingface model server in the background
  log "INFO" "Launching huggingface model server on port $HF_SERVER_PORT"
  python scripts/huggingface_model_server.py --model $MODEL_NAME --port $HF_SERVER_PORT > hugging_server_output.log 2>&1 &  HUGGINGFACE_SERVER_PID=$!

  # Add these lines:
  sleep 2
  if ! ps -p $HUGGINGFACE_SERVER_PID > /dev/null; then
    log "ERROR" "Huggingface server process died immediately after launch"
    if [ -f hugging_server_output.log ]; then
      log "ERROR" "Server output:"
      cat hugging_server_output.log | while read line; do
        log "ERROR" "  $line"
      done
    fi
    handle_error "Huggingface server failed to start" 1
  fi

  if [ $? -ne 0 ]; then
    handle_error "Failed to start huggingface model server" 1
  fi

  log "INFO" "Huggingface model server launched with PID $HUGGINGFACE_SERVER_PID"
  debug_log "Huggingface server command: python scripts/huggingface_model_server.py --model $MODEL_NAME --port $HF_SERVER_PORT"

  # Wait for the server to initialize
  MAX_ATTEMPTS=45
  ATTEMPT=0
  SERVER_READY=false

  while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    log "INFO" "Checking if the server is ready (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)..."

    # Check if process is still running
    if ! ps -p $HUGGINGFACE_SERVER_PID > /dev/null; then
      log "ERROR" "Huggingface server process died unexpectedly"
      log "ERROR" "Last 20 lines of hugging_server_output.log:"
      tail -20 hugging_server_output.log | while read line; do
        log "ERROR" "  $line"
      done
      handle_error "Huggingface server process died" 1
    fi

    # Check the readiness endpoint
    CURL_OUTPUT=$(curl -s http://127.0.0.1:$HF_SERVER_PORT/ready)
    CURL_STATUS=$?

    if [ $CURL_STATUS -ne 0 ]; then
      log "WARN" "Curl command failed with status $CURL_STATUS"
      debug_log "Curl failed: $CURL_OUTPUT"
    elif echo "$CURL_OUTPUT" | grep -q '"status":"ready"'; then
      log "INFO" "Server is ready!"
      SERVER_READY=true
      break
    else
      debug_log "Server not ready yet. Response: $CURL_OUTPUT"
    fi

    ATTEMPT=$((ATTEMPT+1))
    sleep 5  # Wait before retrying
  done

  if [ "$SERVER_READY" = false ]; then
    log "ERROR" "Server did not become ready in time after $MAX_ATTEMPTS attempts"
    log "ERROR" "Last 20 lines of hugging_server_output.log:"
    tail -20 hugging_server_output.log | while read line; do
      log "ERROR" "  $line"
    done
    handle_error "Server initialization timeout" 1
  fi
fi

# Step 1: Launch the server with retry logic
log "INFO" "Starting server container..."
MAX_SERVER_RETRIES=5
SERVER_CREATED=false

# Check if the sqsh file exists
if [ ! -f "agent-studio-server.sqsh" ]; then
  handle_error "agent-studio-server.sqsh file not found" 1
fi

for ((i=1; i<=MAX_SERVER_RETRIES; i++)); do
  log "INFO" "Server creation attempt $i of $MAX_SERVER_RETRIES"

  # Clean up any existing containers
  log "INFO" "Cleaning up existing server containers"
  enroot remove -f agent-studio-server 2>&1 | tee -a $DEBUG_FILE || true
  rm -rf /opt/sunk/tmp/enroot-data/user-*/agent-studio-server 2>&1 | tee -a $DEBUG_FILE || true

  # Create the container
  log "INFO" "Creating server container"
  ENROOT_CREATE_OUTPUT=$(enroot create -n agent-studio-server agent-studio-server.sqsh 2>&1)
  ENROOT_CREATE_STATUS=$?

  debug_log "Enroot create output: $ENROOT_CREATE_OUTPUT"

  if [ $ENROOT_CREATE_STATUS -eq 0 ]; then
    log "INFO" "Server container created successfully"

    # Check if required directories exist
    for dir in "${PWD}/agent_studio" "${PWD}/eval_online_benchmarks/files" "${PWD}/supervisor_logs"; do
      if [ ! -d "$dir" ]; then
        log "WARN" "Directory $dir does not exist, this might cause issues"
      fi
    done

    # Start the container
    log "INFO" "Starting server container with ports: ENV=$ENV_SERVER_PORT, VNC=$VNC_PORT, SERVER=$SERVER_SOCKET, API_WEB=$API_WEB_SOCKET, API=$API_SOCKET"
    ENROOT_START_OUTPUT=$(enroot start --env VNC_PASSWORD=123456 \
        --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
        --env VNC_PORT=$VNC_PORT \
        --env SERVER_SOCKET=$SERVER_SOCKET \
        --env API_WEB_SOCKET=$API_WEB_SOCKET \
        --env API_SOCKET=$API_SOCKET \
        --env ENV_SERVER_PORT=$ENV_SERVER_PORT \
        --mount /dev/shm:/dev/shm  \
        --mount ${PWD}/agent_studio:/home/ubuntu/agent_studio/agent_studio \
        --mount ${PWD}/eval_online_benchmarks/files:/home/ubuntu/agent_studio/data \
        --mount ${PWD}/supervisor_logs/:/var/log \
        --root --rw agent-studio-server 2>&1 &)

    SERVER_PID=$!
    debug_log "Enroot start output: $ENROOT_START_OUTPUT"

    # Verify the process is running
    if ps -p $SERVER_PID > /dev/null; then
      log "INFO" "Server launched with PID $SERVER_PID"
      SERVER_CREATED=true
      break
    else
      log "ERROR" "Server process failed to start or died immediately"
      debug_log "Process $SERVER_PID not found after starting"
    fi
  else
    log "ERROR" "Server creation failed with status $ENROOT_CREATE_STATUS"
    debug_log "Enroot create error: $ENROOT_CREATE_OUTPUT"
    sleep 5
  fi
done

if [ "$SERVER_CREATED" = false ]; then
  handle_error "Failed to create server after $MAX_SERVER_RETRIES attempts" 1
fi

# Step 2: Wait for the server to be ready
log "INFO" "Waiting for the server to be ready..."

# More sophisticated server readiness check
SERVER_READY=false
MAX_WAIT_TIME=60  # seconds
WAIT_INTERVAL=5   # seconds
ELAPSED_TIME=0

while [ $ELAPSED_TIME -lt $MAX_WAIT_TIME ]; do
  # Check if server process is still running
  if ! ps -p $SERVER_PID > /dev/null; then
    log "ERROR" "Server process died during initialization"
    handle_error "Server process died unexpectedly" 1
  fi

  # Try to connect to the server ports
  if nc -z localhost $ENV_SERVER_PORT 2>/dev/null; then
    log "INFO" "ENV_SERVER_PORT $ENV_SERVER_PORT is open"
    if nc -z localhost $VNC_PORT 2>/dev/null; then
      log "INFO" "VNC_PORT $VNC_PORT is open"
      SERVER_READY=true
      break
    else
      debug_log "VNC_PORT $VNC_PORT is not yet open"
    fi
  else
    debug_log "ENV_SERVER_PORT $ENV_SERVER_PORT is not yet open"
  fi

  log "INFO" "Waiting for server to be ready... ($ELAPSED_TIME/$MAX_WAIT_TIME seconds)"
  sleep $WAIT_INTERVAL
  ELAPSED_TIME=$((ELAPSED_TIME + WAIT_INTERVAL))
done

if [ "$SERVER_READY" = false ]; then
  log "WARN" "Server ports not detected as open after $MAX_WAIT_TIME seconds"
  log "WARN" "Proceeding anyway, but this might cause issues"
else
  log "INFO" "Server appears to be ready"
fi

# Step 3: Launch the client with retry logic
log "INFO" "Starting client container..."
MAX_CLIENT_RETRIES=5
CLIENT_CREATED=false

# Check if the sqsh file exists
if [ ! -f "agent-studio-client.sqsh" ]; then
  handle_error "agent-studio-client.sqsh file not found" 1
fi

for ((i=1; i<=MAX_CLIENT_RETRIES; i++)); do
  log "INFO" "Client creation attempt $i of $MAX_CLIENT_RETRIES"

  # Clean up any existing containers
  log "INFO" "Cleaning up existing client containers"
  enroot remove -f agent-studio-client 2>&1 | tee -a $DEBUG_FILE || true
  rm -rf /opt/sunk/tmp/enroot-data/user-*/agent-studio-client 2>&1 | tee -a $DEBUG_FILE || true

  # Create the container
  log "INFO" "Creating client container"
  ENROOT_CREATE_OUTPUT=$(enroot create -n agent-studio-client agent-studio-client.sqsh 2>&1)
  ENROOT_CREATE_STATUS=$?

  debug_log "Client enroot create output: $ENROOT_CREATE_OUTPUT"

  if [ $ENROOT_CREATE_STATUS -eq 0 ]; then
    log "INFO" "Client container created successfully"

    # Check if server is still running before starting client
    if ! ps -p $SERVER_PID > /dev/null; then
      log "ERROR" "Server process died before client could start"
      handle_error "Server process not running" 1
    fi

    # Start the client container
    log "INFO" "Starting client container with task path: $PARTICULAR_PATH"
    log "INFO" "Command: as-online-benchmark --task_configs_path $PARTICULAR_PATH --model $MODEL_NAME --env_server_port $ENV_SERVER_PORT --prompting_approach never_giveup_check_output --vnc_port $VNC_PORT --model_server http://0.0.0.0:$HF_SERVER_PORT --remote"

    # Create a timestamp for the client log
    CLIENT_LOG="client-${JOB_NAME}-$(date +%Y%m%d-%H%M%S).log"
    log "INFO" "Client output will be logged to $CLIENT_LOG"

    # Start the client with output capture
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
        " 2>&1 | tee $CLIENT_LOG

    CLIENT_EXIT_CODE=${PIPESTATUS[0]}
    debug_log "Client exit code: $CLIENT_EXIT_CODE"

    if [ $CLIENT_EXIT_CODE -eq 0 ]; then
      log "INFO" "Client successfully connected and completed its task"
      CLIENT_CREATED=true
      break
    else
      log "ERROR" "Client execution failed with exit code $CLIENT_EXIT_CODE"
      log "ERROR" "Last 20 lines of client log:"
      tail -20 $CLIENT_LOG | while read line; do
        log "ERROR" "  $line"
      done
    fi
  else
    log "ERROR" "Client creation failed with status $ENROOT_CREATE_STATUS"
    debug_log "Client enroot create error: $ENROOT_CREATE_OUTPUT"
  fi

  log "INFO" "Client connection attempt $i failed, retrying in 5 seconds..."
  sleep 5  # Wait before retrying
done

if [ "$CLIENT_CREATED" = false ]; then
  log "ERROR" "Failed to create client after $MAX_CLIENT_RETRIES attempts"
  # We don't exit here as we still need to clean up the server
  log "WARN" "Proceeding to cleanup despite client failure"
fi

# Step 4: Clean up
log "INFO" "Starting cleanup process"

# Check if HuggingFace server was started
if [ -n "$HUGGINGFACE_SERVER_PID" ]; then
  if ps -p $HUGGINGFACE_SERVER_PID > /dev/null; then
    log "INFO" "Killing HuggingFace server process with PID $HUGGINGFACE_SERVER_PID"
    kill $HUGGINGFACE_SERVER_PID

    # Verify the process was killed
    sleep 2
    if ps -p $HUGGINGFACE_SERVER_PID > /dev/null; then
      log "WARN" "HuggingFace server process did not terminate, sending SIGKILL"
      kill -9 $HUGGINGFACE_SERVER_PID
    else
      log "INFO" "HuggingFace server process terminated successfully"
    fi
  else
    log "WARN" "HuggingFace server process $HUGGINGFACE_SERVER_PID not found, may have already terminated"
  fi
fi

# Kill the server process
if [ -n "$SERVER_PID" ]; then
  if ps -p $SERVER_PID > /dev/null; then
    log "INFO" "Killing server process with PID $SERVER_PID"
    kill $SERVER_PID

    # Verify the process was killed
    sleep 2
    if ps -p $SERVER_PID > /dev/null; then
      log "WARN" "Server process did not terminate, sending SIGKILL"
      kill -9 $SERVER_PID
    else
      log "INFO" "Server process terminated successfully"
    fi
  else
    log "WARN" "Server process $SERVER_PID not found, may have already terminated"
  fi
fi

# Clean up any remaining containers
log "INFO" "Cleaning up any remaining containers"
enroot remove -f agent-studio-server 2>/dev/null || true
enroot remove -f agent-studio-client 2>/dev/null || true

# Collect and compress logs
log "INFO" "Compressing logs for easier retrieval"
LOG_ARCHIVE="logs-${JOB_NAME}-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf $LOG_ARCHIVE $OUTPUT_FILE $ERROR_FILE $DEBUG_FILE client-${JOB_NAME}-*.log hugging_server_output.log 2>/dev/null || true
log "INFO" "Logs archived to $LOG_ARCHIVE"

# Final status report
log "INFO" "Experiment completed"
log "INFO" "Job name: $JOB_NAME"
log "INFO" "Task path: $PARTICULAR_PATH"
log "INFO" "Model: $MODEL_NAME"
log "INFO" "Log files: $OUTPUT_FILE, $ERROR_FILE, $DEBUG_FILE"
log "INFO" "Log archive: $LOG_ARCHIVE"

# Print summary to original stdout (before redirection)
echo "Experiment $JOB_NAME completed at $(date)" > /dev/tty 2>/dev/null || true
echo "Log files: $OUTPUT_FILE, $ERROR_FILE, $DEBUG_FILE" > /dev/tty 2>/dev/null || true
echo "Log archive: $LOG_ARCHIVE" > /dev/tty 2>/dev/null || true
