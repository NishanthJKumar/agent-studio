#!/bin/bash
#SBATCH --array=1-8
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --account=comem
#SBATCH --qos=h100_lowest
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --gres=gpu:1

############################################
# Config (user-editable)
############################################
USER="njkmr"
ACTOR_MODEL_NAME="gpt-4o-2024-11-20"
CRITIC_ACTOR_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
AGENT="bilevel_planning"
PROMPTING_APPROACH="structured_planning_decoupled_bilevel"
EXP_EPISODES=30
DATA_COLLECTION_DIR="exploration_for_finetuning_vscode_train_30-naive-round1"
PREV_PLANS_SAVE_DIR="exploration_for_finetuning_vscode_train_30-round0"
PLAN_GENERATION_APPROACH="diversity" # "top_score_similarity"
SAVED_WEIGHTS_DIR="../agentic-value-function-finetuning/train-30-finetuned-round1/"
NUM_TRAINING_EPOCHS="150"

# JSON task list (sorted for consistency)
JSON_FILES=(
  "eval_online_benchmarks/tasks/single_gui/vscode/train/08d6db0f-8b06-41ae-bb20-26250e0a760f.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/3b3098d8-6626-4d8d-a291-2ad19b73d0f2.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/8a48df09-4141-4cf9-89c6-e193e0f42451.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/0d38e311-29b5-4925-a480-14a6a82836c8.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/3dcc6db1-7ca5-412d-b519-142724d41ef2.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/8bbe5ae1-611f-474e-a334-ad56c875e4bc.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/1c0ab6a9-2bde-49e7-9b9c-ab579ab3eab7.json"
  "eval_online_benchmarks/tasks/single_gui/vscode/train/5d020bcd-acb7-46ee-a21b-3a261cede5ce.json"
  # "eval_online_benchmarks/tasks/single_gui/vscode/test/1d3861fa-d605-48be-9337-b5188c351663.json"
  # "eval_online_benchmarks/tasks/single_gui/vscode/test/93b4281d-c6e7-4b39-8b3b-0132f1dd8615.json"
  # "eval_online_benchmarks/tasks/single_gui/vscode/test/760b4347-bf5d-4633-ad6d-8047e9271fac.json"
  # "eval_online_benchmarks/tasks/single_gui/vscode/test/d2c5244a-d32b-4bc5-9cf8-616da006ee7a.json"
)

############################################
# Derived / logging setup
############################################

# Validate array bounds
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set."
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -lt 1 ] || [ "$SLURM_ARRAY_TASK_ID" -gt "${#JSON_FILES[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID (must be 1..${#JSON_FILES[@]})"
  exit 1
fi

TASK_CONFIG_PATH="${JSON_FILES[$((SLURM_ARRAY_TASK_ID-1))]}"
BASE_NAME=$(basename "$TASK_CONFIG_PATH" .json)
JOB_NAME="critic-${BASE_NAME}"

# Set output and error file paths using JOB_NAME
OUTPUT_FILE="critic-datacollect-${BASE_NAME}-round1.log"
ERROR_FILE="critic-datacollect-${BASE_NAME}-round1.err"
exec 1>"$OUTPUT_FILE"
exec 2>"$ERROR_FILE"

# Unbuffer Python so logs flush immediately
export PYTHONUNBUFFERED=1

# Basic logging helpers
echo_ts () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Prefer ss; fall back to netstat
_port_in_use () {
  local p="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$p )" | tail -n +2 | grep -q .
  else
    netstat -tuln 2>/dev/null | awk '{print $4}' | grep -qE ":$p\$"
  fi
}

generate_random_port () {
  while :; do
    local PORT=$((RANDOM % 64512 + 1024))
    if ! _port_in_use "$PORT"; then
      echo "$PORT"
      return
    fi
  done
}

# Ports
ENV_SERVER_PORT=$(generate_random_port)
VNC_PORT=$(generate_random_port)
API_WEB_SOCKET=$(generate_random_port)
API_SOCKET=$(generate_random_port)
SERVER_SOCKET=$(generate_random_port)
HF_SERVER_PORT=$(generate_random_port)
MODEL_SERVER_PORT=$(generate_random_port)

echo_ts "Processing task: $TASK_CONFIG_PATH"
echo_ts "Job name: $JOB_NAME"
echo_ts "Array task ID: $SLURM_ARRAY_TASK_ID"
echo_ts "Node: $(hostname)"
echo_ts "Ports assigned:"
echo_ts "  ENV_SERVER_PORT: $ENV_SERVER_PORT"
echo_ts "  VNC_PORT: $VNC_PORT"
echo_ts "  API_WEB_SOCKET: $API_WEB_SOCKET"
echo_ts "  API_SOCKET: $API_SOCKET"
echo_ts "  SERVER_SOCKET: $SERVER_SOCKET"
echo_ts "  MODEL_SERVER_PORT: $MODEL_SERVER_PORT"

SERVER_CONTAINER="agent-studio-server-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
CLIENT_CONTAINER="agent-studio-client-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

SERVER_PID=""
HUGGINGFACE_SERVER_PID=""

cleanup () {
  echo_ts "Cleanup starting..."
  if [ -n "$SERVER_PID" ]; then
    echo_ts "Killing server process PID=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  echo_ts "Removing containers..."
  enroot remove -f "$SERVER_CONTAINER" 2>/dev/null || true
  enroot remove -f "$CLIENT_CONTAINER" 2>/dev/null || true
  if [ -n "$HUGGINGFACE_SERVER_PID" ]; then
    echo_ts "Killing Huggingface server with PID $HUGGINGFACE_SERVER_PID"
    kill "$HUGGINGFACE_SERVER_PID" 2>/dev/null || true
  fi
  echo_ts "Cleanup finished."
}
trap cleanup EXIT INT TERM

############################################
# Health check helpers
############################################
HEALTH_URL="http://127.0.0.1:${ENV_SERVER_PORT}/health"
HF_READY_URL="http://127.0.0.1:${HF_SERVER_PORT}/ready"

check_health () {
  # Returns 0 on HTTP 200 OK and "OK" content, else non-zero
  if command -v curl >/dev/null 2>&1; then
    local HTTP_CODE
    HTTP_CODE=$(curl -sS --max-time 2 -o /tmp/health_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.body -w "%{http_code}" "$HEALTH_URL" || echo "000")
    if [ "$HTTP_CODE" = "200" ] && grep -q "OK" "/tmp/health_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.body"; then
      return 0
    fi
    return 1
  else
    # Fallback: python
    python - <<PY 2>/dev/null
import sys, urllib.request
try:
    with urllib.request.urlopen("${HEALTH_URL}", timeout=2) as r:
        body = r.read().decode("utf-8", errors="ignore")
        sys.exit(0 if r.status==200 and "OK" in body else 1)
except Exception:
    sys.exit(1)
PY
    return $?
  fi
}

wait_for_health () {
  local max_wait_secs="$1"   # e.g., 120
  local poll_every_secs="${2:-2}"
  local waited=0
  while [ "$waited" -lt "$max_wait_secs" ]; do
    if check_health; then
      echo_ts "Server health OK at $HEALTH_URL"
      return 0
    fi
    sleep "$poll_every_secs"
    waited=$((waited + poll_every_secs))
  done
  echo_ts "Server health check timed out after ${max_wait_secs}s."
  return 1
}

check_hf_server_ready () {
  # Returns 0 if HuggingFace server is ready, else non-zero
  if command -v curl >/dev/null 2>&1; then
    curl -sS --max-time 5 "$HF_READY_URL" | grep -q '"status":"ready"'
    return $?
  else
    # Fallback: python
    python - <<PY 2>/dev/null
import sys, urllib.request, json
try:
    with urllib.request.urlopen("${HF_READY_URL}", timeout=5) as r:
        body = r.read().decode("utf-8", errors="ignore")
        data = json.loads(body)
        sys.exit(0 if data.get("status") == "ready" else 1)
except Exception:
    sys.exit(1)
PY
    return $?
  fi
}

wait_for_hf_server () {
  local max_wait_secs="$1"   # e.g., 500 (100 attempts * 5 sec)
  local poll_every_secs="${2:-5}"
  local waited=0
  local attempt=1
  local max_attempts=$((max_wait_secs / poll_every_secs))

  while [ "$waited" -lt "$max_wait_secs" ]; do
    echo_ts "Checking if HF server is ready (attempt $attempt/$max_attempts)..."
    if check_hf_server_ready; then
      echo_ts "HuggingFace server is ready at $HF_READY_URL"
      return 0
    fi
    sleep "$poll_every_secs"
    waited=$((waited + poll_every_secs))
    attempt=$((attempt + 1))
  done
  echo_ts "HuggingFace server readiness check timed out after ${max_wait_secs}s."
  return 1
}

############################################
# Server lifecycle
############################################
start_server () {
  echo_ts "Starting server container: $SERVER_CONTAINER"
  enroot remove -f "$SERVER_CONTAINER" 2>/dev/null || true
  enroot create -n "$SERVER_CONTAINER" agent-studio-server.sqsh

  enroot start \
    --env VNC_PASSWORD=123456 \
    --env ENV_SERVER_PORT="$ENV_SERVER_PORT" \
    --env VNC_PORT="$VNC_PORT" \
    --env SERVER_SOCKET="$SERVER_SOCKET" \
    --env API_WEB_SOCKET="$API_WEB_SOCKET" \
    --env API_SOCKET="$API_SOCKET" \
    --mount /dev/shm:/dev/shm \
    --mount "${PWD}/agent_studio:/home/ubuntu/agent_studio/agent_studio:rbind,ro" \
    --mount "${PWD}/eval_online_benchmarks/files:/home/ubuntu/agent_studio/data:rbind,ro" \
    --root --rw "$SERVER_CONTAINER" &

  SERVER_PID=$!
  echo_ts "Server launched with PID $SERVER_PID"
}

restart_server_with_retries () {
  local retries="${1:-3}"
  local i
  for ((i=1; i<=retries; i++)); do
    # Ensure old server is gone
    if [ -n "$SERVER_PID" ]; then
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      SERVER_PID=""
    fi
    enroot remove -f "$SERVER_CONTAINER" 2>/dev/null || true

    start_server
    if wait_for_health 120 2; then
      return 0
    fi
    echo_ts "Server start attempt $i/$retries failed (health not ready). Retrying..."
  done
  echo_ts "Server failed to reach healthy state after $retries attempts."
  return 1
}

ensure_server_alive () {
  # If health fails OR background process is gone, restart.
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo_ts "Server process PID=$SERVER_PID not running."
    restart_server_with_retries 3 || return 1
  elif ! check_health; then
    echo_ts "Server health check failed; restarting server."
    restart_server_with_retries 3 || return 1
  fi
  return 0
}

############################################
# HuggingFace Server lifecycle
############################################
start_huggingface_server () {
  echo_ts "Starting HuggingFace model server with trained weights..."
  # Activate conda environment
  source /home/$USER/miniconda3/etc/profile.d/conda.sh
  conda activate agent-studio

  # Launch the server in background
  python scripts/huggingface_model_server.py \
    --model "$CRITIC_ACTOR_MODEL_NAME" \
    --port "$HF_SERVER_PORT" \
    --model_weights_path "$SAVED_WEIGHTS_DIR/checkpoint-$NUM_TRAINING_EPOCHS/" &

  HUGGINGFACE_SERVER_PID=$!
  echo_ts "HuggingFace server launched with PID $HUGGINGFACE_SERVER_PID"
}

start_hf_server_with_retries () {
  local retries="${1:-3}"
  local i
  for ((i=1; i<=retries; i++)); do
    # Ensure old server is gone
    if [ -n "$HUGGINGFACE_SERVER_PID" ]; then
      kill "$HUGGINGFACE_SERVER_PID" 2>/dev/null || true
      wait "$HUGGINGFACE_SERVER_PID" 2>/dev/null || true
      HUGGINGFACE_SERVER_PID=""
    fi

    start_huggingface_server
    if wait_for_hf_server 500 5; then
      return 0
    fi
    echo_ts "HuggingFace server start attempt $i/$retries failed (not ready). Retrying..."
  done
  echo_ts "HuggingFace server failed to reach ready state after $retries attempts."
  return 1
}

############################################
# Client run
############################################
run_client_once () {
  echo_ts "Starting data collection client for: $TASK_CONFIG_PATH"
  enroot remove -f "$CLIENT_CONTAINER" 2>/dev/null || true
  enroot create -n "$CLIENT_CONTAINER" agent-studio-client.sqsh

  enroot start \
    --env VNC_PASSWORD=123456 \
    --env ENV_SERVER_PORT="$ENV_SERVER_PORT" \
    --env VNC_PORT="$VNC_PORT" \
    --env SERVER_SOCKET="$SERVER_SOCKET" \
    --env API_WEB_SOCKET="$API_WEB_SOCKET" \
    --env API_SOCKET="$API_SOCKET" \
    --mount "${PWD}:/home/ubuntu/agent_studio" \
    --root --rw "$CLIENT_CONTAINER" -c "
      set -e
      cd /home/ubuntu/agent_studio
      python agent_studio/apps/online_exploration.py \
        --task_configs_path '$TASK_CONFIG_PATH' \
        --agent '$AGENT' \
        --prompting_approach '$PROMPTING_APPROACH' \
        --model '$ACTOR_MODEL_NAME' \
        --vnc_port '$VNC_PORT' \
        --env_server_port '$ENV_SERVER_PORT' \
        --model_server 0.0.0.0:'$MODEL_SERVER_PORT' \
        --remote \
        --exp_episodes '$EXP_EPISODES' \
        --plan_scoring_approach uniform \
        --finetuning_data_path '$DATA_COLLECTION_DIR' \
        --previous_plans_data_path '$PREV_PLANS_SAVE_DIR' \
        --plan_proposing_approach '$PLAN_GENERATION_APPROACH' \
        --save_finetuning_data
    "
  return $?
}

############################################
# Orchestration
############################################
# Start / ensure healthy agent-studio server up-front
restart_server_with_retries 3 || { echo_ts "Fatal: could not start healthy server."; exit 1; }

# Start HuggingFace model server with trained weights (only if not using diversity approach)
if [ "$PLAN_GENERATION_APPROACH" != "diversity" ]; then
  echo_ts "PLAN_GENERATION_APPROACH is '$PLAN_GENERATION_APPROACH', starting HuggingFace server..."
  start_hf_server_with_retries 3 || { echo_ts "Fatal: could not start HuggingFace server."; exit 1; }
fi


# Run client with retries; restart server if it dies mid-run
MAX_CLIENT_ATTEMPTS=5
attempt=1
while [ "$attempt" -le "$MAX_CLIENT_ATTEMPTS" ]; do
  echo_ts "Client attempt ${attempt}/${MAX_CLIENT_ATTEMPTS}..."
  # Always make sure server is alive & healthy before running the client
  if ! ensure_server_alive; then
    echo_ts "Unable to ensure healthy server before client attempt $attempt."
  fi

  if run_client_once; then
    echo_ts "Data collection successfully completed for: $TASK_CONFIG_PATH"
    break
  else
    rc=$?
    echo_ts "Client attempt ${attempt} failed with code ${rc}."

    # Check server status; if dead/unhealthy, restart for next attempt
    if ! ensure_server_alive; then
      echo_ts "Server was down or unhealthy; restarted."
    fi
    attempt=$((attempt + 1))
    sleep 5
  fi
done

if [ "$attempt" -gt "$MAX_CLIENT_ATTEMPTS" ]; then
  echo_ts "All ${MAX_CLIENT_ATTEMPTS} client attempts failed for ${TASK_CONFIG_PATH}."
  exit 2
fi

echo_ts "Cleaning up for task: $BASE_NAME"
# Cleanup is handled by trap on EXIT
echo_ts "Task completed: $TASK_CONFIG_PATH"
