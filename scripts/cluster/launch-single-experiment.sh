#!/bin/bash

# Define variables
USER="njkmr"
PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/os/1d6765b3-b744-4aa4-8287-d14e6d3cddac.json"
MODEL_NAME="gpt-4o-2024-08-06"

# Step 1: Launch the server
SERVER_JOB_ID=$(srun --parsable apptainer exec --no-home --bind /dev/shm:/dev/shm --writable-tmpfs --fakeroot \
  --bind /home/$USER/agent-studio/scripts/agent_server.py:/home/ubuntu/agent_studio/scripts/agent_server.py:ro \
  --bind /home/$USER/agent-studio/agent_studio/envs:/home/ubuntu/agent_studio/agent_studio/envs:ro \
  --bind /home/$USER/agent-studio/agent_studio/utils:/home/ubuntu/agent_studio/agent_studio/utils:ro \
  --bind /home/$USER/agent-studio/agent_studio/agent:/home/ubuntu/agent_studio/agent_studio/agent:ro \
  --bind /home/$USER/agent-studio/agent_studio/config:/home/ubuntu/agent_studio/agent_studio/config \
  --bind /home/$USER/agent-studio/eval_online_benchmarks/files:/home/ubuntu/agent_studio/data:ro \
  --bind supervisor_logs/:/var/log agent-studio-server.sif /home/ubuntu/agent_studio/scripts/docker_startup.sh)

# Wait for the job to start and get the node name
while true; do
  NODE_NAME=$(squeue -j $SERVER_JOB_ID -o "%N" -h)
  if [ -n "$NODE_NAME" ]; then
    break
  fi
  sleep 1
done

# Step 3: SSH into the node and run the client container
ssh $USER@$NODE_NAME << EOF
  cd agent-studio
  apptainer exec --no-home --bind /dev/shm:/dev/shm --writable-tmpfs --fakeroot \
    --bind /home/$USER/agent-studio/agent_studio/:/home/ubuntu/agent_studio/ \
    agent-studio-client.sif /bin/bash -c "
      cd /home/ubuntu/agent_studio
      as-online-benchmark --task_configs_path $PARTICULAR_PATH --model $MODEL_NAME --remote
    "
EOF

# Step 7: Kill the server job
scancel $SERVER_JOB_ID

echo "Workflow completed and server job $SERVER_JOB_ID has been cancelled."
