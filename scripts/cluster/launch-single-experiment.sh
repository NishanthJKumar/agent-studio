#!/bin/bash

# Define variables
USER="njkmr"
PARTICULAR_PATH="<particular-path>"
MODEL_NAME="<model-name>"

# Step 1: Submit the server job and capture the job ID
SERVER_JOB_ID=$(sbatch --parsable server_job.sh)

# Wait for the job to start running and get the node name
while true; do
  JOB_STATE=$(squeue -j $SERVER_JOB_ID -o "%T" -h)
  if [ "$JOB_STATE" == "RUNNING" ]; then
    NODE_NAME=$(squeue -j $SERVER_JOB_ID -o "%N" -h)
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

echo "Workflow completed and server job $SERVER_JOB_ID has been cancelled
