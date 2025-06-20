#!/bin/bash

# Define variables
USER="njkmr"
PARTICULAR_PATH="eval_online_benchmarks/tasks/single_gui/os/1d6765b3-b744-4aa4-8287-d14e6d3cddac.json"
MODEL_NAME="gpt-4o-2024-08-06"

# Step 1: Submit the server job and capture the job ID
SERVER_JOB_ID=$(sbatch --parsable scripts/cluster/server_job.sh)

echo "Launched server job with job ID $SERVER_JOB_ID!"

# Wait for the job to start running and get the node name
while true; do
  JOB_STATE=$(squeue -j $SERVER_JOB_ID -o "%T" -h)
  if [ "$JOB_STATE" == "RUNNING" ]; then
    NODE_NAME=$(squeue -j $SERVER_JOB_ID -o "%N" -h)
    echo "Server is successfully running on node $NODE_NAME"
    break
  fi
  echo "Waiting for the server job to start running..."
  sleep 5
done

# Step 3: SSH into the node and run the client container
echo "Attempting to SSH into node $NODE_NAME..."
while true; do
  ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -t $USER@$NODE_NAME "exit"
  if [ $? -eq 0 ]; then
    echo "SSH connection established to $NODE_NAME"
    break
  fi
  echo "SSH connection failed, retrying..."
  sleep 5
done

ssh -t $USER@$NODE_NAME << EOF
  echo "Running client container on node $NODE_NAME..."
  cd agent-studio
  apptainer exec --no-home --bind /dev/shm:/dev/shm --writable-tmpfs --fakeroot \
    --bind /home/$USER/agent-studio/:/home/ubuntu/agent_studio/ \
    agent-studio-client.sif /bin/bash -c "
      cd /home/ubuntu/agent_studio
    "
