#!/bin/bash

# A script to simplify rebuilding and trying new docker images
# with enroot.

set -euo pipefail

# === CONFIG ===
IMAGE_NAME="agent-studio"
IMAGE_TAG="latest"
ENROOT_IMAGE="${IMAGE_NAME}.sqsh"
CONTAINER_NAME="${IMAGE_NAME}-server-test"

# === STEP 1: Build Docker image ===
echo "🔧 Building Docker image..."
sudo docker build -f dockerfiles/server/Dockerfile.ubuntu22.04.enroot.amd64 . -t ${IMAGE_NAME}:${IMAGE_TAG}

# === STEP 2: Import into Enroot ===
echo "📦 Importing Docker image into Enroot..."
rm -rf ${ENROOT_IMAGE}
sudo enroot import --output ${ENROOT_IMAGE} dockerd://${IMAGE_NAME}:${IMAGE_TAG}

# === STEP 3: Remove existing container (if any) ===
if enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "🧹 Removing old Enroot container: ${CONTAINER_NAME}"
    enroot remove ${CONTAINER_NAME}
fi

# === STEP 4: Create Enroot container ===
echo "📂 Creating Enroot container: ${CONTAINER_NAME}"
enroot create -n ${CONTAINER_NAME} ${ENROOT_IMAGE}

# === STEP 5: Optionally start it (uncomment below if desired) ===
# echo "🚀 Starting Enroot container..."
# enroot start --root --rw ${CONTAINER_NAME}

echo "✅ Done. Use 'enroot start --root ${CONTAINER_NAME}' to enter the container."
