#!/bin/bash

# A script to simplify rebuilding and trying new docker images
# with enroot.

set -euo pipefail

# === PARSE ROLE ARG ===
if [[ $# -ne 1 || ("$1" != "server" && "$1" != "client") ]]; then
    echo "Usage: $0 [server|client]"
    exit 1
fi

ROLE="$1"

# === CONFIG ===
IMAGE_NAME="agent-studio"

if [[ "$ROLE" == "server" ]]; then
    DOCKERFILE="dockerfiles/server/Dockerfile.ubuntu22.04.enroot.amd64"
    ENROOT_IMAGE="${IMAGE_NAME}-server.sqsh"
    CONTAINER_NAME="${IMAGE_NAME}-server"
    IMAGE_TAG="server"
else
    DOCKERFILE="dockerfiles/client/Dockerfile.ubuntu22.04.amd64.client"
    ENROOT_IMAGE="${IMAGE_NAME}-client.sqsh"
    CONTAINER_NAME="${IMAGE_NAME}-client"
    IMAGE_TAG="client"
fi

# === STEP 1: Build Docker image ===
echo "🔧 Building Docker image for $ROLE..."
sudo docker buildx build --platform linux/amd64 -f "$DOCKERFILE" -t ${IMAGE_NAME}:${IMAGE_TAG} --load .

# === STEP 2: Import into Enroot ===
echo "📦 Importing Docker image into Enroot..."
rm -rf "${ENROOT_IMAGE}"
sudo enroot import --output "${ENROOT_IMAGE}" "dockerd://${IMAGE_NAME}:${IMAGE_TAG}"

# === STEP 3: Remove existing container (if any) ===
if enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "🧹 Removing old Enroot container: ${CONTAINER_NAME}"
    enroot remove "${CONTAINER_NAME}"
fi

# === STEP 4: Create Enroot container ===
echo "📂 Creating Enroot container: ${CONTAINER_NAME}"
enroot create -n "${CONTAINER_NAME}" "${ENROOT_IMAGE}"

# === STEP 5: Optionally start it (uncomment below if desired) ===
# echo "🚀 Starting Enroot container..."
# enroot start --root --rw "${CONTAINER_NAME}"

echo "✅ Done. Use 'enroot start --root ${CONTAINER_NAME}' to enter the $ROLE container."
