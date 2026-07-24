#!/bin/bash
# =============================================================================
#  Build and push the Megatron-LM Docker image to GitHub Container Registry.
#
#  Pre-requisites:
#    1. Create a GitHub PAT with write:packages scope:
#       https://github.com/settings/tokens
#    2. Login once:
#       echo YOUR_TOKEN | docker login ghcr.io -u martin-kukla --password-stdin
#
#  Usage:
#    ./k8s/build.sh [TAG]
#
#  Example:
#    ./k8s/build.sh latest
#    ./k8s/build.sh v1.0
# =============================================================================
set -euo pipefail

GITHUB_USER="martin-kukla"
IMAGE_NAME="megatron-lm"
TAG=${1:-"latest"}
IMAGE="ghcr.io/${GITHUB_USER}/${IMAGE_NAME}:${TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building image: ${IMAGE}"
docker build \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${IMAGE}" \
    "${SCRIPT_DIR}/.."   # build context = Megatron-LM repo root

echo "==> Pushing image: ${IMAGE}"
docker push "${IMAGE}"

echo ""
echo "✅ Image pushed: ${IMAGE}"
echo ""
echo "Update k8s/megatron-mpijob.yaml image field to:"
echo "  image: ${IMAGE}"
