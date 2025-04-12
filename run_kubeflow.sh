#!/bin/bash

# Docker config
DOCKERFILE="Dockerfile.train"
IMAGE_NAME="gcr.io/zsc-personal/ml-cloud-pipeline"
CACHE_NAME="${IMAGE_NAME}:cache"
PLATFORM="linux/amd64"
TAG="latest"

delete_old_container_images() {
    echo -e "\n========== Delete recent containers that are not 'cache' or 'latest' ==========\n"
    DELETED_ANYTHING=true
    while [ "$DELETED_ANYTHING" = true ]; do
        DELETED_ANYTHING=false
        echo "Scanning $IMAGE_NAME for deletable digests..."

        gcloud container images list-tags "$IMAGE_NAME" \
            --format="value(digest,tags)" |
            while read digest_and_tags; do
                digest=$(echo "$digest_and_tags" | awk '{print $1}')
                tags=$(echo "$digest_and_tags" | cut -d' ' -f2-)

                if [[ "$tags" == *"latest"* || "$tags" == *"cache"* ]]; then
                    echo -e "\nSkipping sha256:$digest (tagged as latest or cache)\n"
                    continue
                fi

                if gcloud container images delete "$IMAGE_NAME@sha256:$digest" --quiet --force-delete-tags 2>/dev/null; then
                    echo "Deleted sha256:$digest"
                    DELETED_ANYTHING=true
                else
                    echo -e "\nParent manifest still exists for sha256:$digest, skipping\n"
                fi
            done

        if [ "$DELETED_ANYTHING" = false ]; then
            break
        fi
    done

    echo -e "\n========== Final list of remaining images ==========\n"
    gcloud container images list-tags "$IMAGE_NAME"
}

push_new_docker_image() {
    echo -e "\n========== Building and pushing Docker image with caching ==========\n"

    docker buildx build \
        --file "${DOCKERFILE}" \
        --platform "${PLATFORM}" \
        --tag "${IMAGE_NAME}:${TAG}" \
        --cache-from=type=registry,ref="${CACHE_NAME}" \
        --cache-to=type=registry,ref="${CACHE_NAME}",mode=max \
        --push \
        --progress="auto" \
        .

    echo -e "\n========== Build and push complete: ${IMAGE_NAME}:${TAG} =========="
}

run_kubeflow_pipeline() {
    echo -e "\n========== Upload pipeline and run =========="
    python run_kubeflow_pipeline.py
}

delete_old_container_images
push_new_docker_image
run_kubeflow_pipeline
