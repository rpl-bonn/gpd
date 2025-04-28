#!/bin/bash
# Simple wrapper script to use GPD from outside Docker
# This script starts the Docker container if needed and runs the external client

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# Docker container name
CONTAINER_NAME="gpd_container"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or you don't have permission to use Docker."
    echo "Please start Docker and make sure you have the necessary permissions."
    exit 1
fi

# Check if the container is already running
if ! docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
    echo "GPD Docker container is not running. Starting it now..."
    
    # Check if container exists but is stopped
    if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
        echo "Container exists but is stopped. Starting it..."
        docker start $CONTAINER_NAME
    else
        echo "Container does not exist. Creating and starting it..."
        # Start the container with our run_docker_new.sh script
        ./run_docker_new.sh
    fi
    
    # Wait for the service to start
    echo "Waiting for GPD service to start (this may take a few seconds)..."
    sleep 10
else
    echo "GPD Docker container is already running."
fi

# Check if Python is available
if ! command -v python >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    else
        echo "Error: Neither python nor python3 command found."
        exit 1
    fi
else
    PYTHON_CMD="python"
fi

# Check if requests module is available
$PYTHON_CMD -c "import requests" 2>/dev/null || {
    echo "Python requests module not found. Attempting to install it..."
    pip install requests || pip3 install requests || {
        echo "Failed to install requests module. Please install it manually:"
        echo "pip install requests"
        exit 1
    }
}

# Default values
ITEM_CLOUD="item_cloud.pcd"
ENV_CLOUD="env_cloud.pcd"
SERVER_URL="http://localhost:5000/predict"
ROT_RES=24
TOP_N=3
N_BEST=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --item)
            ITEM_CLOUD="$2"
            shift
            shift
            ;;
        --env)
            ENV_CLOUD="$2"
            shift
            shift
            ;;
        --server)
            SERVER_URL="$2"
            shift
            shift
            ;;
        --rot_res)
            ROT_RES="$2"
            shift
            shift
            ;;
        --top_n)
            TOP_N="$2"
            shift
            shift
            ;;
        --n_best)
            N_BEST="$2"
            shift
            shift
            ;;
        --json)
            JSON_OUTPUT="--json"
            shift
            ;;
        *)
            echo "Unknown option: $key"
            echo "Usage: $0 [--item ITEM_CLOUD] [--env ENV_CLOUD] [--server SERVER_URL] [--rot_res ROT_RES] [--top_n TOP_N] [--n_best N_BEST] [--json]"
            exit 1
            ;;
    esac
done

# Make the path absolute if relative
if [[ ! "$ITEM_CLOUD" = /* ]]; then
    ITEM_CLOUD="$SCRIPT_DIR/$ITEM_CLOUD"
fi

if [[ ! "$ENV_CLOUD" = /* ]]; then
    ENV_CLOUD="$SCRIPT_DIR/$ENV_CLOUD"
fi

# Check if the point cloud files exist
if [ ! -f "$ITEM_CLOUD" ]; then
    echo "Error: Item cloud file not found: $ITEM_CLOUD"
    exit 1
fi

if [ ! -f "$ENV_CLOUD" ]; then
    echo "Error: Environment cloud file not found: $ENV_CLOUD"
    exit 1
fi

echo "Using the following parameters:"
echo "  Item cloud: $ITEM_CLOUD"
echo "  Environment cloud: $ENV_CLOUD"
echo "  Server URL: $SERVER_URL"
echo "  Rotation resolution: $ROT_RES"
echo "  Top N: $TOP_N"
echo "  N best: $N_BEST"
if [ ! -z "$JSON_OUTPUT" ]; then
    echo "  Output format: JSON"
fi

# Run the external client
echo "Calling GPD service..."
$PYTHON_CMD gpd_external_client.py \
    --item "$ITEM_CLOUD" \
    --env "$ENV_CLOUD" \
    --server "$SERVER_URL" \
    --rot_res "$ROT_RES" \
    --top_n "$TOP_N" \
    --n_best "$N_BEST" \
    $JSON_OUTPUT

# Print help message
echo
echo "You can use this script with different parameters:"
echo "  ./use_gpd.sh --item path/to/item.pcd --env path/to/env.pcd"
echo "  ./use_gpd.sh --rot_res 12 --n_best 5"
echo "  ./use_gpd.sh --json  # to get JSON output"
