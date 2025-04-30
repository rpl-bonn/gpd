#!/bin/bash
# Client script to test the GPD server API

# Define variables
SERVER_URL="http://localhost:5000/predict"
ITEM_CLOUD="item_cloud.pcd"
ENV_CLOUD="env_cloud.pcd"
CURL_CMD="curl"

# Check if required files exist
if [ ! -f "$ITEM_CLOUD" ]; then
    echo "Error: Item cloud file not found: $ITEM_CLOUD"
    exit 1
fi

if [ ! -f "$ENV_CLOUD" ]; then
    echo "Error: Environment cloud file not found: $ENV_CLOUD"
    exit 1
fi

# Check if curl is installed
if ! command -v $CURL_CMD &>/dev/null; then
    echo "Error: curl is not installed. Please install curl to use this script."
    exit 1
fi

# Default parameter values
ROTATION_RESOLUTION=24
TOP_N=3
N_BEST=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --rot)
            ROTATION_RESOLUTION="$2"
            shift 2
            ;;
        --top_n)
            TOP_N="$2"
            shift 2
            ;;
        --n_best)
            N_BEST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $key"
            echo "Usage: $0 [--rot ROTATION_RESOLUTION] [--top_n TOP_N] [--n_best N_BEST]"
            exit 1
            ;;
    esac
done

echo "Sending grasp detection request to $SERVER_URL"
echo "Item cloud: $ITEM_CLOUD"
echo "Environment cloud: $ENV_CLOUD"
echo "Rotation resolution: $ROTATION_RESOLUTION"
echo "Top N: $TOP_N"
echo "N best: $N_BEST"
echo

# Send request
RESPONSE=$(curl -s -X POST -F "item_cloud=@$ITEM_CLOUD" -F "env_cloud=@$ENV_CLOUD" \
    -F "rotation_resolution=$ROTATION_RESOLUTION" \
    -F "top_n=$TOP_N" \
    -F "n_best=$N_BEST" \
    $SERVER_URL)

# Check if request was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to send request to server"
    exit 1
fi

# Print response
echo "Response from server:"
echo "$RESPONSE" | python -m json.tool

exit 0
