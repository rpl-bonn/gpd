#!/bin/bash
# Simple script to run grasp detection using the server API

# Check if we have at least two arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <item_cloud.ply> <env_cloud.pcd> [options]"
    echo "Options:"
    echo "  --config <config.yaml>   - Optional YAML configuration file"
    echo "  --rotation_resolution N  - Number of discrete yaw angles (default: 24)"
    echo "  --top_n N                - Number of grasp clusters (default: 3)"
    echo "  --n_best N               - Number of raw grasps before clustering (default: 60)"
    exit 1
fi

# Make sure Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Pass all arguments to the test_interface.py script
echo "Running grasp pose detection using server API..."
python test_interface.py "$@"
