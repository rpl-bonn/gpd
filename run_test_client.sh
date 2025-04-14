#!/bin/bash

# Check if the Docker container is running
if ! docker ps | grep -q gpd-container; then
    echo "GPD container is not running. Starting it now..."
    ./start_container.sh
    # Wait for the container to start and the Flask app to be ready
    echo "Waiting for the Flask app to start..."
    sleep 10
fi

# Ask the user if they want to run the test client locally or in the container
echo "How would you like to run the test client?"
echo "1) Run locally (requires dependencies installed)"
echo "2) Run in the Docker container"
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo "Running test_client.py locally..."
    python3 test_client.py
elif [ "$choice" = "2" ]; then
    echo "Running test_client.py in the Docker container..."
    docker exec -it gpd-container python3 /opt/gpd/test_client.py
else
    echo "Invalid choice. Exiting."
    exit 1
fi 