#!/bin/bash

# Build the Docker image if it doesn't exist
docker build -t grasp-pose-detector .

# Run the container with port forwarding and GPU access
docker run -d \
  --name gpd-container \
  -p 5000:5000 \
  --gpus all \
  grasp-pose-detector

echo "Container started. Access the Flask application at http://localhost:5000"
echo "To view logs: docker logs gpd-container" 