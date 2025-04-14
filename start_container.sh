#!/bin/bash

# Build the Docker image
#echo "Building Docker image..."
#docker build -t grasp-pose-detector .

# Remove any existing container with the same name
#echo "Removing any existing container..."
#docker rm -f gpd-container 2>/dev/null

# Run the container with port forwarding and GPU access
echo "Starting container..."
docker run -d \
  --name gpd-container \
  -p 5000:5000 \
  --gpus all \
  -v /home/user/azirar/containers/gpd:/workspace \
  grasp-pose-detector

echo "Container started. Access the Flask application at http://localhost:5000"
echo "To view logs: docker logs gpd-container" 