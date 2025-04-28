#!/bin/bash
# Run GPD Docker with app.py and ROS

# Define variables
IMAGE_NAME="gpd_ros:latest"  # Replace with your actual Docker image name
CONTAINER_NAME="gpd_ros_container"
GPD_DIR=$(dirname $(readlink -f "$0"))  # Directory containing GPD files

# Check if the Docker image exists
if ! docker image inspect $IMAGE_NAME &>/dev/null; then
    echo "Docker image $IMAGE_NAME not found."
    echo "Please build the image first or specify the correct image name."
    exit 1
fi

# Stop and remove any existing container with the same name
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1
    docker rm $CONTAINER_NAME >/dev/null 2>&1
fi

# Create the docker run command
# Map GPD directory to /workspace in container
# Expose port 5000 for API access
# Run in detached mode
echo "Starting GPD Docker container..."
docker run -d \
    --name $CONTAINER_NAME \
    -v "$GPD_DIR:/workspace" \
    -p 5000:5000 \
    --network host \
    $IMAGE_NAME \
    /bin/bash -c "cd /workspace && pip install flask && bash -c 'python app_server.py --host 0.0.0.0 --port 5000 &' && chmod +x launch_gpd.sh && ./launch_gpd.sh"

# Check if container started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start Docker container."
    exit 1
fi

echo "Docker container started successfully."
echo "ROS GPD node is running inside the container."
echo "The app.py is accessible via port 5000."
echo ""
echo "You can access the container shell with:"
echo "  docker exec -it $CONTAINER_NAME /bin/bash"
echo ""
echo "To stop the container:"
echo "  docker stop $CONTAINER_NAME"
echo ""
echo "To test the GPD interface, run:"
echo "  python test_gpd.py --item item_cloud.pcd --env env_cloud.pcd"

# Optional: Wait for services to start and show logs
echo "Displaying container logs. Press Ctrl+C to stop viewing logs (container will continue running)."
docker logs -f $CONTAINER_NAME
