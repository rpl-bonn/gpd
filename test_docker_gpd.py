import os
import subprocess
import time

# Path to the docker run script
DOCKER_SCRIPT = os.path.expanduser("./run_docker_1.sh")

# Example command to run inside the container (can be changed as needed)
TEST_COMMAND = "roslaunch gpd_ros ur5.launch"


def run_docker_and_test():
    print("Starting Docker container using run_docker_1.sh...")
    # Start the docker container in detached mode with bash
    docker_proc = subprocess.Popen(["bash", DOCKER_SCRIPT], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(5)  # Give Docker some time to start

    print("Docker container started. Running test command inside container...")
    # You may need to use docker exec if you want to run a command in an already running container
    # For demonstration, we assume the container is interactive and you can type commands
    # In practice, you may want to use docker exec or modify the script to run the test command directly
    # Here, we just print instructions for manual testing
    print("\nTo test the ROS node, open another terminal and run:")
    print(f"docker ps  # Find your running container ID")
    print(f"docker exec -it <container_id> {TEST_COMMAND}")
    print("\nYou should see output from the gpd_ros node, waiting for a point cloud.")
    print("\nTo stop the container, use 'exit' inside the container shell or 'docker stop <container_id>' from another terminal.")

if __name__ == "__main__":
    run_docker_and_test()
