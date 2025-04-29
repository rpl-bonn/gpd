#!/bin/bash
# Start GPD ROS and the Flask server

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if roslaunch is available
if ! command_exists roslaunch; then
  echo "Error: roslaunch command not found. Make sure ROS is installed and sourced."
  exit 1
fi

# Check if Python is available
if ! command_exists python; then
  echo "Error: python command not found."
  exit 1
fi

# Function to start the ROS node
start_ros_node() {
  echo "Starting GPD ROS node..."
  roslaunch gpd_ros ur5.launch &
  ROS_PID=$!
  echo "GPD ROS node started with PID $ROS_PID"
}

# Function to start the Flask server
start_flask_server() {
  echo "Starting Flask server on port 5000..."
  
  # Try to install Flask if not available
  python -c "import flask" 2>/dev/null || {
    echo "Flask not found. Trying to install it..."
    pip install flask || {
      echo "Could not install Flask. The server will be started without it."
      # Start without Flask
      python app.py &
      return
    }
  }

  # Start Flask server
  python app_server.py --host 0.0.0.0 --port 5000 &
  SERVER_PID=$!
  echo "Flask server started with PID $SERVER_PID"
}

# Start both processes
start_ros_node
start_flask_server

# Wait for both processes
echo
echo "Both processes started. Press Ctrl+C to stop."
echo "You can access the Flask server at http://localhost:5000"
echo

# Trap for Ctrl+C
trap "echo 'Stopping processes...'; kill $ROS_PID 2>/dev/null; kill $SERVER_PID 2>/dev/null; exit 0" INT TERM

# Keep the script running
wait
