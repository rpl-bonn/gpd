#!/bin/bash

# Update package lists
sudo apt-get update

# Install essential Python 2 build tools, pip, and available ROS Python 2 packages via apt
# REMOVED python-numpy from this list - will install via pip
sudo apt-get install -y --no-install-recommends \
    python-pip \
    python-dev \
    build-essential \
    cmake \
    curl \
    python-yaml \
    python-rospkg

# --- Upgrade pip for Python 2 using get-pip.py ---
echo "Attempting to upgrade pip for Python 2..."
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
sudo python2 get-pip.py
rm get-pip.py # Clean up the downloaded script

# Verify pip version (optional)
echo "Checking Python 2 pip version:"
python2 -m pip --version

# Define the Python 2 pip command
PIP_CMD="sudo python2 -m pip"
# --- End pip upgrade ---

# Install/Upgrade NumPy, Cython, and other dependencies using the upgraded Python 2 pip
echo "Using pip command: $PIP_CMD to install/upgrade NumPy, Cython, and other dependencies"
# Upgrade numpy first to a version compatible with python-pcl (pip will choose)
$PIP_CMD install --upgrade numpy
# Install Cython, needed for building python-pcl
$PIP_CMD install --upgrade Cython
# Install other potentially missing dependencies
$PIP_CMD install --upgrade pyyaml catkin_pkg

# Now install python-pcl, forcing reinstall to ensure it uses the current numpy
echo "Using pip command: $PIP_CMD to install python-pcl"
$PIP_CMD install --force-reinstall --no-cache-dir python-pcl

echo "Dependency installation script finished."
echo "Please ensure ROS environment is sourced correctly before running ROS nodes."
echo "Example: source /opt/ros/kinetic/setup.bash"
echo "Example: source /catkin_ws/devel/setup.bash" # If you have a catkin workspace