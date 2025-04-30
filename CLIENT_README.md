# Grasp Pose Detection Client

This package provides a simple client interface to the Grasp Pose Detection (GPD) server running in Docker. It allows you to easily detect grasp poses for objects represented as point clouds.

## Overview

Instead of implementing the grasp detection algorithm yourself, this client connects to a server running in a Docker container, which handles all the complex processing and returns grasp poses.

## Quick Start

### 1. Make sure Docker is running

The GPD server runs inside a Docker container. Ensure Docker is installed and running on your system.

### 2. Using the wrapper script (easiest)

```bash
./run_grasp_detection.sh item_cloud.ply env_cloud.pcd
```

### 3. Using the Python script directly

```bash
python test_interface.py --item item_cloud.ply --env env_cloud.pcd [--options]
```

Available options:
- `--config <config.yaml>` - Optional configuration file
- `--rotation_resolution N` - Number of discrete yaw angles to sample (default: 24)
- `--top_n N` - Number of grasp clusters to return (default: 3)
- `--n_best N` - Number of raw grasps to keep before clustering (default: 60)

## Input Files

- `item_cloud.ply/pcd` - Point cloud of the object to be grasped
- `env_cloud.ply/pcd` - Point cloud of the environment (obstacles)

## Output

The script will print the detected grasp poses, including:
- Transformation matrices (position and orientation)
- Grasp widths
- Grasp scores

## Docker Container

The script will automatically start the Docker container if it's not already running. You can also manually start it using:

```bash
bash run_docker_new.sh
```

## Troubleshooting

1. **Docker not running**
   - Start Docker and try again

2. **Container not starting**
   - Check Docker logs: `docker logs gpd_container`

3. **Connection errors**
   - Ensure the server is running: `docker ps | grep gpd_container`
   - Check server status: `curl http://localhost:5000/health`

4. **Invalid point clouds**
   - Ensure your point clouds are in PCD or PLY format
   - Check that the point clouds are not empty
