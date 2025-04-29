#!/bin/bash

# This script installs Open3D in the Docker container and modifies the server code
# to use Open3D when PCL is not available

# Install Open3D in the container
docker exec gpd-container pip3 install open3d

# Copy the modified server code to the container
# (First we create the modifications locally, then copy them)

# Create a backup of the original file
cp app_server.py app_server.py.bak

# Modify the server code to use Open3D when PCL is not available
sed -i 's/# Try to import PCL for point cloud handling/# Try to import PCL for point cloud handling\ntry:\n    import open3d as o3d\n    has_o3d = True\nexcept ImportError:\n    has_o3d = False\n    print("Open3D not found, some fallbacks will not work.")\n/' app_server.py

# Modify the point cloud loading code
sed -i 's/# Load point clouds with PCL\n            if pcl is None:\n                return jsonify({'\''error'\'': '\''PCL module not available'\''}), 500/# Try to load point clouds with PCL or Open3D\n            if pcl is None and has_o3d:\n                # Use Open3D for point cloud handling\n                try:\n                    # Load point clouds\n                    item_cloud_o3d = o3d.io.read_point_cloud(item_temp.name)\n                    env_cloud_o3d = o3d.io.read_point_cloud(env_temp.name)\n                    \n                    # Convert to numpy arrays\n                    item_points = np.asarray(item_cloud_o3d.points, dtype=np.float32)\n                    env_points = np.asarray(env_cloud_o3d.points, dtype=np.float32)\n                    \n                    # Create our PointCloud objects\n                    item_cloud = PointCloud(item_points)\n                    env_cloud = PointCloud(env_points)\n                except Exception as e:\n                    return jsonify({'\''error'\'': f'\''Failed to load point clouds with Open3D: {str(e)}'\''})    \n            elif pcl is None:\n                return jsonify({'\''error'\'': '\''Neither PCL nor Open3D module available'\''}), 500/' app_server.py

# Copy the modified file to the container
docker cp app_server.py gpd-container:/workspace/app_server.py

# Restart the server process in the container
docker exec gpd-container pkill -f app_server.py
docker exec -d gpd-container python3 /workspace/app_server.py

echo "Open3D installed and server modified. Please try running the client again."
