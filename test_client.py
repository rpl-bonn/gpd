import os
import tempfile
import numpy as np
import open3d as o3d
import requests
import json

GPD_SERVER_URL = "http://localhost:5000/detect_grasps"  # Changed from 0.0.0.0 to localhost

def predict_full_grasp(item_cloud: o3d.geometry.PointCloud,
                       env_cloud: o3d.geometry.PointCloud,
                       rotation_resolution=24,
                       top_n=3,
                       n_best=1,
                       timeout=90):
    """
    Merge the item and environment point clouds, then call the GPD server.
    """
    # For simplicity, we assume merged_cloud = item_cloud + env_cloud.
    merged_cloud = item_cloud + env_cloud

    # Save the merged cloud to a temporary file.
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "client_temp_cloud.pcd")
    o3d.io.write_point_cloud(file_path, merged_cloud)

    # Prepare parameters.
    params = {
        "rotation_resolution": str(rotation_resolution),
        "top_n": str(top_n),
        "n_best": str(n_best)
    }

    # Send the file via HTTP POST.
    with open(file_path, "rb") as f:
        files = {"point_cloud": f}
        response = requests.post(GPD_SERVER_URL, files=files, data=params, timeout=timeout)

    os.remove(file_path)

    response.raise_for_status()  # Raise an error for bad status codes.
    result = response.json()
    
    # Convert result fields into numpy arrays as needed:
    tf_matrices = np.array(result["tf_matrices"])
    widths = np.array(result["widths"])
    scores = np.array(result["scores"])

    return tf_matrices, widths, scores


if __name__ == "__main__":
    # Create a simple test item cloud (a cube)
    item_points = []
    for x in np.linspace(-0.05, 0.05, 10):
        for y in np.linspace(-0.05, 0.05, 10):
            for z in np.linspace(0, 0.1, 10):
                item_points.append([x, y, z])
    
    item_cloud = o3d.geometry.PointCloud()
    item_cloud.points = o3d.utility.Vector3dVector(np.array(item_points))
    
    # Create a simple environment cloud (a plane)
    env_points = []
    for x in np.linspace(-0.2, 0.2, 20):
        for y in np.linspace(-0.2, 0.2, 20):
            env_points.append([x, y, -0.01])  # Slightly below the object
    
    env_cloud = o3d.geometry.PointCloud()
    env_cloud.points = o3d.utility.Vector3dVector(np.array(env_points))
    
    print("Testing GPD server with simple point clouds...")
    # Call the GPD server
    tf_matrices, widths, scores = predict_full_grasp(
        item_cloud, 
        env_cloud, 
        rotation_resolution=8,  # Lower resolution for faster testing
        top_n=3,
        n_best=1
    )
    
    # Display results
    print(f"Found {len(scores)} grasp candidates")
    for i in range(len(scores)):
        print(f"Grasp {i+1}:")
        print(f"  Score: {scores[i]}")
        print(f"  Width: {widths[i]}")
        print(f"  Transform matrix:")
        print(tf_matrices[i])
        
