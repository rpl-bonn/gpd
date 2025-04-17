"""
Util functions for communicating with graspnet docker server.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import open3d as o3d
import requests
import json
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple, List

# Constants
# max gripper width is 0.175m, but in nn is 0.100m, therefore we scale models
SCALE = 0.1 / 0.175
MAX_GRIPPER_WIDTH = 0.07
GRIPPER_HEIGHT = 0.24227 * SCALE

def predict_full_grasp(
    item_cloud: o3d.geometry.PointCloud,
    env_cloud: o3d.geometry.PointCloud,
    config: Optional[dict] = None,
    server_ip: str = "127.0.0.1",
    server_port: int = 5000,
    rotation_resolution: int = 24,
    top_n: int = 3,
    n_best: int = 1,
    timeout: int = 90
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge the item and environment point clouds, then call the GraspNet server.
    
    Parameters:
        item_cloud: Point cloud of the item to grasp
        env_cloud: Point cloud of the environment
        config: Configuration dictionary (optional)
        server_ip: Server IP address
        server_port: Server port
        rotation_resolution: Number of rotations to consider
        top_n: Number of top grasps to return
        n_best: Number of best grasps to select
        timeout: Request timeout in seconds
        
    Returns:
        tuple: (tf_matrices, widths, scores)
            - tf_matrices: Numpy array of transformation matrices
            - widths: Numpy array of gripper widths
            - scores: Numpy array of grasp scores
    """
    # If config is provided, override default parameters
    if config is not None:
        # Check if config has server information
        if "servers" in config and "graspnet" in config["servers"]:
            server_config = config["servers"]["graspnet"]
            server_ip = server_config.get("ip", server_ip)
            server_port = server_config.get("port", server_port)
    
    # Construct the server URL
    server_url = f"http://{server_ip}:{server_port}/detect_grasps"
    
    # Merge the item and environment point clouds
    merged_cloud = item_cloud + env_cloud
    
    # Get the bounding box of the item to calculate centroid for transform correction later
    item_bbox = item_cloud.get_axis_aligned_bounding_box()
    item_center = item_bbox.get_center()
    print(f"DEBUG: Item center is at {item_center}")

    # Save the merged cloud to a temporary file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "client_temp_cloud.pcd")
    o3d.io.write_point_cloud(file_path, merged_cloud)

    # Prepare parameters
    params = {
        "rotation_resolution": str(rotation_resolution),
        "top_n": str(top_n),
        "n_best": str(n_best)
    }

    # Send the file via HTTP POST
    with open(file_path, "rb") as f:
        files = {"point_cloud": f}
        response = requests.post(server_url, files=files, data=params, timeout=timeout)

    # Clean up temporary file
    os.remove(file_path)

    # Raise an error for bad status codes
    response.raise_for_status()
    
    # Parse the response
    result = response.json()
    # Debug: print raw server response and array lengths
    print("DEBUG: GraspNet server returned:", result)
    print(f"DEBUG: tf_matrices count={len(result.get('tf_matrices', []))}, widths count={len(result.get('widths', []))}, scores count={len(result.get('scores', []))}")
    
    # Convert result fields into numpy arrays
    tf_matrices = np.array(result["tf_matrices"])
    widths = np.array(result["widths"])
    scores = np.array(result["scores"])
    
    # Apply transformation to move grasp poses to the actual item location
    if len(tf_matrices) > 0:
        print("Transforming grasp poses to item location...")
        corrected_tf_matrices = []
        for tf in tf_matrices:
            # Create a copy of the transformation matrix
            corrected_tf = tf.copy()
            # Set the translation part to be relative to the item's center
            corrected_tf[0, 3] += item_center[0]  # X coordinate
            corrected_tf[1, 3] += item_center[1]  # Y coordinate
            corrected_tf[2, 3] += item_center[2]  # Z coordinate
            corrected_tf_matrices.append(corrected_tf)
        
        tf_matrices = np.array(corrected_tf_matrices)
        print("Grasp poses transformed:")
        for i, tf in enumerate(tf_matrices):
            print(f"Grasp {i+1} position: [{tf[0,3]:.4f}, {tf[1,3]:.4f}, {tf[2,3]:.4f}]")

    return tf_matrices, widths, scores
  
def get_best_grasp(
    item_cloud: o3d.geometry.PointCloud,
    env_cloud: o3d.geometry.PointCloud,
    config: Optional[dict] = None,
    server_ip: str = "127.0.0.1",
    server_port: int = 5000,
    rotation_resolution: int = 24,
    top_n: int = 5,
    n_best: int = 1
) -> dict:
    """
    Get the best grasp pose for an object.
    
    Parameters:
        item_cloud: Point cloud of the item to grasp
        env_cloud: Point cloud of the environment
        config: Configuration dictionary (optional)
        server_ip: Server IP address
        server_port: Server port
        rotation_resolution: Number of rotations to consider
        top_n: Number of top grasps to return
        n_best: Number of best grasps to select
        
    Returns:
        dict: Best grasp information with transformation matrix, width, and score
    """
    tf_matrices, widths, scores = predict_full_grasp(
        item_cloud, 
        env_cloud, 
        config=config,
        server_ip=server_ip,
        server_port=server_port,
        rotation_resolution=rotation_resolution,
        top_n=top_n,
        n_best=n_best
    )
    
    # Check if any grasps were found
    if len(scores) == 0:
        return {
            "success": False,
            "message": "No grasps found"
        }
    
    # Return the best grasp
    best_idx = np.argmax(scores)
    return {
        "success": True,
        "transform": tf_matrices[best_idx].tolist(),
        "width": float(widths[best_idx]),
        "score": float(scores[best_idx])
    }

def create_test_point_clouds() -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Create sample point clouds for testing purposes.
    
    Returns:
        tuple: (item_cloud, env_cloud)
    """
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
    
    return item_cloud, env_cloud

def visualize_grasps(
    item_cloud: o3d.geometry.PointCloud,
    env_cloud: o3d.geometry.PointCloud,
    tf_matrices: np.ndarray,
    widths: np.ndarray,
    scores: np.ndarray,
    save_path: Optional[str] = None
) -> o3d.geometry.PointCloud:
    """
    Visualize the detected grasp poses as coordinate frames in Open3D.
    
    Parameters:
        item_cloud: Point cloud of the item to grasp
        env_cloud: Point cloud of the environment
        tf_matrices: Transformation matrices of the grasp poses
        widths: Widths of the gripper for each grasp
        scores: Scores for each grasp
        save_path: Path to save the visualization point cloud (optional)
        
    Returns:
        o3d.geometry.PointCloud: Combined point cloud with grasp visualizations
    """
    # Create a new point cloud to visualize
    visualized_cloud = o3d.geometry.PointCloud()
    
    # Add the item and environment clouds
    visualized_cloud += item_cloud
    visualized_cloud += env_cloud
    
    # Create a list to store all grasp frames for visualization
    grasp_frames = []
    
    # Color for the different grasps (from red=best to blue=worst)
    color_map = plt.cm.jet
    
    # Normalize scores for coloring
    if len(scores) > 0:
        score_min = min(scores)
        score_max = max(scores)
        score_range = score_max - score_min if score_max > score_min else 1.0
    
    # Create a coordinate frame for each grasp
    for i, (transform, width, score) in enumerate(zip(tf_matrices, widths, scores)):
        # Create a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05,  # Adjust size as needed
            origin=transform[:3, 3]  # Use the translation part of the transform
        )
        
        # Apply rotation from transform
        frame.rotate(transform[:3, :3], center=transform[:3, 3])
        
        # Add the frame to the list
        grasp_frames.append(frame)
        
        # Create grasp width visualization (a line between the fingers)
        finger_width = width
        
        # Calculate finger positions in local coordinate frame
        # Assuming the gripper closes along the x-axis
        left_finger = transform @ np.array([-finger_width/2, 0, 0, 1])
        right_finger = transform @ np.array([finger_width/2, 0, 0, 1])
        
        # Create points for fingers
        finger_points = np.vstack([left_finger[:3], right_finger[:3]])
        finger_cloud = o3d.geometry.PointCloud()
        finger_cloud.points = o3d.utility.Vector3dVector(finger_points)
        
        # Color based on score (normalized)
        if len(scores) > 1:
            norm_score = (score - score_min) / score_range
            color = color_map(norm_score)[:3]  # Get RGB from colormap
        else:
            color = [1, 0, 0]  # Red for single grasp
            
        # Set the color for finger points
        finger_cloud.paint_uniform_color(color)
        
        # Add to visualization
        visualized_cloud += finger_cloud
    
    # Save the visualization if requested
    if save_path is not None:
        # Convert path extension to .ply if it's .pcd
        if save_path.endswith('.pcd'):
            save_path = save_path.replace('.pcd', '.ply')
        
        o3d.io.write_point_cloud(save_path, visualized_cloud)
        print(f"Visualization saved to {save_path}")
        
        # Also save a separate file with only the grasp frames for clarity
        grasp_cloud = o3d.geometry.PointCloud()
        for i, (transform, width, score) in enumerate(zip(tf_matrices, widths, scores)):
            # Create points for grasp center
            center_point = transform[:3, 3]
            grasp_cloud.points.append(center_point)
            
            # Add color based on score
            if len(scores) > 1:
                norm_score = (score - score_min) / score_range
                color = color_map(norm_score)[:3]
            else:
                color = [1, 0, 0]  # Red for single grasp
                
            grasp_cloud.colors.append(color)
        
        grasp_frames_path = save_path.replace('.ply', '_grasp_frames.ply')
        o3d.io.write_point_cloud(grasp_frames_path, grasp_cloud)
        print(f"Grasp frames saved to {grasp_frames_path}")
    
    return visualized_cloud

# Test function that can be run directly
def test_grasp_detection():
    """
    Test the grasp detection with the provided point clouds.
    """
    print("Testing GraspNet server with provided point clouds...")
    
    # Load the provided point clouds
    print("Loading point clouds from files...")
    item_cloud_path = "/home/user/azirar/docker_containers/grasp_pose_detection/gpd/item_cloud.ply"
    env_cloud_path = "/home/user/azirar/docker_containers/grasp_pose_detection/gpd/env_cloud.ply"
    
    # Check if files exist
    if not os.path.exists(item_cloud_path) or not os.path.exists(env_cloud_path):
        print(f"Error: Point cloud files not found at {item_cloud_path} or {env_cloud_path}")
        return
    
    # Load the point clouds using Open3D
    item_cloud = o3d.io.read_point_cloud(item_cloud_path)
    env_cloud = o3d.io.read_point_cloud(env_cloud_path)
    
    print(f"Loaded item point cloud with {len(item_cloud.points)} points")
    print(f"Loaded environment point cloud with {len(env_cloud.points)} points")
    
    # Print information about the point clouds for debugging
    print(f"Item cloud has colors? {hasattr(item_cloud, 'colors') and len(item_cloud.colors) > 0}")
    print(f"Environment cloud has colors? {hasattr(env_cloud, 'colors') and len(env_cloud.colors) > 0}")
    
    # Calculate and print bounding boxes to check scale/positioning
    item_bbox = item_cloud.get_axis_aligned_bounding_box()
    env_bbox = env_cloud.get_axis_aligned_bounding_box()
    print(f"Item cloud bounding box min: {item_bbox.min_bound}, max: {item_bbox.max_bound}")
    print(f"Environment cloud bounding box min: {env_bbox.min_bound}, max: {env_bbox.max_bound}")
    
    # Add more verbose logging to show progress
    print("Sending point clouds to GraspNet server for grasp detection...")
    
    # Call the GraspNet server with increased parameters for better detection chances
    tf_matrices, widths, scores = predict_full_grasp(
        item_cloud, 
        env_cloud, 
        rotation_resolution=32,  # Increased resolution for better detection
        top_n=100,               # Increased to get more grasp candidates
        n_best=3,                # Increased to get more of the best grasps
        timeout=120              # Increased timeout for larger point clouds
    )
    
    # Display results
    if len(scores) > 0:
        print(f"Found {len(scores)} grasp candidates")
        for i in range(len(scores)):
            print(f"Grasp {i+1}:")
            print(f"  Score: {scores[i]}")
            print(f"  Width: {widths[i]}")
            print(f"  Transform matrix:")
            print(tf_matrices[i])
        
        # Visualize the grasp poses and save the visualization
        print("Visualizing grasp poses...")
        visualization_path = "/home/user/azirar/docker_containers/grasp_pose_detection/gpd/grasp_visualization.ply"
        visualize_grasps(
            item_cloud,
            env_cloud,
            tf_matrices,
            widths,
            scores,
            save_path=visualization_path
        )
        print(f"Visualization saved to {visualization_path}")
        print(f"You can view the visualization using Rerun, Open3D or CloudCompare.")
    else:
        print("No grasps found.")

# Main function to run the test if this script is executed directly
if __name__ == "__main__":
    test_grasp_detection()
