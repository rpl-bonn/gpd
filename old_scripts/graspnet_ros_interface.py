#!/usr/bin/env python3
"""
ROS interface for the GraspNet grasp detection system.
This module provides functions to communicate with GPD (Grasp Pose Detection) via ROS
services instead of the HTTP interface used in graspnet_interface.py.
"""

import os
import numpy as np
import open3d as o3d
import rospy
import copy
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple, List, Dict
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from gpd_ros.msg import CloudIndexed, CloudSources, GraspConfigList, GraspConfig
from gpd_ros.srv import detect_grasps

# Constants
# max gripper width is 0.175m, but in nn is 0.100m, therefore we scale models
SCALE = 0.1 / 0.175
MAX_GRIPPER_WIDTH = 0.07
GRIPPER_HEIGHT = 0.24227 * SCALE

def o3d_to_ros_cloud(cloud: o3d.geometry.PointCloud) -> PointCloud2:
    """
    Convert an Open3D point cloud to a ROS PointCloud2 message.
    
    Args:
        cloud: Open3D point cloud
        
    Returns:
        ROS PointCloud2 message
    """
    # Extract point cloud data
    points = np.asarray(cloud.points)
    
    # Create header
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "base_link"  # Use appropriate frame ID for your system
    
    # Define fields for the point cloud
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    
    # Create PointCloud2 message
    cloud_msg = pc2.create_cloud(header, fields, points)
    
    return cloud_msg

def create_cloud_indexed_msg(item_cloud: o3d.geometry.PointCloud,
                           env_cloud: o3d.geometry.PointCloud) -> CloudIndexed:
    """
    Create a CloudIndexed message for the GPD ROS service.
    
    Args:
        item_cloud: Point cloud of the item to grasp
        env_cloud: Point cloud of the environment
        
    Returns:
        CloudIndexed message for GPD ROS service
    """
    # Merge the point clouds
    merged_cloud = copy.deepcopy(item_cloud) + copy.deepcopy(env_cloud)
    
    # Create the ROS PointCloud2 message
    cloud_msg = o3d_to_ros_cloud(merged_cloud)
    
    # Create CloudSources message (indicates where the points came from)
    cloud_sources = CloudSources()
    cloud_sources.cloud = cloud_msg
    cloud_sources.view_points = [Point(0, 0, 0)]  # Default view point at origin
    
    # Create indices for the object points (item cloud points)
    # We need to identify which points in the merged cloud came from the item
    item_indices = list(range(len(item_cloud.points)))
    
    # Create CloudIndexed message
    cloud_indexed = CloudIndexed()
    cloud_indexed.cloud_sources = cloud_sources
    cloud_indexed.indices = item_indices
    
    return cloud_indexed

def transform_matrix_from_ros_grasp(grasp: GraspConfig) -> np.ndarray:
    """
    Convert a ROS GraspConfig message to a 4x4 transformation matrix.
    
    Args:
        grasp: ROS GraspConfig message
        
    Returns:
        4x4 transformation matrix
    """
    # Extract position
    position = np.array([grasp.position.x, grasp.position.y, grasp.position.z])
    
    # Extract orientation axes
    approach = np.array([grasp.approach.x, grasp.approach.y, grasp.approach.z])
    binormal = np.array([grasp.binormal.x, grasp.binormal.y, grasp.binormal.z])
    axis = np.array([grasp.axis.x, grasp.axis.y, grasp.axis.z])
    
    # Normalize the axes
    approach = approach / np.linalg.norm(approach)
    binormal = binormal / np.linalg.norm(binormal)
    axis = axis / np.linalg.norm(axis)
    
    # Create rotation matrix (approach = -Z, binormal = Y, axis = X)
    rotation = np.column_stack((axis, binormal, -approach))
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    
    return transform

def predict_grasps_ros(item_cloud: o3d.geometry.PointCloud,
                      env_cloud: o3d.geometry.PointCloud,
                      timeout: int = 90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict grasp poses using the GPD ROS service.
    
    Args:
        item_cloud: Point cloud of the item to grasp
        env_cloud: Point cloud of the environment
        timeout: Service call timeout in seconds
        
    Returns:
        Tuple of (transformation matrices, widths, scores)
    """
    # Initialize ROS node if not already initialized
    if not rospy.get_node_uri():
        rospy.init_node('grasp_client', anonymous=True)
    
    # Prepare the cloud indexed message
    cloud_indexed = create_cloud_indexed_msg(item_cloud, env_cloud)
    
    # Wait for the service to be available
    service_name = '/detect_grasps'
    rospy.loginfo(f"Waiting for service {service_name}...")
    rospy.wait_for_service(service_name, timeout=timeout)
    
    try:
        # Create service proxy
        detect_grasps_service = rospy.ServiceProxy(service_name, detect_grasps)
        
        # Call the service
        rospy.loginfo(f"Calling service {service_name}...")
        response = detect_grasps_service(cloud_indexed)
        grasps = response.grasp_configs.grasps
        
        # Process the grasp configurations
        tf_matrices = []
        widths = []
        scores = []
        
        for grasp in grasps:
            # Convert GraspConfig to transformation matrix
            tf_matrix = transform_matrix_from_ros_grasp(grasp)
            tf_matrices.append(tf_matrix)
            
            # Extract width and score
            widths.append(grasp.width)
            scores.append(grasp.score)
        
        # Convert lists to numpy arrays
        tf_matrices = np.array(tf_matrices)
        widths = np.array(widths)
        scores = np.array(scores)
        
        return tf_matrices, widths, scores
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return np.array([]), np.array([]), np.array([])

def get_best_grasp_ros(item_cloud: o3d.geometry.PointCloud,
                     env_cloud: o3d.geometry.PointCloud,
                     timeout: int = 90) -> Dict:
    """
    Get the best grasp pose for an object using the ROS interface.
    
    Args:
        item_cloud: Point cloud of the item to grasp
        env_cloud: Point cloud of the environment
        timeout: Service call timeout in seconds
        
    Returns:
        Dictionary with best grasp information (transform, width, score)
    """
    # Call predict_grasps_ros to get all grasps
    tf_matrices, widths, scores = predict_grasps_ros(item_cloud, env_cloud, timeout)
    
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

def visualize_grasps(item_cloud: o3d.geometry.PointCloud,
                    env_cloud: o3d.geometry.PointCloud,
                    tf_matrices: np.ndarray,
                    widths: np.ndarray,
                    scores: np.ndarray,
                    save_path: Optional[str] = None) -> o3d.geometry.PointCloud:
    """
    Visualize the detected grasp poses as coordinate frames in Open3D.
    
    Args:
        item_cloud: Point cloud of the item to grasp
        env_cloud: Point cloud of the environment
        tf_matrices: Transformation matrices of the grasp poses
        widths: Widths of the gripper for each grasp
        scores: Scores for each grasp
        save_path: Path to save the visualization point cloud (optional)
        
    Returns:
        Open3D PointCloud with grasp visualizations
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

def test_grasp_detection_ros():
    """
    Test the ROS-based grasp detection with provided point clouds.
    """
    print("Testing GraspNet with ROS interface using provided point clouds...")
    
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
    
    # Call the ROS-based grasp detection
    print("Calling ROS-based grasp detection...")
    tf_matrices, widths, scores = predict_grasps_ros(item_cloud, env_cloud)
    
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
        visualization_path = "/home/user/azirar/docker_containers/grasp_pose_detection/gpd/ros_grasp_visualization.ply"
        visualize_grasps(
            item_cloud,
            env_cloud,
            tf_matrices,
            widths,
            scores,
            save_path=visualization_path
        )
        print(f"Visualization saved to {visualization_path}")
    else:
        print("No grasps found.")

# Main function to run the test if this script is executed directly
if __name__ == "__main__":
    test_grasp_detection_ros()
