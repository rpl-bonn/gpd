#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External client to use GPD service from outside the Docker container.
This script sends point cloud data to the GPD service running in the Docker container
and returns the grasp poses. It mimics the graspnet_interface from stretch-compose.

Reference: https://github.com/rpl-bonn/stretch-compose/blob/Yasmin/source/scripts/my_robot_scripts/graspnet_planning.py
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    import requests
    has_requests = True
except ImportError:
    has_requests = False
    print("Warning: requests module not found. HTTP API will not be available.")
    print("Please install it with: pip install requests")

try:
    # Import Open3D for point cloud handling
    import open3d as o3d
    has_o3d = True
except ImportError:
    has_o3d = False
    print("Warning: Open3D not found. Please install it with: pip install open3d")
    
# We're no longer trying to use PCL since it's difficult to install on modern Python
has_pcl = False

# Define a PointCloud class to match with the repository interface
class PointCloud:
    """Simple point cloud class compatible with the interface."""
    def __init__(self, points=None):
        """Initialize with points (numpy array of shape Nx3)."""
        self.points = points if points is not None else np.array([], dtype=np.float32)

    @classmethod
    def from_file(cls, file_path):
        """Load from PCD or PLY file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if has_o3d:
            # Use Open3D to load the point cloud
            o3d_cloud = o3d.io.read_point_cloud(file_path)
            points = np.asarray(o3d_cloud.points, dtype=np.float32)
            return cls(points)
        else:
            raise ImportError("Open3D is required but not available. Please install with: pip install open3d")
    
    def save(self, file_path):
        """Save to PCD file."""
        if has_o3d:
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(self.points)
            o3d.io.write_point_cloud(file_path, o3d_cloud)
            return True
        else:
            return False

# Define a Config class to match with the repository interface
@dataclass
class Config:
    """Configuration class for grasp detection."""
    gripper_width: float = 0.08
    finger_depth: float = 0.05
    hand_depth: float = 0.10
    object_min_height: float = 0.005

# Logger class for compatibility
class Logger:
    """Simple logger class."""
    def __init__(self, name="GPD"):
        self.name = name
    
    def info(self, msg):
        print(f"[INFO] {self.name}: {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {self.name}: {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {self.name}: {msg}")


def load_point_cloud(file_path):
    """Load a point cloud from a PCD or PLY file."""
    print(f"Loading point cloud from: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    
    try:
        # Use our PointCloud class that handles both Open3D and PCL
        cloud = PointCloud.from_file(file_path)
        print(f"Loaded {len(cloud.points)} points from file")
        return cloud
    except Exception as e:
        print(f"Error loading point cloud file: {e}")
        return None


def predict_full_grasp(
    item_cloud: PointCloud,
    env_cloud: PointCloud,
    config: Config,
    logger: Optional[Logger] = None,
    rotation_resolution: int = 24,
    top_n: int = 3,
    n_best: int = 1,
    timeout: int = 90,
    vis_block: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict a grasp position from the item point cloud and its environment.
    This function matches the interface from stretch-compose repository.
    
    Args:
        item_cloud: the point cloud of the item to be grasped
        env_cloud: the point cloud of the environment (typically within some radius)
        config: config object
        logger: for logging (optional)
        rotation_resolution: number of different angles for grasp detection
        top_n: number of different grasps per angle
        n_best: number of best grasps to return
        timeout: seconds for http request timeout
        vis_block: visualize grasp before returning
        
    Returns:
        Tuple containing:
        - transformation matrices (Nx4x4 numpy array)
        - grasp widths (N numpy array)
        - grasp scores (N numpy array)
    """
    if not has_requests:
        if logger:
            logger.error("Requests module not available. Cannot call GPD service.")
        else:
            print("Error: requests module not available. Cannot call GPD service.")
        return np.array([]), np.array([]), np.array([])
    
    # Create temporary files for the point clouds
    server_url = "http://localhost:5000/predict"
    temp_dir = os.path.dirname(os.path.abspath(__file__))
    item_cloud_path = os.path.join(temp_dir, "temp_item_cloud.pcd")
    env_cloud_path = os.path.join(temp_dir, "temp_env_cloud.pcd")
    
    # Save the point clouds to temporary files
    if logger:
        logger.info(f"Saving point clouds to temporary files")
    else:
        print("Saving point clouds to temporary files")
    
    try:
        item_cloud.save(item_cloud_path)
        env_cloud.save(env_cloud_path)
    except Exception as e:
        if logger:
            logger.error(f"Failed to save point clouds: {e}")
        else:
            print(f"Error: Failed to save point clouds: {e}")
        return np.array([]), np.array([]), np.array([])
    
    # Prepare files for upload
    if logger:
        logger.info(f"Sending request to GPD service at {server_url}")
    else:
        print(f"Sending request to GPD service at {server_url}")
    
    try:
        files = {
            'item_cloud': open(item_cloud_path, 'rb'),
            'env_cloud': open(env_cloud_path, 'rb')
        }
        
        # Prepare parameters
        data = {
            'rotation_resolution': rotation_resolution,
            'top_n': top_n,
            'n_best': n_best,
            'vis_block': 1 if vis_block else 0
        }
        
        # Send the request
        start_time = time.time()
        response = requests.post(server_url, files=files, data=data, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            if logger:
                logger.info(f"Request completed in {elapsed_time:.2f} seconds")
            else:
                print(f"Request completed in {elapsed_time:.2f} seconds")
            
            # Convert the JSON result to NumPy arrays
            tf_matrices = np.array(result.get('tf_matrices', []))
            widths = np.array(result.get('widths', []))
            scores = np.array(result.get('scores', []))
            
            return tf_matrices, widths, scores
        else:
            if logger:
                logger.error(f"Request failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
            else:
                print(f"Error: Request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
            return np.array([]), np.array([]), np.array([])
    except Exception as e:
        if logger:
            logger.error(f"Error calling GPD service: {e}")
        else:
            print(f"Error calling GPD service: {e}")
        return np.array([]), np.array([]), np.array([])
    finally:
        # Close the files
        if 'files' in locals():
            files['item_cloud'].close()
            files['env_cloud'].close()
        
        # Clean up temporary files
        try:
            if os.path.exists(item_cloud_path):
                os.remove(item_cloud_path)
            if os.path.exists(env_cloud_path):
                os.remove(env_cloud_path)
        except:
            pass


def print_grasp_results(tf_matrices, widths, scores):
    """Process and display the grasp results."""
    if len(tf_matrices) == 0:
        print("No valid grasps found")
        return
    
    print(f"\nFound {len(tf_matrices)} grasp(s):")
    
    for i, (tf_matrix, width, score) in enumerate(zip(tf_matrices, widths, scores)):
        print(f"\nGrasp {i+1}:")
        print(f"  Score: {score:.4f}")
        print(f"  Width: {width:.4f}")
        
        # Extract position from the transformation matrix
        position = [tf_matrix[0, 3], tf_matrix[1, 3], tf_matrix[2, 3]]
        print(f"  Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        
        # Extract orientation vectors
        approach = tf_matrix[:3, 0]
        binormal = tf_matrix[:3, 1]
        axis = tf_matrix[:3, 2]
        
        print(f"  Approach: [{approach[0]:.4f}, {approach[1]:.4f}, {approach[2]:.4f}]")
        print(f"  Binormal: [{binormal[0]:.4f}, {binormal[1]:.4f}, {binormal[2]:.4f}]")
        print(f"  Axis:     [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
        
        # Display the transformation matrix
        print("  Transformation Matrix:")
        for row in tf_matrix:
            print("    " + " ".join([f"{val:7.4f}" for val in row]))


def visualize_grasps(tf_matrices, widths, scores=None):
    """
    Visualize the grasp poses (if Open3D is available).
    
    Args:
        tf_matrices: Transformation matrices for the grasps (Nx4x4)
        widths: Grasp widths (N)
        scores: Grasp scores (N, optional)
    """
    if not has_o3d or len(tf_matrices) == 0:
        return
    
    try:
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Grasp Visualization", width=800, height=600)
        
        # Create coordinate frames for each grasp
        for i, (tf_matrix, width) in enumerate(zip(tf_matrices, widths)):
            # Create a coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=tf_matrix[:3, 3]
            )
            
            # Rotate the frame to match the grasp orientation
            frame.rotate(tf_matrix[:3, :3], center=tf_matrix[:3, 3])
            
            # Add the frame to the visualization
            vis.add_geometry(frame)
            
            # Create a line to represent the gripper width
            binormal = tf_matrix[:3, 1]
            p1 = tf_matrix[:3, 3] + binormal * (width / 2)
            p2 = tf_matrix[:3, 3] - binormal * (width / 2)
            
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(np.vstack([p1, p2]))
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color
            
            vis.add_geometry(line)
        
        # Configure the view
        opt = vis.get_render_option()
        opt.background_color = np.array([0.8, 0.8, 0.8])  # Light gray background
        opt.point_size = 3.0
        
        # Run the visualization
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"Error visualizing grasps: {e}")
        return


class GPDInterface:
    """
    Interface to GPD service, mimicking the interface in stretch-compose repository.
    This class provides methods to predict grasp poses using the GPD service.
    """
    
    def __init__(self, logger=None):
        """Initialize the interface."""
        self.logger = logger if logger else Logger("GPDInterface")
    
    def predict_full_grasp(
        self,
        item_cloud: PointCloud,
        env_cloud: PointCloud,
        config: Config,
        rotation_resolution: int = 24,
        top_n: int = 3,
        n_best: int = 1,
        timeout: int = 90,
        vis_block: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict a grasp position from the item point cloud and its environment.
        Same interface as in stretch-compose repository.
        """
        return predict_full_grasp(
            item_cloud,
            env_cloud,
            config,
            self.logger,
            rotation_resolution,
            top_n,
            n_best,
            timeout,
            vis_block,
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="External client for GPD service")
    parser.add_argument("--item", type=str, default="item_cloud.pcd",
                        help="Path to item point cloud PCD/PLY file")
    parser.add_argument("--env", type=str, default="env_cloud.pcd",
                        help="Path to environment point cloud PCD/PLY file")
    parser.add_argument("--rot_res", type=int, default=24,
                        help="Rotation resolution (default: 24)")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Top N grasps per angle (default: 3)")
    parser.add_argument("--n_best", type=int, default=3,
                        help="N best grasps to return (default: 3)")
    parser.add_argument("--vis", action="store_true",
                        help="Visualize the grasps")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    # Create logger
    logger = Logger("GPDClient")
    
    # Load point clouds
    item_cloud = load_point_cloud(args.item)
    if item_cloud is None:
        return 1
    
    env_cloud = load_point_cloud(args.env)
    if env_cloud is None:
        return 1
    
    # Create config
    config = Config()
    
    # Create interface instance
    interface = GPDInterface(logger=logger)
    
    # Call predict_full_grasp function
    logger.info("Calling predict_full_grasp function...")
    start_time = time.time()
    
    tf_matrices, widths, grasp_scores = interface.predict_full_grasp(
        item_cloud=item_cloud,
        env_cloud=env_cloud,
        config=config,
        rotation_resolution=args.rot_res,
        top_n=args.top_n,
        n_best=args.n_best,
        vis_block=args.vis,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Grasp prediction completed in {elapsed_time:.2f} seconds")
    
    if args.json:
        # Output as JSON
        result = {
            'tf_matrices': tf_matrices.tolist() if len(tf_matrices) > 0 else [],
            'widths': widths.tolist() if len(widths) > 0 else [],
            'grasp_scores': grasp_scores.tolist() if len(grasp_scores) > 0 else []
        }
        print(json.dumps(result, indent=2))
    else:
        # Display the results
        print_grasp_results(tf_matrices, widths, grasp_scores)
    
    # Visualize if requested
    if args.vis and has_o3d and len(tf_matrices) > 0:
        logger.info("Visualizing grasps...")
        visualize_grasps(tf_matrices, widths, grasp_scores)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
