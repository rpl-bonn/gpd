#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPD (Grasp Pose Detection) interface for external access.
This module provides access to the GPD algorithm through a simple API.
"""
from __future__ import division, print_function

import os
import time
import json
import tempfile
import subprocess
import numpy as np
import logging

# Python 2 compatibility - typing module not available
# Remove the type annotations when running the code

# Import pcl for point cloud handling
try:
    import pcl
except ImportError:
    logging.warning("PCL Python bindings not found. Some functionality may be limited.")

# Define a PointCloud type for type hints - replace with your actual PointCloud class if needed
class PointCloud:
    """Placeholder for actual PointCloud class. Replace with your implementation."""
    def __init__(self, points=None):
        self.points = points if points is not None else np.array([])
        
    def to_pcl(self):
        """Convert to PCL point cloud"""
        if 'pcl' not in globals():
            raise ImportError("PCL Python bindings required for this operation")
        
        cloud = pcl.PointCloud()
        cloud.from_array(self.points.astype(np.float32))
        return cloud
        
    @classmethod
    def from_pcl(cls, pcl_cloud):
        """Create from PCL point cloud"""
        return cls(np.array(pcl_cloud.to_array()))

# Define a simple Config class if not importing the original
class Config:
    """Simple configuration class."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Logger:
    """Simple logger class."""
    def __init__(self, name="GPD"):
        self.name = name
        
    def info(self, msg):
        print("[INFO] {0}: {1}".format(self.name, msg))
        
    def warning(self, msg):
        print("[WARNING] {0}: {1}".format(self.name, msg))
        
    def error(self, msg):
        print("[ERROR] {0}: {1}".format(self.name, msg))

def _save_point_cloud_to_file(cloud, filepath):
    """Save a point cloud to a file in PCD format."""
    try:
        pcl_cloud = cloud.to_pcl()
        pcl.save(pcl_cloud, filepath, binary=False)
        return True
    except Exception as e:
        print("Failed to save point cloud: {0}".format(e))
        return False

def _read_grasp_file(filepath, n_best=1):
    """Read grasp data from a JSON file."""
    if not os.path.exists(filepath):
        raise IOError("Grasp file not found: {0}".format(filepath))
    
    with open(filepath, 'r') as f:
        grasp_data = json.load(f)
    
    if not grasp_data:
        raise ValueError("Empty grasp data")
    
    # Sort by score and take n_best
    sorted_grasps = sorted(grasp_data, key=lambda g: g.get('score', 0), reverse=True)[:n_best]
    
    # Extract data and convert to arrays
    tf_matrices = []
    widths = []
    scores = []
    
    for grasp in sorted_grasps:
        pos = np.array([grasp['position']['x'], grasp['position']['y'], grasp['position']['z']])
        approach = np.array([grasp['approach']['x'], grasp['approach']['y'], grasp['approach']['z']])
        binormal = np.array([grasp['binormal']['x'], grasp['binormal']['y'], grasp['binormal']['z']])
        axis = np.array([grasp['axis']['x'], grasp['axis']['y'], grasp['axis']['z']])
        
        # Create rotation matrix
        rot_matrix = np.column_stack((approach, binormal, axis))
        
        # Create transformation matrix (4x4)
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = rot_matrix
        tf_matrix[:3, 3] = pos
        
        tf_matrices.append(tf_matrix)
        widths.append(grasp['width'])
        scores.append(grasp['score'])
    
    return np.array(tf_matrices), np.array(widths), np.array(scores)

def _predict(
    item_cloud,
    env_cloud,
    limits,
    rotations,
    config,
    logger=None,
    top_n=3,
    n_best=1,
    timeout=90,
    top_down_grasp=False,
    vis_block=False,
):
    """
    Predict grasp poses using GPD algorithm.
    
    Args:
        item_cloud: Point cloud of the object to grasp
        env_cloud: Point cloud of the environment
        limits: Limits for grasp search
        rotations: Rotation angles to consider
        config: Configuration parameters
        logger: Logger instance
        top_n: Number of grasps per rotation
        n_best: Number of best grasps to return
        timeout: Timeout in seconds
        top_down_grasp: Whether to prioritize top-down grasps
        vis_block: Block for visualization
        
    Returns:
        Tuple containing transformation matrices, widths, and scores
    """
    if logger is None:
        logger = Logger()
    
    # Create temporary directory for files
    temp_dir = tempfile.mkdtemp()
    # temp_dir = "/workspace"  # For testing purposes, use a fixed directory
    item_file = os.path.join(temp_dir, "item_cloud.pcd")
    env_file = os.path.join(temp_dir, "env_cloud.pcd")
    output_file = os.path.join(temp_dir, "detected_grasps.json")
    
    # Save point clouds to temporary files
    logger.info("Saving item cloud to {0}".format(item_file))
    if not _save_point_cloud_to_file(item_cloud, item_file):
        logger.error("Failed to save item cloud. App,py")
        return np.array([]), np.array([]), np.array([])
    
    logger.info("Saving environment cloud to {0}".format(env_file))
    if not _save_point_cloud_to_file(env_cloud, env_file):
        logger.error("Failed to save environment cloud")
        return np.array([]), np.array([]), np.array([])
    
    # Create symlinks for GPD to find the files
    workspace_dir = "/workspace"
    if not os.path.exists(workspace_dir):
        try:
            os.makedirs(workspace_dir)
        except OSError:
            # Directory might have been created by another process
            if not os.path.isdir(workspace_dir):
                raise
    
    workspace_item_path = os.path.join(workspace_dir, "item_cloud.pcd")
    workspace_env_path = os.path.join(workspace_dir, "env_cloud.pcd")
    workspace_output_path = os.path.join(workspace_dir, "detected_grasps.json")
    
    # Create symlinks or copy files to workspace
    try:
        if os.path.exists(workspace_item_path):
            os.unlink(workspace_item_path)
        if os.path.exists(workspace_env_path):
            os.unlink(workspace_env_path)
        
        # Either create symlinks or copy files
        try:
            os.symlink(item_file, workspace_item_path)
            os.symlink(env_file, workspace_env_path)
        except OSError:
            # If symlinks not supported, copy files
            import shutil
            shutil.copy2(item_file, workspace_item_path)
            shutil.copy2(env_file, workspace_env_path)
            
        logger.info("Files prepared for GPD")
    except Exception as e:
        logger.error("Error linking files: {0}".format(e))
        return np.array([]), np.array([]), np.array([])
    
    # Run the GPD service
    logger.info("Running GPD service to detect grasps")
    
    # Prepare the command with adjusted parameters
    cmd = ["python", "/workspace/run_ros_serv.py"]
    
    try:
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for the process with timeout
        wait_time = 0
        check_interval = 1.0
        while process.poll() is None and wait_time < timeout:
            time.sleep(check_interval)
            wait_time += check_interval
            
            # Check if output file exists and has content
            if os.path.exists(workspace_output_path):
                try:
                    with open(workspace_output_path, 'r') as f:
                        if json.load(f):  # If valid JSON and not empty
                            logger.info("Grasp detection completed")
                            break
                except ValueError:
                    # File exists but is not valid JSON yet
                    pass
        
        # Check if process timed out
        if wait_time >= timeout:
            logger.warning("GPD process timed out after {0} seconds".format(timeout))
            process.terminate()
            return np.array([]), np.array([]), np.array([])
        
        # Get process output
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error("GPD process failed with return code {0}".format(process.returncode))
            logger.error("Stderr: {0}".format(stderr.decode('utf-8')))
            return np.array([]), np.array([]), np.array([])
        
        execution_time = time.time() - start_time
        logger.info("GPD completed in {0:.2f} seconds".format(execution_time))
        
        # Read output and parse grasps
        if os.path.exists(workspace_output_path):
            return _read_grasp_file(workspace_output_path, n_best)
        else:
            logger.error("Output file not found: {0}".format(workspace_output_path))
            return np.array([]), np.array([]), np.array([])
            
    except Exception as e:
        logger.error("Error running GPD: {0}".format(e))
        return np.array([]), np.array([]), np.array([])

def predict_full_grasp(
    item_cloud,
    env_cloud,
    config,
    logger=None,
    rotation_resolution=24,
    top_n=3,
    n_best=1,
    timeout=90,
    vis_block=False,
):
    """
    Predict a grasp position from the item point cloud and its environment.
    :param item_cloud: the point cloud of the item to be grasped
    :param env_cloud: the point cloud of the environment (typically within some radius)
    :param config: config
    :param logger: for logging
    :param rotation_resolution: number of different angles for grasp detection
    :param top_n: number of different grasps per angle
    :param n_best: number of best grasps to return
    :param timeout: seconds for http request timeout
    :param vis_block: visualize grasp before returning
    :return: transformation matrix, width, and scores of best found grasp
    """
    if logger is None:
        logger = Logger("GPDInterface")
    
    logger.info("Starting grasp prediction with GPD")
    
    # Generate rotation angles based on rotation_resolution
    rotations = np.linspace(0, 2*np.pi, rotation_resolution, endpoint=False)
    
    # Define search limits (can be adjusted based on config)
    limits = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    
    # Call the prediction function
    tf_matrices, widths, scores = _predict(
        item_cloud=item_cloud,
        env_cloud=env_cloud,
        limits=limits,
        rotations=rotations,
        config=config,
        logger=logger,
        top_n=top_n,
        n_best=n_best,
        timeout=timeout,
        vis_block=vis_block
    )
    
    if len(tf_matrices) == 0:
        logger.warning("No valid grasps found")
        # Return empty arrays with correct shapes
        return np.empty((0, 4, 4)), np.empty(0), np.empty(0)
    
    logger.info("Found {0} valid grasps".format(len(tf_matrices)))
    return tf_matrices, widths, scores

if __name__ == "__main__":
    # Example usage (will run if script is executed directly)
    print("GPD interface module loaded")
    print("This module provides an interface to the GPD algorithm.")
    print("Import this module and use the predict_full_grasp function to detect grasps.")
