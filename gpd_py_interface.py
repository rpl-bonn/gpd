#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPD Python interface module.
This module provides a simple API for using GPD directly from Python code.
"""
from __future__ import division, print_function

import os
import sys
import json
import time
import numpy as np
import subprocess
import tempfile
try:
    import pcl
    has_pcl = True
except ImportError:
    has_pcl = False

# Define a simple PointCloud class
class PointCloud(object):
    """Simple point cloud class with PCL integration."""
    def __init__(self, points=None):
        """Initialize with points (numpy array of shape Nx3)."""
        self.points = points if points is not None else np.array([], dtype=np.float32)
    
    def to_pcl(self):
        """Convert to PCL point cloud."""
        if not has_pcl:
            raise ImportError("PCL Python bindings not available")
        
        cloud = pcl.PointCloud()
        cloud.from_array(self.points.astype(np.float32))
        return cloud
    
    @classmethod
    def from_pcl(cls, pcl_cloud):
        """Create from PCL point cloud."""
        return cls(np.array(pcl_cloud.to_array()))
    
    @classmethod
    def from_file(cls, file_path):
        """Load from PCD file."""
        if not has_pcl:
            raise ImportError("PCL Python bindings required for loading PCD files")
        if not os.path.exists(file_path):
            raise IOError("File not found: {}".format(file_path))
        
        pcl_cloud = pcl.load(file_path)
        return cls.from_pcl(pcl_cloud)
    
    def save(self, file_path):
        """Save to PCD file."""
        if not has_pcl:
            raise ImportError("PCL Python bindings required for saving PCD files")
        
        pcl_cloud = self.to_pcl()
        pcl.save(pcl_cloud, file_path)

# Define a simple configuration class
class Config(object):
    """Configuration class for GPD."""
    def __init__(self, **kwargs):
        """Initialize with keyword arguments."""
        self.__dict__.update(kwargs)
    
    def __str__(self):
        """String representation."""
        return str(self.__dict__)

# Define a simple logger class
class Logger(object):
    """Simple logger class."""
    def __init__(self, name="GPD"):
        """Initialize with name."""
        self.name = name
    
    def info(self, message):
        """Log info message."""
        print("[INFO] {}: {}".format(self.name, message))
    
    def warning(self, message):
        """Log warning message."""
        print("[WARNING] {}: {}".format(self.name, message))
    
    def error(self, message):
        """Log error message."""
        print("[ERROR] {}: {}".format(self.name, message))

class GPDInterface(object):
    """Interface for GPD grasp detection."""
    
    def __init__(self, workspace_dir="/workspace", logger=None):
        """Initialize the interface."""
        self.workspace_dir = workspace_dir
        self.logger = logger if logger is not None else Logger("GPDInterface")
        
        # Create workspace directory if it doesn't exist
        if not os.path.exists(self.workspace_dir):
            try:
                os.makedirs(self.workspace_dir)
            except OSError:
                if not os.path.isdir(self.workspace_dir):
                    raise
    
    def predict_full_grasp(
        self,
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
        
        Args:
            item_cloud: the point cloud of the item to be grasped
            env_cloud: the point cloud of the environment (typically within some radius)
            config: configuration object
            logger: logger object
            rotation_resolution: number of different angles for grasp detection
            top_n: number of different grasps per angle
            n_best: number of best grasps to return
            timeout: seconds for process timeout
            vis_block: visualize grasp before returning
            
        Returns:
            Tuple of:
                - transformation matrices (Nx4x4 numpy array)
                - grasp widths (N numpy array)
                - grasp scores (N numpy array)
        """
        if logger is None:
            logger = self.logger
        
        logger.info("Starting grasp prediction with GPD")
        
        # Generate rotation angles based on rotation_resolution
        rotations = np.linspace(0, 2*np.pi, rotation_resolution, endpoint=False)
        
        # Define search limits
        limits = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
        
        # Call the prediction function
        return self._predict(
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
    
    def _predict(
        self,
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
            logger = self.logger
        
        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        item_file = os.path.join(temp_dir, "item_cloud.pcd")
        env_file = os.path.join(temp_dir, "env_cloud.pcd")
        output_file = os.path.join(temp_dir, "detected_grasps.json")
        
        # Save point clouds to temporary files
        logger.info("Saving item cloud to {}".format(item_file))
        try:
            item_cloud.save(item_file)
        except Exception as e:
            logger.error("Failed to save item cloud: {}".format(e))
            return np.array([]), np.array([]), np.array([])
        
        logger.info("Saving environment cloud to {}".format(env_file))
        try:
            env_cloud.save(env_file)
        except Exception as e:
            logger.error("Failed to save environment cloud: {}".format(e))
            return np.array([]), np.array([]), np.array([])
        
        # Create workspace files
        workspace_item_path = os.path.join(self.workspace_dir, "item_cloud.pcd")
        workspace_env_path = os.path.join(self.workspace_dir, "env_cloud.pcd")
        workspace_output_path = os.path.join(self.workspace_dir, "detected_grasps.json")
        
        # Copy files to workspace
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
            logger.error("Error linking files: {}".format(e))
            return np.array([]), np.array([]), np.array([])
        
        # Run the GPD service
        logger.info("Running GPD service to detect grasps")
        
        # Prepare the command with adjusted parameters
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = ["python", os.path.join(script_dir, "run_ros_serv.py")]
        
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
                logger.warning("GPD process timed out after {} seconds".format(timeout))
                process.terminate()
                return np.array([]), np.array([]), np.array([])
            
            # Get process output
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error("GPD process failed with return code {}".format(process.returncode))
                logger.error("Stderr: {}".format(stderr.decode('utf-8')))
                return np.array([]), np.array([]), np.array([])
            
            execution_time = time.time() - start_time
            logger.info("GPD completed in {:.2f} seconds".format(execution_time))
            
            # Read output and parse grasps
            if os.path.exists(workspace_output_path):
                return self._read_grasp_file(workspace_output_path, n_best)
            else:
                logger.error("Output file not found: {}".format(workspace_output_path))
                return np.array([]), np.array([]), np.array([])
                
        except Exception as e:
            logger.error("Error running GPD: {}".format(e))
            return np.array([]), np.array([]), np.array([])
    
    def _read_grasp_file(self, filepath, n_best=1):
        """Read grasp data from a JSON file."""
        if not os.path.exists(filepath):
            raise IOError("Grasp file not found: {}".format(filepath))
        
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


# Create a default interface instance for easy import
gpd_interface = GPDInterface()

# Default predict function that uses the default interface
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
    :param config: config object
    :param logger: for logging
    :param rotation_resolution: number of different angles for grasp detection
    :param top_n: number of different grasps per angle
    :param n_best: number of best grasps to return
    :param timeout: seconds for process timeout
    :param vis_block: visualize grasp before returning
    :return: transformation matrix, width, and scores of best found grasp
    """
    return gpd_interface.predict_full_grasp(
        item_cloud,
        env_cloud,
        config,
        logger,
        rotation_resolution,
        top_n,
        n_best,
        timeout,
        vis_block,
    )

# Simple example of usage
if __name__ == "__main__":
    print("GPD Python Interface")
    print("This module provides functions to detect grasps using GPD.")
    print("")
    print("Sample usage:")
    print("  from gpd_py_interface import PointCloud, Config, predict_full_grasp")
    print("  item_cloud = PointCloud.from_file('item_cloud.pcd')")
    print("  env_cloud = PointCloud.from_file('env_cloud.pcd')")
    print("  config = Config(gripper_width=0.08)")
    print("  tf_matrices, widths, scores = predict_full_grasp(item_cloud, env_cloud, config)")
    print("")
    print("See direct_client.py for a more complete example.")
