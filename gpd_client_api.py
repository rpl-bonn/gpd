#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPD Client API Module
This module provides a simple API to interact with the GPD service running in a Docker container.
"""
from __future__ import division, print_function

import os
import sys
import json
import time
import subprocess
import numpy as np

try:
    import requests
    has_requests = True
except ImportError:
    has_requests = False

# Try to import point cloud libraries
try:
    import pcl
    has_pcl = True
except ImportError:
    has_pcl = False


class GPDClient(object):
    """Client for GPD service."""
    
    def __init__(self, server_url="http://localhost:5000/predict",
                 container_name="gpd_container",
                 docker_script="run_docker_new.sh"):
        """Initialize the GPD client."""
        self.server_url = server_url
        self.container_name = container_name
        self.docker_script = docker_script
        
        # Get script directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if requests module is available
        if not has_requests:
            print("Warning: requests module not available. HTTP API will not be available.")
    
    def ensure_docker_running(self, timeout=60):
        """Ensure that the Docker container is running."""
        # Check if Docker is available
        try:
            subprocess.check_output(["docker", "info"], stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Docker is not available or not running")
            return False
        
        # Check if the container is running
        try:
            output = subprocess.check_output(
                ["docker", "ps", "-q", "--filter", f"name={self.container_name}"]
            ).decode().strip()
            
            if output:
                # Container is already running
                return True
            
            # Check if container exists but is stopped
            output = subprocess.check_output(
                ["docker", "ps", "-aq", "--filter", f"name={self.container_name}"]
            ).decode().strip()
            
            if output:
                print(f"Container {self.container_name} exists but is not running. Starting it...")
                subprocess.check_call(["docker", "start", self.container_name])
            else:
                # Container does not exist, start it with the script
                docker_script_path = os.path.join(self.script_dir, self.docker_script)
                if os.path.exists(docker_script_path):
                    print(f"Starting Docker container using {self.docker_script}...")
                    subprocess.Popen(["bash", docker_script_path])
                else:
                    print(f"Error: Docker script {self.docker_script} not found")
                    return False
            
            # Wait for the container to be ready
            start_time = time.time()
            while True:
                try:
                    if time.time() - start_time > timeout:
                        print(f"Error: Timeout waiting for container {self.container_name}")
                        return False
                    
                    # Check if container is running
                    output = subprocess.check_output(
                        ["docker", "ps", "-q", "--filter", f"name={self.container_name}"]
                    ).decode().strip()
                    
                    if output:
                        # Check if the service is ready
                        try:
                            response = requests.get(self.server_url.replace("/predict", "/health"), timeout=2)
                            if response.status_code == 200:
                                print("Docker container is ready")
                                return True
                        except:
                            # Service not ready yet
                            pass
                    
                    time.sleep(3)
                except KeyboardInterrupt:
                    print("Container startup interrupted")
                    return False
            
        except subprocess.CalledProcessError:
            print("Error checking Docker container status")
            return False
    
    def predict_grasps(self, item_cloud, env_cloud=None, rotation_resolution=24, top_n=3, n_best=1):
        """
        Predict grasps using the GPD service.
        
        Args:
            item_cloud: Path to PCD file or PCL point cloud object for the item
            env_cloud: Path to PCD file or PCL point cloud object for the environment (optional)
            rotation_resolution: Number of rotation angles to try
            top_n: Number of grasps per rotation angle
            n_best: Number of best grasps to return
        
        Returns:
            Tuple of (transformation matrices, widths, scores) as NumPy arrays
        """
        # Check if requests module is available
        if not has_requests:
            print("Error: requests module not available. Cannot call GPD service.")
            return np.array([]), np.array([]), np.array([])
        
        # Ensure Docker is running
        if not self.ensure_docker_running():
            print("Error: Could not ensure Docker container is running")
            return np.array([]), np.array([]), np.array([])
        
        # Handle item point cloud
        item_cloud_path = self._get_point_cloud_path(item_cloud, "item_cloud")
        if not item_cloud_path:
            return np.array([]), np.array([]), np.array([])
        
        # Handle environment point cloud
        env_cloud_path = self._get_point_cloud_path(env_cloud, "env_cloud")
        if not env_cloud_path:
            return np.array([]), np.array([]), np.array([])
        
        # Call the GPD service
        result = self._call_gpd_service(
            item_cloud_path,
            env_cloud_path,
            rotation_resolution,
            top_n,
            n_best
        )
        
        # Convert result to NumPy arrays
        return self._convert_result_to_numpy(result)
    
    def _get_point_cloud_path(self, point_cloud, default_name):
        """
        Get the path to a point cloud file.
        
        Args:
            point_cloud: Path to PCD file or PCL point cloud object
            default_name: Default name for the point cloud file
        
        Returns:
            Path to the point cloud file
        """
        if point_cloud is None:
            # Create an empty point cloud if None is provided
            if has_pcl:
                cloud = pcl.PointCloud()
                cloud.from_array(np.zeros((10, 3), dtype=np.float32))
                temp_path = os.path.join(self.script_dir, f"{default_name}_temp.pcd")
                pcl.save(cloud, temp_path)
                return temp_path
            else:
                print(f"Error: PCL not available and {default_name} is None")
                return None
        
        if isinstance(point_cloud, str):
            # Point cloud is a file path
            if os.path.exists(point_cloud):
                return point_cloud
            else:
                print(f"Error: Point cloud file not found: {point_cloud}")
                return None
        
        if has_pcl and isinstance(point_cloud, pcl.PointCloud):
            # Point cloud is a PCL point cloud object
            temp_path = os.path.join(self.script_dir, f"{default_name}_temp.pcd")
            pcl.save(point_cloud, temp_path)
            return temp_path
        
        # Unsupported type
        print(f"Error: Unsupported point cloud type: {type(point_cloud)}")
        return None
    
    def _call_gpd_service(self, item_path, env_path, rotation_resolution, top_n, n_best):
        """Call the GPD service via HTTP API."""
        try:
            files = {
                'item_cloud': open(item_path, 'rb'),
                'env_cloud': open(env_path, 'rb')
            }
            
            data = {
                'rotation_resolution': rotation_resolution,
                'top_n': top_n,
                'n_best': n_best
            }
            
            print(f"Sending request to GPD service at {self.server_url}...")
            response = requests.post(self.server_url, files=files, data=data)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: Request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"Error calling GPD service: {e}")
            return None
        finally:
            # Close the files
            if 'files' in locals():
                files['item_cloud'].close()
                files['env_cloud'].close()
    
    def _convert_result_to_numpy(self, result):
        """Convert the JSON result to NumPy arrays."""
        if result is None or 'error' in result:
            return np.array([]), np.array([]), np.array([])
        
        # Extract data
        tf_matrices = result.get('tf_matrices', [])
        widths = result.get('widths', [])
        scores = result.get('scores', [])
        
        if not tf_matrices:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to NumPy arrays
        tf_matrices_np = np.array(tf_matrices)
        widths_np = np.array(widths)
        scores_np = np.array(scores)
        
        return tf_matrices_np, widths_np, scores_np


# Create a default client instance for easy import
gpd_client = GPDClient()


def predict_grasps(item_cloud, env_cloud=None, rotation_resolution=24, top_n=3, n_best=1):
    """
    Predict grasps using the GPD service.
    This is a convenience function that uses the default GPDClient instance.
    
    Args:
        item_cloud: Path to PCD file or PCL point cloud object for the item
        env_cloud: Path to PCD file or PCL point cloud object for the environment (optional)
        rotation_resolution: Number of rotation angles to try
        top_n: Number of grasps per rotation angle
        n_best: Number of best grasps to return
    
    Returns:
        Tuple of (transformation matrices, widths, scores) as NumPy arrays
    """
    return gpd_client.predict_grasps(
        item_cloud,
        env_cloud,
        rotation_resolution,
        top_n,
        n_best
    )


# Example usage
if __name__ == "__main__":
    print("GPD Client API Module")
    print("This module provides functions to interact with the GPD service.")
    print("")
    print("Example usage:")
    print("  from gpd_client_api import predict_grasps")
    print("  tf_matrices, widths, scores = predict_grasps('item_cloud.pcd', 'env_cloud.pcd')")
    print("  # Or create a custom client")
    print("  from gpd_client_api import GPDClient")
    print("  client = GPDClient(server_url='http://localhost:5000/predict')")
    print("  tf_matrices, widths, scores = client.predict_grasps('item_cloud.pcd', 'env_cloud.pcd')")
    
    # If run as a script with arguments, use those to call the service
    if len(sys.argv) > 1:
        item_cloud = sys.argv[1]
        env_cloud = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Calling GPD service with {item_cloud} and {env_cloud}...")
        tf_matrices, widths, scores = predict_grasps(item_cloud, env_cloud)
        
        print(f"Found {len(tf_matrices)} grasp(s)")
        for i, (tf_matrix, width, score) in enumerate(zip(tf_matrices, widths, scores)):
            print(f"Grasp {i+1}:")
            print(f"  Score: {score:.4f}")
            print(f"  Width: {width:.4f}")
            print(f"  Position: [{tf_matrix[0, 3]:.4f}, {tf_matrix[1, 3]:.4f}, {tf_matrix[2, 3]:.4f}]")
