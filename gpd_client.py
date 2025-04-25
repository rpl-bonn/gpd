#!/usr/bin/env python3
"""
GPD Client - External interface to the GPD Docker container
This script provides a simple interface to send point clouds to the GPD Docker container
and retrieve grasp poses.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import requests
import open3d as o3d
import subprocess
from typing import Tuple, List, Dict, Union, Optional
import matplotlib.pyplot as plt

class GPDClient:
    """Client to interact with the GPD Docker container"""
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """
        Initialize the GPD client.
        
        Args:
            server_url: URL of the GPD server
        """
        self.server_url = server_url
        self.last_visualization_path = None
    
    def check_server_health(self) -> Dict:
        """
        Check if the server is running.
        
        Returns:
            Dict with server status information
        """
        try:
            response = requests.get(f"{self.server_url}/health")
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": "Could not connect to server"}
    
    def start_docker_container(self, docker_script_path: str = "./run_docker_1.sh") -> None:
        """
        Start the Docker container.
        
        Args:
            docker_script_path: Path to the Docker run script
        """
        print(f"Starting Docker container using {docker_script_path}...")
        process = subprocess.Popen(
            ["bash", docker_script_path], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        # Wait for container to start up
        time.sleep(5)
        
        # Execute the GPD Docker app inside the container
        container_id = self._get_container_id()
        if container_id:
            print(f"Container started with ID: {container_id}")
            print("Starting GPD Docker app inside container...")
            cmd = "python3 /workspace/gpd_docker_app.py"
            subprocess.Popen(
                ["docker", "exec", "-d", container_id, "bash", "-c", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for the app to start
            time.sleep(3)
            
            # Check if server is running
            health = self.check_server_health()
            if health.get("status") == "ok":
                print("GPD Docker app started successfully!")
            else:
                print("Warning: GPD Docker app might not have started correctly.")
        else:
            print("Warning: Could not find container ID, the app may need to be started manually.")
    
    def _get_container_id(self) -> Optional[str]:
        """Get the container ID of the running GPD Docker container"""
        try:
            output = subprocess.check_output(
                ["docker", "ps", "--filter", "ancestor=hiveroboticsai/gpd:latest", "--format", "{{.ID}}"],
                text=True
            )
            container_id = output.strip()
            return container_id if container_id else None
        except subprocess.SubprocessError:
            return None
    
    def detect_grasps(self, 
                     item_cloud: Union[str, o3d.geometry.PointCloud], 
                     env_cloud: Union[str, o3d.geometry.PointCloud, None] = None,
                     best_only: bool = False,
                     visualize: bool = True) -> Dict:
        """
        Detect grasps on a point cloud.
        
        Args:
            item_cloud: Path to item point cloud file or Open3D point cloud object
            env_cloud: Path to environment point cloud file or Open3D point cloud object
            best_only: Whether to return only the best grasp
            visualize: Whether to visualize the results
            
        Returns:
            Dict with detection results
        """
        # Check if server is running
        health = self.check_server_health()
        if health.get("status") != "ok":
            return {"success": False, "message": "Server is not running"}
        
        # Prepare request data
        files = {}
        data = {
            "best_only": str(best_only).lower(),
            "visualize": str(visualize).lower()
        }
        
        # Handle item cloud
        if isinstance(item_cloud, str):
            # Item cloud is a file path
            if os.path.exists(item_cloud):
                print(f"Using item cloud from path: {item_cloud}")
                # Check if the file is accessible from inside the Docker container
                if item_cloud.startswith('/workspace'):
                    # Path is accessible inside the container
                    data["item_cloud_path"] = item_cloud
                else:
                    # Path may not be accessible inside the container, upload the file
                    files["item_cloud"] = open(item_cloud, 'rb')
            else:
                return {"success": False, "message": f"Item cloud file not found: {item_cloud}"}
        else:
            # Item cloud is an Open3D point cloud object
            print("Processing provided item point cloud object")
            # Save to temp file and upload
            temp_file = f"/tmp/{time.time()}_item.ply"
            o3d.io.write_point_cloud(temp_file, item_cloud)
            files["item_cloud"] = open(temp_file, 'rb')
        
        # Handle environment cloud (optional)
        if env_cloud is not None:
            if isinstance(env_cloud, str):
                # Environment cloud is a file path
                if os.path.exists(env_cloud):
                    print(f"Using environment cloud from path: {env_cloud}")
                    # Check if the file is accessible from inside the Docker container
                    if env_cloud.startswith('/workspace'):
                        # Path is accessible inside the container
                        data["env_cloud_path"] = env_cloud
                    else:
                        # Path may not be accessible inside the container, upload the file
                        files["env_cloud"] = open(env_cloud, 'rb')
                else:
                    print(f"Warning: Environment cloud file not found: {env_cloud}")
            else:
                # Environment cloud is an Open3D point cloud object
                print("Processing provided environment point cloud object")
                # Save to temp file and upload
                temp_file = f"/tmp/{time.time()}_env.ply"
                o3d.io.write_point_cloud(temp_file, env_cloud)
                files["env_cloud"] = open(temp_file, 'rb')
        
        # Set visualization path if requested
        if visualize:
            visualization_path = f"/workspace/grasp_visualization_{time.time()}.ply"
            data["visualization_path"] = visualization_path
            self.last_visualization_path = visualization_path
        
        # Send request to server
        try:
            print("Sending grasp detection request...")
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/detect_grasps",
                files=files,
                data=data
            )
            elapsed_time = time.time() - start_time
            print(f"Request completed in {elapsed_time:.2f} seconds")
            
            # Close any open files
            for file in files.values():
                file.close()
            
            # Clean up temp files
            for key, file in files.items():
                file_path = file.name
                if os.path.exists(file_path) and file_path.startswith('/tmp/'):
                    os.remove(file_path)
            
            # Parse response
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    if best_only:
                        print(f"Best grasp found with score: {result.get('score')}")
                    else:
                        num_grasps = result.get("num_grasps", 0)
                        print(f"Found {num_grasps} grasp candidates")
                    
                    if visualize and "visualization_path" in result:
                        print(f"Visualization saved to {result['visualization_path']}")
                else:
                    print(f"Failed to find grasps: {result.get('message', 'Unknown error')}")
                
                return result
            else:
                print(f"Error: Server returned status code {response.status_code}")
                return {"success": False, "message": f"Server error: {response.text}"}
        
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to server")
            return {"success": False, "message": "Could not connect to server"}
        except Exception as e:
            print(f"Error: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def visualize_results(self, result: Dict = None) -> None:
        """
        Visualize the detection results.
        
        Args:
            result: Detection result dictionary (optional, uses last result if None)
        """
        if not result and not self.last_visualization_path:
            print("No visualization available")
            return
        
        visualization_path = result.get("visualization_path", self.last_visualization_path)
        
        if not visualization_path:
            print("No visualization path available")
            return
        
        if not os.path.exists(visualization_path):
            # Try with /workspace prefix (docker container path)
            container_id = self._get_container_id()
            if container_id:
                # Copy the visualization file from the container to the host
                host_path = f"/tmp/grasp_visualization_{time.time()}.ply"
                subprocess.run(
                    ["docker", "cp", f"{container_id}:{visualization_path}", host_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if os.path.exists(host_path):
                    visualization_path = host_path
                    print(f"Copied visualization from container to {host_path}")
                else:
                    print(f"Could not copy visualization from container: {visualization_path}")
                    return
        
        # Load and visualize the point cloud
        try:
            print(f"Loading visualization from {visualization_path}...")
            vis_cloud = o3d.io.read_point_cloud(visualization_path)
            print("Displaying visualization...")
            o3d.visualization.draw_geometries([vis_cloud])
        except Exception as e:
            print(f"Error visualizing results: {str(e)}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPD Client for grasp detection")
    parser.add_argument('--item', type=str, required=True, help='Path to item point cloud file')
    parser.add_argument('--env', type=str, help='Path to environment point cloud file')
    parser.add_argument('--best-only', action='store_true', help='Return only the best grasp')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--start-docker', action='store_true', help='Start the Docker container')
    parser.add_argument('--server-url', type=str, default='http://localhost:5000', help='URL of the GPD server')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create GPD client
    client = GPDClient(server_url=args.server_url)
    
    # Start Docker container if requested
    if args.start_docker:
        client.start_docker_container()
    
    # Check if server is running
    health = client.check_server_health()
    if health.get("status") != "ok":
        print("Server is not running. Use --start-docker to start it.")
        return
    
    # Detect grasps
    result = client.detect_grasps(
        item_cloud=args.item,
        env_cloud=args.env,
        best_only=args.best_only,
        visualize=not args.no_vis
    )
    
    # Display detailed results
    if result.get("success", False):
        if args.best_only:
            print("\nBest grasp details:")
            print(f"Score: {result.get('score')}")
            print(f"Width: {result.get('width')}")
            print("Transform matrix:")
            print(np.array(result.get("transform")))
        else:
            print("\nGrasp candidates:")
            scores = result.get("scores", [])
            widths = result.get("widths", [])
            transforms = result.get("transforms", [])
            
            for i in range(min(5, len(scores))):  # Show top 5 grasps
                print(f"\nGrasp {i+1}:")
                print(f"  Score: {scores[i]}")
                print(f"  Width: {widths[i]}")
                print("  Transform matrix:")
                print(np.array(transforms[i]))
        
        # Visualize if requested
        if not args.no_vis:
            print("\nPress Q to close the visualization window when done.")
            client.visualize_results(result)
    else:
        print(f"\nFailed to find grasps: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()
