#!/usr/bin/env python3
"""
Example client for the GraspNet ROS interface.
This script demonstrates how to use the GraspNet ROS interface to detect grasps.

Usage:
    - Direct ROS method:
      python ros_client.py --mode ros --item item_cloud.ply --env env_cloud.ply
      
    - HTTP server method (requires app_ros.py to be running):
      python ros_client.py --mode http --item item_cloud.ply --env env_cloud.ply
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import requests
import json
import time

# Add the current directory to the Python path to import the ROS interface module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graspnet_ros_interface import predict_grasps_ros, get_best_grasp_ros, visualize_grasps

def detect_grasps_ros(item_cloud_path, env_cloud_path=None, best_only=False, visualize=True):
    """
    Detect grasps using the direct ROS interface.
    
    Args:
        item_cloud_path: Path to the item point cloud file
        env_cloud_path: Path to the environment point cloud file (optional)
        best_only: Whether to return only the best grasp
        visualize: Whether to visualize the results
        
    Returns:
        If best_only is True, returns the best grasp as a dict
        Otherwise, returns a tuple of (transformation matrices, widths, scores)
    """
    print(f"Loading item cloud from {item_cloud_path}...")
    item_cloud = o3d.io.read_point_cloud(item_cloud_path)
    
    if env_cloud_path:
        print(f"Loading environment cloud from {env_cloud_path}...")
        env_cloud = o3d.io.read_point_cloud(env_cloud_path)
    else:
        print("No environment cloud provided, using empty environment...")
        env_cloud = o3d.geometry.PointCloud()
    
    print(f"Item cloud has {len(item_cloud.points)} points")
    print(f"Environment cloud has {len(env_cloud.points)} points")
    
    # Start timing
    start_time = time.time()
    
    # Detect grasps
    if best_only:
        print("Getting the best grasp...")
        result = get_best_grasp_ros(item_cloud, env_cloud)
        
        if result["success"]:
            print(f"Best grasp found with score: {result['score']}")
            print(f"Width: {result['width']}")
            print(f"Transform matrix:")
            print(np.array(result["transform"]))
            
            # Visualize the best grasp
            if visualize:
                print("Visualizing the best grasp...")
                tf_matrices = np.array([result["transform"]])
                widths = np.array([result["width"]])
                scores = np.array([result["score"]])
                
                visualization_path = os.path.splitext(item_cloud_path)[0] + "_best_grasp.ply"
                visualize_grasps(item_cloud, env_cloud, tf_matrices, widths, scores, visualization_path)
                print(f"Visualization saved to {visualization_path}")
                
            return result
        else:
            print(f"Failed to find grasps: {result.get('message', 'Unknown error')}")
            return result
    else:
        print("Detecting all grasps...")
        tf_matrices, widths, scores = predict_grasps_ros(item_cloud, env_cloud)
        
        # Display results
        if len(scores) > 0:
            print(f"Found {len(scores)} grasp candidates")
            for i in range(min(5, len(scores))):  # Show top 5 grasps
                print(f"Grasp {i+1}:")
                print(f"  Score: {scores[i]}")
                print(f"  Width: {widths[i]}")
            
            # Visualize grasps
            if visualize:
                print("Visualizing grasps...")
                visualization_path = os.path.splitext(item_cloud_path)[0] + "_grasps.ply"
                visualize_grasps(item_cloud, env_cloud, tf_matrices, widths, scores, visualization_path)
                print(f"Visualization saved to {visualization_path}")
                
            return tf_matrices, widths, scores
        else:
            print("No grasps found.")
            return np.array([]), np.array([]), np.array([])
            
    print(f"Grasp detection completed in {time.time() - start_time:.2f} seconds")

def detect_grasps_http(item_cloud_path, env_cloud_path=None, server_url='http://localhost:5000', best_only=False):
    """
    Detect grasps using the HTTP server interface.
    
    Args:
        item_cloud_path: Path to the item point cloud file
        env_cloud_path: Path to the environment point cloud file (optional)
        server_url: URL of the server
        best_only: Whether to return only the best grasp
        
    Returns:
        The server's JSON response
    """
    print(f"Using HTTP server at {server_url}")
    
    # Prepare files for upload
    files = {
        'item_cloud': open(item_cloud_path, 'rb')
    }
    
    if env_cloud_path:
        files['env_cloud'] = open(env_cloud_path, 'rb')
    
    # Prepare form data
    data = {
        'get_best_only': 'true' if best_only else 'false'
    }
    
    # Start timing
    start_time = time.time()
    
    # Send the request
    print("Sending request to server...")
    try:
        response = requests.post(f"{server_url}/detect_grasps", files=files, data=data)
        print(f"Request completed in {time.time() - start_time:.2f} seconds")
        
        # Close files
        for file in files.values():
            file.close()
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            
            if 'success' in result and result['success']:
                if best_only:
                    # Display the best grasp
                    print(f"Best grasp found with score: {result['score']}")
                    print(f"Width: {result['width']}")
                    print(f"Transform matrix:")
                    print(np.array(result["transform"]))
                else:
                    # Display grasp statistics
                    scores = result.get('scores', [])
                    print(f"Found {len(scores)} grasp candidates")
                    for i in range(min(5, len(scores))):  # Show top 5 grasps
                        print(f"Grasp {i+1}:")
                        print(f"  Score: {scores[i]}")
                        print(f"  Width: {result['widths'][i]}")
                
                return result
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")
                return result
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(response.text)
            return {
                "success": False,
                "message": f"Server error: {response.status_code}"
            }
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {
            "success": False,
            "message": f"Connection error: {str(e)}"
        }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GraspNet ROS Client')
    parser.add_argument('--mode', choices=['ros', 'http'], default='ros',
                        help='Interface mode: direct ROS or HTTP server')
    parser.add_argument('--item', required=True, help='Path to item point cloud')
    parser.add_argument('--env', help='Path to environment point cloud')
    parser.add_argument('--best-only', action='store_true', help='Get only the best grasp')
    parser.add_argument('--server', default='http://localhost:5000', help='Server URL (for HTTP mode)')
    parser.add_argument('--no-visualization', action='store_true', help='Disable visualization (for ROS mode)')
    args = parser.parse_args()
    
    # Check if item point cloud file exists
    if not os.path.exists(args.item):
        print(f"Error: Item point cloud file {args.item} not found")
        return 1
        
    # Check if environment point cloud file exists, if specified
    if args.env and not os.path.exists(args.env):
        print(f"Error: Environment point cloud file {args.env} not found")
        return 1
    
    # Detect grasps using the specified mode
    if args.mode == 'ros':
        detect_grasps_ros(
            args.item,
            args.env,
            best_only=args.best_only,
            visualize=not args.no_visualization
        )
    else:  # HTTP mode
        detect_grasps_http(
            args.item,
            args.env,
            server_url=args.server,
            best_only=args.best_only
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
