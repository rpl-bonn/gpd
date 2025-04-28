#!/usr/bin/env python3
"""
Simple client for GPD grasp detection service
This script demonstrates making requests to the GPD service with PLY point clouds
"""

import os
import sys
import argparse
import requests
import json
import time
import numpy as np

# Try to import visualization libraries
try:
    import open3d as o3d
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Open3D not available. Visualization will be disabled.")
    VISUALIZATION_AVAILABLE = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPD Client Example")
    parser.add_argument("--host", type=str, default="localhost", 
                        help="GPD server host (default: localhost)")
    parser.add_argument("--port", type=int, default=5000, 
                        help="GPD server port (default: 5000)")
    parser.add_argument("--cloud", type=str, default="item_cloud.ply", 
                        help="Path to point cloud file (default: item_cloud.ply)")
    parser.add_argument("--env", type=str, default="env_cloud.ply",
                        help="Optional environment cloud (default: env_cloud.ply)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout in seconds (default: 30)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results (requires Open3D)")
    return parser.parse_args()

def send_request(url, cloud_file, timeout=30, indices=None):
    """Send point cloud to GPD server"""
    files = {"cloud": open(cloud_file, "rb")}
    data = {"timeout": str(timeout)}
    
    if indices:
        data["indices"] = ",".join(map(str, indices))
    
    try:
        print(f"Sending request to {url} with file {cloud_file}")
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, "response") and e.response:
            print(f"Response: {e.response.text}")
        return None

def visualize_results(cloud_file, grasps):
    """Visualize point cloud and detected grasps"""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available (Open3D required)")
        return
    
    # Load the point cloud
    cloud = o3d.io.read_point_cloud(cloud_file)
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)
    
    # Add coordinate frames for each grasp
    for grasp in grasps:
        # Extract position
        pos = [
            grasp["position"]["x"],
            grasp["position"]["y"],
            grasp["position"]["z"]
        ]
        
        # Create a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05,  # Size of the frame
            origin=pos   # Position of the frame
        )
        vis.add_geometry(frame)
    
    # Set visualization options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main():
    args = parse_args()
    
    # Construct the URL
    url = f"http://{args.host}:{args.port}/detect_grasps"
    
    # Check if the cloud file exists
    if not os.path.exists(args.cloud):
        print(f"Error: Could not find cloud file: {args.cloud}")
        return 1
    
    print(f"\nRequesting grasps for: {args.cloud}")
    start_time = time.time()
    
    # Send the request
    response = send_request(url, args.cloud, args.timeout)
    elapsed = time.time() - start_time
    
    if not response:
        print("Error: Failed to get response from server")
        return 1
    
    # Check for errors
    if "error" in response:
        print(f"Error from server: {response['error']}")
        return 1
    
    # Process results
    grasps = response.get("grasps", [])
    count = len(grasps)
    
    print(f"\nReceived {count} grasps in {elapsed:.2f} seconds")
    
    if count > 0:
        # Display best grasp
        best_grasp = max(grasps, key=lambda g: g["score"])
        
        print("\nBest grasp:")
        print(f"  Position: ({best_grasp['position']['x']:.3f}, {best_grasp['position']['y']:.3f}, {best_grasp['position']['z']:.3f})")
        print(f"  Approach: ({best_grasp['orientation']['x']:.3f}, {best_grasp['orientation']['y']:.3f}, {best_grasp['orientation']['z']:.3f})")
        print(f"  Width: {best_grasp['width']:.3f}")
        print(f"  Score: {best_grasp['score']:.3f}")
        
        # Visualize if requested
        if args.visualize:
            print("\nVisualizing results (close window to exit)")
            visualize_results(args.cloud, grasps)
    else:
        print("No grasps found.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
