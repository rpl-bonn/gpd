#!/usr/bin/env python3
"""
GPD Client Script - Example for calling the GPD ROS-based API from outside the Docker container.

This script demonstrates how to:
1. Send PLY point clouds to the GPD server running in Docker
2. Process the returned grasp poses
3. Optionally visualize the results

Usage:
    python client_test.py [--host HOST] [--port PORT] [--visualize]
"""

import os
import sys
import argparse
import numpy as np
import requests
import json
import time

try:
    import open3d as o3d
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Open3D not found. Visualization will be disabled.")
    VISUALIZATION_AVAILABLE = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GPD Client for PLY point clouds')
    parser.add_argument('--host', type=str, default='localhost',
                        help='GPD server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5000,
                        help='GPD server port (default: 5000)')
    parser.add_argument('--item', type=str, default='item_cloud.ply',
                        help='Item point cloud file (default: item_cloud.ply)')
    parser.add_argument('--env', type=str, default='env_cloud.ply',
                        help='Environment point cloud file (default: env_cloud.ply)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in seconds (default: 120)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results with Open3D')
    return parser.parse_args()

def send_point_cloud(cloud_file, server_url, timeout=120, indices=None, max_retries=3):
    """Send point cloud to GPD server with retry mechanism"""
    print(f"Sending point cloud: {cloud_file}")
    
    # First check if the server is reachable and ROS system is ready
    try:
        health_check = requests.get(f"{server_url.rsplit('/', 1)[0]}/health", timeout=5)
        if health_check.status_code == 200:
            print("Server health check passed")
        else:
            print(f"Warning: Server health check failed with status {health_check.status_code}")
    except requests.exceptions.RequestException:
        print("Could not reach server for health check. Proceeding anyway...")
    
    # Set up retry with increasing timeouts
    retries = 0
    current_timeout = timeout
    
    while retries <= max_retries:
        try:
            # Open the file for each attempt to prevent file handle issues
            with open(cloud_file, 'rb') as f:
                files = {'cloud': f}
                data = {'timeout': str(current_timeout)}
                
                # If indices are provided, add them to the request
                if indices is not None:
                    data['indices'] = ','.join([str(idx) for idx in indices])
                    print(f"Using {len(indices)} sample indices")
                
                print(f"Attempt {retries + 1} with timeout {current_timeout} seconds")
                response = requests.post(server_url, files=files, data=data, timeout=current_timeout)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Server returned error: {response.status_code}")
                    print(f"Response: {response.text}")
                    
            # If we got here, there was an issue with the response
            retries += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Error sending request (attempt {retries + 1}): {e}")
            retries += 1
        
        if retries <= max_retries:
            # Increase timeout for next retry
            current_timeout *= 1.5
            current_timeout = int(current_timeout)
            print(f"Retrying in 3 seconds with increased timeout: {current_timeout} seconds")
            time.sleep(3)
        else:
            print("Max retries reached. Giving up.")
            return None

def create_grasp_geometry(grasp, hand_depth=0.05, finger_width=0.01, hand_height=0.02, color=None):
    """Create visualization geometry for a single grasp"""
    if color is None:
        color = [0, 0.7, 0]  # Default green color
        
    # Extract grasp parameters
    position = [grasp['position']['x'], grasp['position']['y'], grasp['position']['z']]
    approach = [grasp['orientation']['x'], grasp['orientation']['y'], grasp['orientation']['z']]
    
    # Normalize approach vector
    approach_norm = np.linalg.norm(approach)
    if approach_norm > 0:
        approach = [v / approach_norm for v in approach]
    
    # Compute orthogonal vectors to create a coordinate frame
    # First, find a vector not parallel to approach
    if abs(approach[0]) < 0.9:
        temp = [1, 0, 0]
    else:
        temp = [0, 1, 0]
    
    # Compute binormal (perpendicular to approach)
    binormal = np.cross(approach, temp)
    binormal_norm = np.linalg.norm(binormal)
    if binormal_norm > 0:
        binormal = [v / binormal_norm for v in binormal]
    
    # Compute normal (perpendicular to approach and binormal)
    normal = np.cross(approach, binormal)
    
    # Create grasp coordinate frame visualization
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=position)
    
    # Create a box for the hand
    hand = o3d.geometry.TriangleMesh.create_box(
        width=hand_height, 
        height=finger_width, 
        depth=hand_depth
    )
    
    # Transform the hand to align with the grasp
    R = np.array([normal, binormal, approach]).T
    hand.rotate(R, center=[0, 0, 0])
    hand.translate(position)
    
    # Set hand color
    hand.paint_uniform_color(color)
    
    return [frame, hand]

def visualize_grasps(cloud_file, grasps, best_only=False):
    """Visualize point cloud and detected grasps"""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install Open3D for visualization.")
        return
    
    # Load point cloud
    cloud = o3d.io.read_point_cloud(cloud_file)
    
    # Create visualization geometries
    geometries = [cloud]
    
    # Add grasp visualizations
    if best_only and grasps:
        # Only visualize the highest scoring grasp
        best_grasp = max(grasps, key=lambda g: g['score'])
        geometries.extend(create_grasp_geometry(best_grasp, color=[0, 1, 0]))  # Green for best
    else:
        # Visualize all grasps with varying colors based on score
        scores = [g['score'] for g in grasps]
        min_score = min(scores) if scores else 0
        score_range = max(scores) - min_score if scores else 1
        
        for i, grasp in enumerate(grasps):
            # Normalize score between 0 and 1
            normalized_score = (grasp['score'] - min_score) / score_range if score_range > 0 else 0.5
            
            # Create color: blue to red gradient based on score
            color = [normalized_score, 0, 1 - normalized_score]
            geometries.extend(create_grasp_geometry(grasp, color=color))
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GPD Grasp Visualization")
    
    # Add geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Configure view
    vis.get_render_option().background_color = [0.2, 0.2, 0.2]
    vis.get_render_option().point_size = 2
    vis.get_render_option().show_coordinate_frame = True
    
    # Reset view to show all geometries
    vis.reset_view_point(True)
    
    # Run visualization
    print("Visualizing grasps. Close window to continue.")
    vis.run()
    vis.destroy_window()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Construct server URL
    server_url = f"http://{args.host}:{args.port}/detect_grasps"
    
    # Check if the point cloud files exist
    if not os.path.exists(args.item):
        print(f"Error: Item point cloud file not found: {args.item}")
        return 1
    
    # Prepare files for the request
    start_time = time.time()
    print(f"Requesting grasp detection from {server_url}")
    
    # First check if health endpoint is available
    try:
        health_url = f"http://{args.host}:{args.port}/health"
        print(f"Checking server health at {health_url}")
        health_response = requests.get(health_url, timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Server health: {health_data}")
            if health_data.get('subscribers', 0) == 0:
                print("WARNING: No ROS subscribers connected to the cloud topic!")
                print("The ROS grasp detection node may not be running.")
        else:
            print(f"Health check failed with status {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Could not check server health: {e}")
    
    # Send the request with point cloud
    grasps_response = send_point_cloud(args.item, server_url, timeout=args.timeout)
    
    elapsed = time.time() - start_time
    
    # Check if we got a valid response
    if not grasps_response:
        print("Error: Failed to get response from server")
        return 1
    
    # Check for error in response
    if 'error' in grasps_response:
        print(f"Error from server: {grasps_response['error']}")
        return 1
    
    # Process and display results
    if 'grasps' in grasps_response:
        grasps = grasps_response['grasps']
        count = grasps_response.get('count', len(grasps))
        
        print(f"\nDetected {count} grasps in {elapsed:.2f} seconds")
        
        if count > 0:
            # Find the best grasp
            best_grasp = max(grasps, key=lambda g: g['score'])
            
            print("\nBest grasp:")
            print(f"  Position: ({best_grasp['position']['x']:.3f}, {best_grasp['position']['y']:.3f}, {best_grasp['position']['z']:.3f})")
            print(f"  Approach: ({best_grasp['orientation']['x']:.3f}, {best_grasp['orientation']['y']:.3f}, {best_grasp['orientation']['z']:.3f})")
            print(f"  Width: {best_grasp['width']:.3f}")
            print(f"  Score: {best_grasp['score']:.3f}")
            
            # Show top 3 grasps if we have more than 3
            if count > 1:
                print("\nTop 3 grasps:")
                
                # Sort by score
                sorted_grasps = sorted(grasps, key=lambda g: g['score'], reverse=True)
                
                # Print top 3 (or fewer if we don't have 3)
                for i, grasp in enumerate(sorted_grasps[:3]):
                    print(f"  #{i+1} - Score: {grasp['score']:.3f}, Position: ({grasp['position']['x']:.3f}, {grasp['position']['y']:.3f}, {grasp['position']['z']:.3f})")
            
            # Visualize results if requested
            if args.visualize and VISUALIZATION_AVAILABLE:
                visualize_grasps(args.item, grasps)
        else:
            print("No valid grasps found.")
    else:
        print("Error: Unexpected response format")
        print(json.dumps(grasps_response, indent=2))
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
