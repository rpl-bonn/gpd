#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
External client to use GPD service from outside the Docker container.
This script sends point cloud data to the GPD service running in the Docker container
and returns the grasp poses.
"""
from __future__ import division, print_function

import os
import sys
import json
import time
import argparse
import numpy as np

try:
    import requests
    has_requests = True
except ImportError:
    has_requests = False
    print("Warning: requests module not found. HTTP API will not be available.")
    print("Please install it with: pip install requests")

try:
    # Try to import PCL for point cloud handling
    import pcl
    has_pcl = True
except ImportError:
    has_pcl = False
    print("Warning: PCL Python bindings not found. Will use placeholder point clouds.")
    print("Consider installing python-pcl for full functionality.")


def load_point_cloud(file_path):
    """Load a point cloud from a PCD file."""
    print("Loading point cloud from: {}".format(file_path))
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("Error: File not found: {}".format(file_path))
        return None
    
    # Check if PCL is available
    if not has_pcl:
        print("PCL module not available. Cannot load point cloud.")
        return None
    
    # Load point cloud using PCL
    try:
        pcl_cloud = pcl.load(file_path)
        points = np.array(pcl_cloud.to_array(), dtype=np.float32)
        print("Loaded {} points from PCD file".format(points.shape[0]))
        return pcl_cloud
    except Exception as e:
        print("Error loading PCD file: {}".format(e))
        return None


def call_gpd_service(item_cloud_path, env_cloud_path, server_url="http://localhost:5000/predict",
                     rotation_resolution=24, top_n=3, n_best=1):
    """Call the GPD service running in Docker via HTTP API."""
    if not has_requests:
        print("Error: requests module not available. Cannot call GPD service.")
        return None
    
    # Check if files exist
    if not os.path.exists(item_cloud_path):
        print("Error: Item cloud file not found: {}".format(item_cloud_path))
        return None
    
    if not os.path.exists(env_cloud_path):
        print("Error: Environment cloud file not found: {}".format(env_cloud_path))
        return None
    
    # Prepare files for upload
    print("Sending request to GPD service at {}...".format(server_url))
    try:
        files = {
            'item_cloud': open(item_cloud_path, 'rb'),
            'env_cloud': open(env_cloud_path, 'rb')
        }
        
        # Prepare parameters
        data = {
            'rotation_resolution': rotation_resolution,
            'top_n': top_n,
            'n_best': n_best
        }
        
        # Send the request
        start_time = time.time()
        response = requests.post(server_url, files=files, data=data)
        elapsed_time = time.time() - start_time
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("Request completed in {:.2f} seconds".format(elapsed_time))
            return result
        else:
            print("Error: Request failed with status code {}".format(response.status_code))
            print("Response: {}".format(response.text))
            return None
    except Exception as e:
        print("Error calling GPD service: {}".format(e))
        return None
    finally:
        # Close the files
        files['item_cloud'].close()
        files['env_cloud'].close()


def process_grasp_result(result):
    """Process and display the grasp results."""
    if result is None or 'error' in result:
        print("No valid grasps found or error occurred")
        if result is not None and 'error' in result:
            print("Error: {}".format(result['error']))
        return
    
    # Extract data
    tf_matrices = result.get('tf_matrices', [])
    widths = result.get('widths', [])
    scores = result.get('scores', [])
    
    if not tf_matrices:
        print("No valid grasps found")
        return
    
    print("\nFound {} grasp(s):".format(len(tf_matrices)))
    
    for i, (tf_matrix, width, score) in enumerate(zip(tf_matrices, widths, scores)):
        print("\nGrasp {}:".format(i+1))
        print("  Score: {:.4f}".format(score))
        print("  Width: {:.4f}".format(width))
        
        # Extract position from the transformation matrix
        position = [tf_matrix[0][3], tf_matrix[1][3], tf_matrix[2][3]]
        print("  Position: [{:.4f}, {:.4f}, {:.4f}]".format(*position))
        
        # Display the transformation matrix
        print("  Transformation Matrix:")
        for row in tf_matrix:
            print("    " + " ".join(["{:7.4f}".format(val) for val in row]))


def convert_to_numpy_array(result):
    """Convert the JSON result to NumPy arrays."""
    if result is None or 'error' in result:
        return None, None, None
    
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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="External client for GPD service")
    parser.add_argument("--item", type=str, required=True,
                        help="Path to item point cloud PCD file")
    parser.add_argument("--env", type=str, required=True,
                        help="Path to environment point cloud PCD file")
    parser.add_argument("--server", type=str, default="http://localhost:5000/predict",
                      help="URL of the GPD service (default: http://localhost:5000/predict)")
    parser.add_argument("--rot_res", type=int, default=24,
                      help="Rotation resolution (default: 24)")
    parser.add_argument("--top_n", type=int, default=3,
                      help="Top N grasps per angle (default: 3)")
    parser.add_argument("--n_best", type=int, default=3,
                      help="N best grasps to return (default: 3)")
    parser.add_argument("--json", action="store_true",
                      help="Output results as JSON")
    args = parser.parse_args()
    
    # Call GPD service
    result = call_gpd_service(
        item_cloud_path=args.item,
        env_cloud_path=args.env,
        server_url=args.server,
        rotation_resolution=args.rot_res,
        top_n=args.top_n,
        n_best=args.n_best
    )
    
    if args.json:
        # Output as JSON
        if result:
            print(json.dumps(result, indent=2))
    else:
        # Display the results
        process_grasp_result(result)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
