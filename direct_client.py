#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct client for GPD interface without HTTP.
This script demonstrates how to use the GPD interface directly.
"""
from __future__ import division, print_function

import os
import sys
import argparse
import numpy as np
import time

# Import app.py for the grasp prediction functions
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Try to import PCL for point cloud handling
try:
    import pcl
    has_pcl = True
except ImportError:
    print("PCL Python bindings not found. Will use placeholder point clouds.")
    has_pcl = False

# Import the needed functions from app.py
try:
    from app import PointCloud, Config, Logger, predict_full_grasp
    print("Successfully imported functions from app.py")
except ImportError as e:
    print("Error importing from app.py: {}".format(e))
    sys.exit(1)

def create_example_point_cloud():
    """Create an example point cloud for testing."""
    print("Creating example point cloud")
    
    # Create a simple cube point cloud
    points = []
    for x in np.linspace(-0.1, 0.1, 10):
        for y in np.linspace(-0.1, 0.1, 10):
            for z in np.linspace(-0.1, 0.1, 10):
                points.append([x, y, z])
    
    # Convert to numpy array
    points = np.array(points, dtype=np.float32)
    
    # Create PointCloud object
    point_cloud = PointCloud(points)
    
    print("Created point cloud with {} points".format(len(points)))
    return point_cloud

def load_point_cloud(file_path):
    """Load a point cloud from a PCD file."""
    print("Loading point cloud from: {}".format(file_path))
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("Error: File not found: {}".format(file_path))
        return None
    
    # Check if PCL is available
    if not has_pcl:
        print("PCL module not available. Using example point cloud instead.")
        return create_example_point_cloud()
    
    # Load point cloud using PCL
    try:
        pcl_cloud = pcl.load(file_path)
        points = np.array(pcl_cloud.to_array(), dtype=np.float32)
        print("Loaded {} points from PCD file".format(points.shape[0]))
        
        # Create PointCloud object
        point_cloud = PointCloud(points)
        return point_cloud
    except Exception as e:
        print("Error loading PCD file: {}".format(e))
        print("Using example point cloud instead.")
        return create_example_point_cloud()

def print_grasp_results(tf_matrices, widths, scores):
    """Print grasp results in a nice format."""
    if len(tf_matrices) == 0:
        print("\nNo valid grasps found.")
        return
    
    print("\nFound {} grasp(s):".format(len(tf_matrices)))
    
    for i, (matrix, width, score) in enumerate(zip(tf_matrices, widths, scores)):
        print("\nGrasp {}:".format(i+1))
        print("  Score: {:.4f}".format(score))
        print("  Width: {:.4f}".format(width))
        print("  Position: [{:.4f}, {:.4f}, {:.4f}]".format(
            matrix[0, 3], matrix[1, 3], matrix[2, 3]
        ))
        
        # Extract rotation as a quaternion (simplified)
        print("  Transformation Matrix:")
        for row in matrix:
            print("    " + " ".join(["{:7.4f}".format(x) for x in row]))

def main():
    """Main function to test the GPD interface."""
    parser = argparse.ArgumentParser(description="Direct GPD Interface Client")
    parser.add_argument("--item", type=str, default="item_cloud.pcd", help="Item point cloud file path")
    parser.add_argument("--env", type=str, default="env_cloud.pcd", help="Environment point cloud file path")
    parser.add_argument("--vis", action="store_true", help="Visualize grasp before returning")
    parser.add_argument("--rot_res", type=int, default=24, help="Rotation resolution")
    parser.add_argument("--top_n", type=int, default=3, help="Number of grasps per angle")
    parser.add_argument("--n_best", type=int, default=1, help="Number of best grasps to return")
    args = parser.parse_args()
    
    # Create a logger
    logger = Logger("GPD_DirectClient")
    logger.info("Starting direct GPD interface test")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build absolute paths for the input files if they're not absolute already
    item_file = args.item if os.path.isabs(args.item) else os.path.join(current_dir, args.item)
    env_file = args.env if os.path.isabs(args.env) else os.path.join(current_dir, args.env)
    
    # Load point clouds
    item_cloud = load_point_cloud(item_file)
    if item_cloud is None:
        return 1
    
    env_cloud = load_point_cloud(env_file)
    if env_cloud is None:
        return 1
    
    # Create a simple configuration
    config = Config(
        # Add any configuration parameters here
        gripper_width=0.08,  # Maximum gripper width in meters
        finger_depth=0.05,   # Finger depth in meters
        hand_depth=0.10,     # Hand depth in meters
        object_min_height=0.005,  # Minimum height of objects to grasp
    )
    
    # Call the predict_full_grasp function
    logger.info("Calling predict_full_grasp")
    print("\nPredicting grasps... (this may take a while)")
    
    start_time = time.time()
    breakpoint()
    try:
        tf_matrices, widths, grasp_scores = predict_full_grasp(
            item_cloud=item_cloud,
            env_cloud=env_cloud,
            config=config,
            logger=logger,
            rotation_resolution=args.rot_res,
            top_n=args.top_n,
            n_best=args.n_best,
            vis_block=args.vis
        )
        
        execution_time = time.time() - start_time
        logger.info("Grasp prediction completed in {:.2f} seconds".format(execution_time))
        
        # Print results
        print_grasp_results(tf_matrices, widths, grasp_scores)
        
    except Exception as e:
        logger.error("Error during grasp prediction: {}".format(e))
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("Test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
