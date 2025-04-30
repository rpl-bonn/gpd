#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for GPD interface.
This script demonstrates how to use the GPD interface to predict grasps.
"""
from __future__ import division, print_function

import os
import sys
import numpy as np
import argparse
import time

# Try to import PCL for point cloud handling
try:
    import pcl
except ImportError:
    print("PCL Python bindings not found. Will use placeholder point clouds.")
    pcl = None

# Import app.py for the grasp prediction functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import PointCloud, Config, Logger, predict_full_grasp

def load_point_cloud(file_path):
    """Load a point cloud from a PCD or PLY file."""
    print("Loading point cloud from: {}".format(file_path))
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("Error: File not found: {}".format(file_path))
        return None
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.pcd':
        if pcl is None:
            print("PCL module required to load PCD files")
            return None
        try:
            cloud = pcl.load(file_path)
            points = np.array(cloud.to_array())
            print("Loaded {} points from PCD file".format(points.shape[0]))
            
            # Create PointCloud object
            point_cloud = PointCloud(points)
            return point_cloud
        except Exception as e:
            print("Error loading PCD file: {}".format(e))
            return None
    elif ext == '.ply':
        print("PLY files need to be converted to PCD first")
        # You can add PLY loading functionality here
        return None
    else:
        print("Unsupported file format: {}".format(ext))
        return None

def main():
    """Main function to test the GPD interface."""
    parser = argparse.ArgumentParser(description="Test GPD interface")
    parser.add_argument("--item", type=str, default="item_cloud.pcd", help="Item point cloud file path")
    parser.add_argument("--env", type=str, default="env_cloud.pcd", help="Environment point cloud file path")
    parser.add_argument("--vis", action="store_true", help="Visualize grasp before returning")
    parser.add_argument("--rot_res", type=int, default=24, help="Rotation resolution")
    parser.add_argument("--top_n", type=int, default=3, help="Number of grasps per angle")
    parser.add_argument("--n_best", type=int, default=1, help="Number of best grasps to return")
    args = parser.parse_args()
    
    # Create a logger
    logger = Logger("GPDTest")
    logger.info("Starting GPD interface test")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build absolute paths for the input files
    item_file = os.path.join(current_dir, args.item)
    env_file = os.path.join(current_dir, args.env)
    
    # Load point clouds
    item_cloud = load_point_cloud(item_file)
    if item_cloud is None:
        logger.error("Failed to load item point cloud")
        return 1
    
    env_cloud = load_point_cloud(env_file)
    if env_cloud is None:
        logger.error("Failed to load environment point cloud")
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
    start_time = time.time()
    
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
    if len(tf_matrices) > 0:
        logger.info("Found {} grasp(s)".format(len(tf_matrices)))
        
        for i, (matrix, width, score) in enumerate(zip(tf_matrices, widths, grasp_scores)):
            logger.info("Grasp {}: Score = {:.4f}, Width = {:.4f}".format(i+1, score, width))
            logger.info("Transformation matrix:")
            print(matrix)
            print()
    else:
        logger.warning("No valid grasps found")
    
    logger.info("Test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
