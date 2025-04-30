#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of using the GPD Python interface.
This script shows how to use the GPD Python interface with example point clouds.
"""
from __future__ import division, print_function

import os
import sys
import argparse
import numpy as np

# Try to import the GPD Python interface
try:
    from gpd_py_interface import PointCloud, Config, Logger, predict_full_grasp
except ImportError:
    # If the module is not installed, add the current directory to the path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    
    # Try again
    try:
        from gpd_py_interface import PointCloud, Config, Logger, predict_full_grasp
    except ImportError as e:
        print("Error importing GPD Python interface: {}".format(e))
        sys.exit(1)

def main():
    """Run the example."""
    parser = argparse.ArgumentParser(description="GPD Python Interface Example")
    parser.add_argument("--item", type=str, default="workspace/item_cloud.pcd",
                        help="Path to item point cloud PCD file")
    parser.add_argument("--env", type=str, default="workspace/env_cloud.pcd",
                        help="Path to environment point cloud PCD file")
    parser.add_argument("--n_best", type=int, default=3,
                        help="Number of best grasps to return")
    parser.add_argument("--vis", action="store_true",
                        help="Visualize grasps (if supported)")
    args = parser.parse_args()
    
    # Create logger
    logger = Logger("GPDExample")
    logger.info("Starting GPD example")
    
    # Load point clouds
    try:
        # Make paths absolute if they are not already
        item_path = args.item if os.path.isabs(args.item) else os.path.join(os.getcwd(), args.item)
        env_path = args.env if os.path.isabs(args.env) else os.path.join(os.getcwd(), args.env)
        
        logger.info("Loading item point cloud from {}".format(item_path))
        item_cloud = PointCloud.from_file(item_path)
        
        logger.info("Loading environment point cloud from {}".format(env_path))
        env_cloud = PointCloud.from_file(env_path)
    except Exception as e:
        logger.error("Error loading point clouds: {}".format(e))
        return 1
    
    # Create configuration
    config = Config(
        gripper_width=0.08,
        finger_depth=0.05,
        hand_depth=0.10,
        object_min_height=0.005,
    )
    
    # Predict grasps
    logger.info("Predicting grasps")
    
    try:
        tf_matrices, widths, scores = predict_full_grasp(
            item_cloud=item_cloud,
            env_cloud=env_cloud,
            config=config,
            logger=logger,
            rotation_resolution=12,  # Lower resolution for faster results
            top_n=3,
            n_best=args.n_best,
            vis_block=args.vis,
        )
    except Exception as e:
        logger.error("Error predicting grasps: {}".format(e))
        return 1
    
    # Print results
    if len(tf_matrices) == 0:
        logger.warning("No grasps found")
        return 1
    
    logger.info("Found {} grasps".format(len(tf_matrices)))
    
    for i, (tf_matrix, width, score) in enumerate(zip(tf_matrices, widths, scores)):
        print("\nGrasp {}:".format(i + 1))
        print("  Score: {:.4f}".format(score))
        print("  Width: {:.4f}".format(width))
        print("  Position: [{:.4f}, {:.4f}, {:.4f}]".format(
            tf_matrix[0, 3], tf_matrix[1, 3], tf_matrix[2, 3]
        ))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
