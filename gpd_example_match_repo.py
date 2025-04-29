#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script showing how to use the GPD interface in the same way as the stretch-compose repository.
This script demonstrates using the interface with the same pattern as in:
https://github.com/rpl-bonn/stretch-compose/blob/Yasmin/source/scripts/my_robot_scripts/graspnet_planning.py
"""

import os
import sys
import numpy as np
import argparse

# Add the current directory to the path to import gpd_external_client
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import from our GPD external client
from gpd_external_client import PointCloud, Config, GPDInterface, Logger

def main():
    """Main function that mimics the graspnet_planning.py from stretch-compose."""
    parser = argparse.ArgumentParser(description="GPD Interface Example")
    parser.add_argument("--item", type=str, default="item_cloud.pcd",
                        help="Path to item point cloud file")
    parser.add_argument("--env", type=str, default="env_cloud.pcd",
                        help="Path to environment point cloud file")
    parser.add_argument("--vis", action="store_true",
                        help="Visualize grasps")
    args = parser.parse_args()
    
    # Create a logger
    logger = Logger("Example")
    
    # Load the point clouds
    logger.info(f"Loading point cloud from {args.item}")
    try:
        item_cloud = PointCloud.from_file(args.item)
    except Exception as e:
        logger.error(f"Failed to load item point cloud: {e}")
        return 1
    
    logger.info(f"Loading point cloud from {args.env}")
    try:
        env_cloud = PointCloud.from_file(args.env)
    except Exception as e:
        logger.error(f"Failed to load environment point cloud: {e}")
        return 1
    
    # Create config
    config = Config(
        gripper_width=0.08,
        finger_depth=0.05,
        hand_depth=0.10,
        object_min_height=0.005
    )
    
    # Create the interface
    graspnet_interface = GPDInterface(logger=logger)
    
    # Call predict_full_grasp with the same pattern as in stretch-compose repository
    logger.info("Predicting grasps...")
    
    # This exactly matches the pattern from the repository:
    # https://github.com/rpl-bonn/stretch-compose/blob/Yasmin/source/scripts/my_robot_scripts/graspnet_planning.py#L127
    tf_matrices, widths, grasp_scores = graspnet_interface.predict_full_grasp(
        item_cloud,
        env_cloud,
        config,
        rotation_resolution=24,
        top_n=3,
        n_best=60,
        vis_block=args.vis,
    )
    
    # Print the results
    if len(tf_matrices) == 0:
        logger.info("No valid grasps found")
        return 1
    
    logger.info(f"Found {len(tf_matrices)} grasp poses")
    
    # Print the best grasp
    best_idx = np.argmax(grasp_scores) if len(grasp_scores) > 0 else 0
    
    logger.info(f"Best grasp (index {best_idx}):")
    logger.info(f"  Score: {grasp_scores[best_idx]:.4f}")
    logger.info(f"  Width: {widths[best_idx]:.4f}")
    logger.info(f"  Position: [{tf_matrices[best_idx][0,3]:.4f}, {tf_matrices[best_idx][1,3]:.4f}, {tf_matrices[best_idx][2,3]:.4f}]")
    
    # Display the transformation matrix
    logger.info("  Transformation Matrix:")
    for row in tf_matrices[best_idx]:
        print("    " + " ".join([f"{val:7.4f}" for val in row]))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
