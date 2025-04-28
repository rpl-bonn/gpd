#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for using the GPD Client API.
This script demonstrates how to use the GPD Client API to predict grasp poses.
"""
from __future__ import division, print_function

import os
import sys
import argparse
import numpy as np
import time

# Add the current directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    # Import the GPD Client API
    from gpd_client_api import predict_grasps, GPDClient
    
    # Import visualization utility if available
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
except ImportError as e:
    print(f"Error importing GPD Client API: {e}")
    sys.exit(1)


def visualize_grasp(tf_matrix, width):
    """Visualize a grasp with matplotlib."""
    if not has_matplotlib:
        print("Matplotlib not available. Cannot visualize grasp.")
        return
    
    # Extract position and orientation
    position = tf_matrix[:3, 3]
    approach = tf_matrix[:3, 0]  # First column is approach vector
    binormal = tf_matrix[:3, 1]  # Second column is binormal vector
    axis = tf_matrix[:3, 2]      # Third column is hand axis vector
    
    # Create a figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the grasp center
    ax.scatter([position[0]], [position[1]], [position[2]], color='red', s=100, marker='o')
    
    # Plot the approach, binormal, and axis vectors
    scale = width / 2
    ax.quiver(position[0], position[1], position[2],
              approach[0], approach[1], approach[2],
              color='red', length=scale, arrow_length_ratio=0.2)
    
    ax.quiver(position[0], position[1], position[2],
              binormal[0], binormal[1], binormal[2],
              color='green', length=scale, arrow_length_ratio=0.2)
    
    ax.quiver(position[0], position[1], position[2],
              axis[0], axis[1], axis[2],
              color='blue', length=scale, arrow_length_ratio=0.2)
    
    # Plot the gripper endpoints
    p1 = position + binormal * (width / 2)
    p2 = position - binormal * (width / 2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', lw=2)
    
    # Add arrows at the gripper endpoints
    arrow_scale = scale * 0.5
    ax.quiver(p1[0], p1[1], p1[2],
              approach[0], approach[1], approach[2],
              color='gray', length=arrow_scale, arrow_length_ratio=0.2)
    
    ax.quiver(p2[0], p2[1], p2[2],
              approach[0], approach[1], approach[2],
              color='gray', length=arrow_scale, arrow_length_ratio=0.2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Grasp Visualization')
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GPD Client API Example")
    parser.add_argument("--item", type=str, default="item_cloud.pcd",
                        help="Path to item point cloud PCD file")
    parser.add_argument("--env", type=str, default="env_cloud.pcd",
                        help="Path to environment point cloud PCD file")
    parser.add_argument("--server", type=str, default="http://localhost:5000/predict",
                        help="URL of the GPD service")
    parser.add_argument("--rot_res", type=int, default=24,
                        help="Rotation resolution")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Top N grasps per angle")
    parser.add_argument("--n_best", type=int, default=3,
                        help="N best grasps to return")
    parser.add_argument("--vis", action="store_true",
                        help="Visualize the grasp")
    args = parser.parse_args()
    
    # Create a custom client
    client = GPDClient(server_url=args.server)
    
    # Make the path absolute if relative
    item_path = args.item if os.path.isabs(args.item) else os.path.join(os.getcwd(), args.item)
    env_path = args.env if os.path.isabs(args.env) else os.path.join(os.getcwd(), args.env)
    
    print(f"Item cloud: {item_path}")
    print(f"Environment cloud: {env_path}")
    
    # Predict grasps
    print(f"Calling GPD service at {args.server}...")
    start_time = time.time()
    tf_matrices, widths, scores = client.predict_grasps(
        item_path,
        env_path,
        rotation_resolution=args.rot_res,
        top_n=args.top_n,
        n_best=args.n_best
    )
    elapsed_time = time.time() - start_time
    
    # Display results
    if len(tf_matrices) == 0:
        print("No valid grasps found.")
        return 1
    
    print(f"\nFound {len(tf_matrices)} grasp(s) in {elapsed_time:.2f} seconds:")
    
    for i, (tf_matrix, width, score) in enumerate(zip(tf_matrices, widths, scores)):
        print(f"\nGrasp {i+1}:")
        print(f"  Score: {score:.4f}")
        print(f"  Width: {width:.4f}")
        
        # Position (translation part of the transformation matrix)
        position = tf_matrix[0:3, 3]
        print(f"  Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        
        # Orientation (rotation part of the transformation matrix)
        approach = tf_matrix[:3, 0]  # First column
        binormal = tf_matrix[:3, 1]  # Second column
        axis = tf_matrix[:3, 2]      # Third column
        
        print(f"  Approach: [{approach[0]:.4f}, {approach[1]:.4f}, {approach[2]:.4f}]")
        print(f"  Binormal: [{binormal[0]:.4f}, {binormal[1]:.4f}, {binormal[2]:.4f}]")
        print(f"  Axis:     [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
        
        # Full transformation matrix
        print("  Transformation Matrix:")
        for row in tf_matrix:
            print("    " + " ".join([f"{val:7.4f}" for val in row]))
    
    # Visualize the first grasp if requested
    if args.vis and len(tf_matrices) > 0:
        print("\nVisualizing the best grasp...")
        visualize_grasp(tf_matrices[0], widths[0])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
