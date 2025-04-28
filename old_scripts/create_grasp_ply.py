#!/usr/bin/env python3
"""
Create a PLY file with clear visual representations of the detected grasps.
This script loads grasp data and creates explicit 3D models of grippers at each grasp pose.
"""

import open3d as o3d
import numpy as np
import os
import copy
import matplotlib.pyplot as plt

def create_gripper_mesh(transform_matrix, width, score, score_min, score_max):
    """Create a gripper mesh at the given pose with the given width."""
    # Create a more visible gripper representation
    
    # Base dimensions
    base_width = 0.03
    base_height = 0.015
    base_depth = 0.03
    
    # Finger dimensions
    finger_width = 0.01
    finger_height = 0.04
    finger_depth = 0.02
    
    # Create base
    base = o3d.geometry.TriangleMesh.create_box(
        width=base_width, 
        height=base_height, 
        depth=base_depth
    )
    base.translate([-base_width/2, -base_height/2, -base_depth/2])
    
    # Create fingers
    left_finger = o3d.geometry.TriangleMesh.create_box(
        width=finger_width, 
        height=finger_height, 
        depth=finger_depth
    )
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=finger_width, 
        height=finger_height, 
        depth=finger_depth
    )
    
    # Position fingers at correct width
    left_finger.translate([-width/2 - finger_width/2, 0, -finger_depth/2])
    right_finger.translate([width/2 - finger_width/2, 0, -finger_depth/2])
    
    # Combine into a single mesh
    gripper = base
    gripper += left_finger
    gripper += right_finger
    
    # Color based on score (red for highest score, blue for lowest)
    score_range = score_max - score_min if score_max > score_min else 1.0
    norm_score = (score - score_min) / score_range
    
    # Use colormap for coloring (red to blue)
    cmap = plt.cm.jet
    color = cmap(norm_score)[:3]  # Get RGB from colormap
    
    # Apply color
    gripper.paint_uniform_color(color)
    
    # Transform to correct pose
    gripper.transform(transform_matrix)
    
    return gripper

def create_grasp_visualization_ply():
    """Create a PLY file with visible representations of the grasps."""
    print("Loading point clouds...")
    
    # Load the point clouds
    item_cloud_path = "item_cloud.ply"
    env_cloud_path = "env_cloud.ply"
    
    if not os.path.exists(item_cloud_path) or not os.path.exists(env_cloud_path):
        print(f"Error: Point cloud files not found: {item_cloud_path} or {env_cloud_path}")
        return False
    
    item_cloud = o3d.io.read_point_cloud(item_cloud_path)
    env_cloud = o3d.io.read_point_cloud(env_cloud_path)
    
    print("Getting grasp data...")
    
    # Import the graspnet interface to get grasp data
    try:
        from graspnet_interface import predict_full_grasp
        
        # Get grasp data
        tf_matrices, widths, scores = predict_full_grasp(
            item_cloud,
            env_cloud,
            rotation_resolution=32,
            top_n=100,
            n_best=5,  # Increased to get more grasps
            timeout=120
        )
        
        if len(scores) == 0:
            print("No grasps found. Please run test_grasp_detection() first.")
            return False
        
        print(f"Found {len(scores)} grasps")
        
        # Get score range for coloring
        score_min = min(scores)
        score_max = max(scores)
        
        # Create a new point cloud for the scene
        scene_cloud = copy.deepcopy(env_cloud)
        scene_cloud += item_cloud
        
        # Create visible 3D gripper models for each grasp
        gripper_meshes = []
        for i, (transform, width, score) in enumerate(zip(tf_matrices, widths, scores)):
            print(f"Creating gripper model for grasp {i+1} (score: {score}, width: {width})")
            
            # Create a gripper mesh
            gripper = create_gripper_mesh(transform, width, score, score_min, score_max)
            gripper_meshes.append(gripper)
        
        # Save scene with separate gripper models
        output_path = "grasp_visualized_grippers.ply"
        
        # Convert gripper meshes to point clouds for better visibility
        gripper_points = o3d.geometry.PointCloud()
        for gripper_mesh in gripper_meshes:
            # Sample points from the mesh surface
            gripper_pc = gripper_mesh.sample_points_uniformly(number_of_points=500)
            gripper_points += gripper_pc
        
        # Make gripper points larger for better visibility
        # (This doesn't affect the PLY file, but shows the intent)
        print("Adding gripper visualizations to the scene")
        
        # Combine with scene
        visualization_cloud = copy.deepcopy(scene_cloud)
        visualization_cloud += gripper_points
        
        # Save the combined visualization
        o3d.io.write_point_cloud(output_path, visualization_cloud)
        print(f"Saved visualization to {output_path}")
        
        # Also save just the grippers for clarity
        grippers_only_path = "grasp_grippers_only.ply"
        o3d.io.write_point_cloud(grippers_only_path, gripper_points)
        print(f"Saved grippers-only visualization to {grippers_only_path}")
        
        return True
    
    except Exception as e:
        print(f"Error creating grasp visualization: {e}")
        return False

def main():
    """Main function to create visualization."""
    success = create_grasp_visualization_ply()
    
    if success:
        # Display instructions for viewing
        print("\nVisualization created successfully!")
        print("To view the visualizations:")
        print("1. Use 'python -m open3d.visualization.open3d_visualizer grasp_visualized_grippers.ply'")
        print("2. Or use MeshLab/CloudCompare/Rerun to view the PLY files")
        print("\nThe grasp_grippers_only.ply file shows just the grippers without the scene")
        
        # Try to display with Open3D if available
        try:
            print("\nAttempting to show visualization...")
            import open3d.visualization
            
            # Load and display
            vis_cloud = o3d.io.read_point_cloud("grasp_visualized_grippers.ply")
            o3d.visualization.draw_geometries([vis_cloud])
        except Exception as e:
            print(f"Could not display visualization: {e}")
    else:
        print("Failed to create visualization. See error messages above.")

if __name__ == "__main__":
    main()
