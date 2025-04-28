#!/usr/bin/env python3
"""
Interactive visualization of grasp poses with Open3D.
This script loads the point clouds and grasp poses and displays them in an interactive viewer.
"""

import open3d as o3d
import numpy as np
import os
import sys
import copy

def create_coordinate_frame(transform_matrix, size=0.05):
    """Create a coordinate frame at the given pose with the given size."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(transform_matrix)
    return frame

def create_gripper_model(transform_matrix, width, color=[0, 1, 0]):
    """Create a simple gripper model at the given pose with the given width and color."""
    # Create base of the gripper
    base_height = 0.02
    base_width = 0.04
    base = o3d.geometry.TriangleMesh.create_box(width=base_width, height=base_height, depth=base_width)
    
    # Move base to origin
    base.translate([-base_width/2, -base_height/2, -base_width/2])
    
    # Create fingers
    finger_width = 0.01
    finger_height = 0.04
    finger_depth = 0.02
    
    left_finger = o3d.geometry.TriangleMesh.create_box(width=finger_width, height=finger_height, depth=finger_depth)
    right_finger = o3d.geometry.TriangleMesh.create_box(width=finger_width, height=finger_height, depth=finger_depth)
    
    # Position the fingers
    left_finger.translate([-width/2 - finger_width/2, 0, -finger_depth/2])
    right_finger.translate([width/2 - finger_width/2, 0, -finger_depth/2])
    
    # Combine into one mesh
    gripper = base
    gripper += left_finger
    gripper += right_finger
    
    # Paint the gripper
    gripper.paint_uniform_color(color)
    
    # Transform the gripper to the grasp pose
    gripper.transform(transform_matrix)
    
    return gripper

def visualize_grasps():
    """Load point clouds and grasp poses and visualize them."""
    print("Loading point clouds and grasp poses...")
    
    # Load point clouds
    item_cloud_path = "item_cloud.ply"
    env_cloud_path = "env_cloud.ply"
    
    if not os.path.exists(item_cloud_path) or not os.path.exists(env_cloud_path):
        print(f"Error: Point cloud files not found: {item_cloud_path} or {env_cloud_path}")
        return
    
    item_cloud = o3d.io.read_point_cloud(item_cloud_path)
    env_cloud = o3d.io.read_point_cloud(env_cloud_path)
    
    # Load grasp visualization file
    grasp_vis_path = "grasp_visualization.ply"
    grasp_frames_path = "grasp_visualization_grasp_frames.ply"
    
    # Check if grasp visualization exists
    if not os.path.exists(grasp_frames_path):
        print(f"Error: Grasp visualization not found at {grasp_frames_path}")
        print("Please run the test_grasp_detection() function first.")
        return
    
    # Load grasp frames
    grasp_frames = o3d.io.read_point_cloud(grasp_frames_path)
    
    # Create visualization geometries
    vis_geometries = []
    
    # Add point clouds
    # Make environment cloud gray
    env_cloud_colored = copy.deepcopy(env_cloud)
    if not env_cloud_colored.has_colors():
        env_cloud_colored.paint_uniform_color([0.8, 0.8, 0.8])
    vis_geometries.append(env_cloud_colored)
    
    # Make item cloud blue
    item_cloud_colored = copy.deepcopy(item_cloud)
    if not item_cloud_colored.has_colors():
        item_cloud_colored.paint_uniform_color([0.0, 0.0, 1.0])
    vis_geometries.append(item_cloud_colored)
    
    # Load grasp data from the server response
    # This is a simplification - in a real application, you'd load the actual grasp data
    # from the server or from the saved files. We'll generate some test data here.
    
    try:
        print("Getting transforms from grasp detection results...")
        # Try to load from the test script output
        from graspnet_interface import test_grasp_detection, predict_full_grasp
        
        # Get the transformation matrices from the test function
        tf_matrices, widths, scores = predict_full_grasp(
            item_cloud,
            env_cloud,
            rotation_resolution=32,
            top_n=100,
            n_best=3,
            timeout=120
        )
        
        print(f"Found {len(scores)} grasps")
        
        # Create a coordinate frame for each grasp
        for i, (transform, width, score) in enumerate(zip(tf_matrices, widths, scores)):
            print(f"Adding grasp {i+1} with score {score}")
            
            # Create coordinate frame
            frame = create_coordinate_frame(transform, size=0.05)
            vis_geometries.append(frame)
            
            # Create gripper model
            # Map score to a color (red for high score, blue for low score)
            if len(scores) > 1:
                score_min = min(scores)
                score_max = max(scores)
                score_range = score_max - score_min if score_max > score_min else 1.0
                norm_score = (score - score_min) / score_range
                color = [norm_score, 0.5*(1-norm_score), 1-norm_score]  # red to blue
            else:
                color = [1, 0, 0]  # Red for single grasp
            
            gripper = create_gripper_model(transform, width, color)
            vis_geometries.append(gripper)
    
    except Exception as e:
        print(f"Warning: Could not load grasp data: {e}")
        print("Displaying only point clouds and grasp centers.")
    
        # Just add the grasp center points
        if grasp_frames.has_points():
            vis_geometries.append(grasp_frames)
    
    # Set up the visualizer
    print("Starting visualizer...")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Grasp Visualization")
    
    # Add geometries to the visualizer
    for geom in vis_geometries:
        vis.add_geometry(geom)
    
    # Set up the camera position
    ctr = vis.get_view_control()
    ctr.set_front([-1, -1, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    # Add keyboard instructions
    print("\nKeyboard controls:")
    print("  Q/Esc: Quit the visualizer")
    print("  H: Show help")
    print("  R: Reset camera view")
    print("  ,/.: Decrease/Increase size of points")
    print("  Mouse: Rotate/Pan/Zoom the view")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_grasps()
