#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge environment and item point clouds, publish CloudIndexed to gpd_ros,
and visualize detected grasps by coloring the merged cloud.
Compatible with Python 2.x (tested on 2.7). No f-strings or type hints.
"""

from __future__ import division, print_function

import rospy
import pcl
import struct
import json
import os
import sys
import threading
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Int64
from geometry_msgs.msg import Point
from gpd_ros.msg import CloudIndexed, CloudSources, GraspConfigList

# ---------- Configuration ----------------------------------------------------
ENV_PCD_FILE_PATH    = "/workspace/env_cloud.pcd"   # Environment scene cloud
ITEM_PCD_FILE_PATH   = "/workspace/item_cloud.pcd"  # Segmented object cloud
OUTPUT_JSON_PATH     = "/workspace/detected_grasps.json"
ENV_CLOUD_TOPIC      = "/cloud_stitched"            # Match working script's topic
CLOUD_INDEXED_TOPIC  = "/cloud_stitched"           # Match working script's topic
GRASP_TOPIC          = "/detect_grasps/clustered_grasps"
COMBINED_CLOUD_TOPIC = "/cloud_with_grasps"         # Colored grasp visualization
FRAME_ID             = "base_link"
NODE_NAME            = "cloud_indexed_publisher"
NUM_GRASPS_TO_VIS    = 3
# Tolerance for KD-tree matching (1 mm default)
MATCH_EPS            = rospy.get_param("~match_eps", 1e-3)
# -----------------------------------------------------------------------------

# -------- Globals ------------------------------------------------------------
grasps_saved = False
top_grasps   = []
grasps_received_event = threading.Event()
# -----------------------------------------------------------------------------

# ---------------------------- Utility functions ------------------------------
def load_pcd(filepath):
    rospy.loginfo("Loading PCD file: %s" % filepath)
    try:
        cloud = pcl.load(filepath)
        rospy.loginfo("Loaded %d points." % cloud.size)
        return cloud
    except Exception as e:
        rospy.logerr("Failed to load %s: %s" % (filepath, e))
        return None


def create_point_cloud2_message(pcl_cloud, frame_id):
    """
    Create a ROS PointCloud2 message from a PCL pointcloud.
    Handles both XYZ and XYZRGB cloud types.
    """
    header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    
    # Check if this is an XYZRGB cloud by checking the first point format
    has_rgb = False
    points = pcl_cloud.to_list()
    if points and len(points[0]) >= 4:  # XYZRGB format
        fields.append(PointField('rgb', 12, PointField.UINT32, 1))
        has_rgb = True
        point_step = 16  # 4 fields * 4 bytes
    else:  # XYZ format
        point_step = 12  # 3 fields * 4 bytes
    
    # Pack the data
    data = bytearray()
    for p in points:
        if has_rgb:
            # Ensure we have 4 values even if input provides only 3
            if len(p) == 3:
                x, y, z = p
                rgb = 0xFFFFFF  # Default to white if no color
            else:
                x, y, z, rgb = p[:4]  # Take first 4 values
            data.extend(struct.pack('<fffI', x, y, z, int(rgb)))
        else:
            # XYZ only
            x, y, z = p[:3]  # Take first 3 values
            data.extend(struct.pack('<fff', x, y, z))
    
    return PointCloud2(
        header=header,
        height=1,
        width=pcl_cloud.size,
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=point_step,
        row_step=point_step * pcl_cloud.size,
        data=bytes(data)
    )


def merge_clouds(env_cloud, item_cloud):
    """
    Concatenate two clouds into one merged cloud.
    Handles XYZ and XYZRGB point cloud types.
    """
    # First, convert the clouds to list format for easier handling
    env_points = env_cloud.to_list()
    item_points = item_cloud.to_list()
    
    # Determine point format by checking the first point
    env_has_rgb = len(env_points[0]) >= 4 if env_points else False
    item_has_rgb = len(item_points[0]) >= 4 if item_points else False
    
    # If either cloud has RGB, ensure all points have RGB values
    if env_has_rgb or item_has_rgb:
        # Add default RGB (white) to XYZ points
        if not env_has_rgb:
            env_points = [(p[0], p[1], p[2], 0xFFFFFF) for p in env_points]
        if not item_has_rgb:
            item_points = [(p[0], p[1], p[2], 0xFFFFFF) for p in item_points]
        
        # Create new XYZRGB cloud
        merged = pcl.PointCloud_PointXYZRGB()
    else:
        # Both are XYZ only, so create XYZ cloud
        merged = pcl.PointCloud()
    
    # Combine the point lists and convert to the merged cloud
    merged_points = env_points + item_points
    merged.from_list(merged_points)
    
    rospy.loginfo("Merged cloud contains %d points." % merged.size)
    return merged


def compute_indices(env_np, item_np, eps=MATCH_EPS):
    """
    Find indices of item points within the merged environment cloud via cKDTree.
    Handles both XYZ and XYZRGB arrays by only using the XYZ components.
    """
    # Ensure we only use the XYZ components for matching
    env_xyz = env_np[:, :3]
    item_xyz = item_np[:, :3]
    
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(env_xyz)
        idxs = []
        for p in item_xyz:
            neighbors = tree.query_ball_point(p, r=eps)
            if neighbors:
                idxs.append(neighbors[0])
        unique_idxs = sorted(set(idxs))
        rospy.loginfo("Matched %d/%d item points within %.4f m." % (
            len(unique_idxs), item_np.shape[0], eps))
        return unique_idxs
    except ImportError:
        rospy.logwarn("SciPy not found; using brute-force matcher.")
        idxs = []
        for i_env, p_env in enumerate(env_xyz):
            if np.any(np.linalg.norm(item_xyz - p_env, axis=1) < eps):
                idxs.append(i_env)
        return sorted(set(idxs))


def align_item_to_env(env_cloud, item_cloud, max_iter=25, corr_dist=0.05):
    """
    Simple pass-through function that returns the item cloud directly.
    The original ICP alignment was causing compatibility issues.
    """
    rospy.loginfo("Using direct cloud merging without ICP alignment.")
    return item_cloud
# -----------------------------------------------------------------------------

# ----------------------------- ROS callbacks ---------------------------------
def grasp_callback(msg):
    global grasps_saved, top_grasps
    if grasps_received_event.is_set():
        return
    if not msg.grasps:
        rospy.logwarn("Received empty grasp list.")
        grasps_received_event.set()
        return

    sorted_grasps = sorted(msg.grasps, key=lambda g: g.score.data, reverse=True)
    top_grasps = sorted_grasps[:NUM_GRASPS_TO_VIS]
    rospy.loginfo("Stored top %d grasps." % len(top_grasps))

    if not grasps_saved:
        all_data = [{
            'score': g.score.data,
            'position': {'x': g.position.x, 'y': g.position.y, 'z': g.position.z},
            'approach': {'x': g.approach.x, 'y': g.approach.y, 'z': g.approach.z},
            'axis': {'x': g.axis.x, 'y': g.axis.y, 'z': g.axis.z},
            'binormal': {'x': g.binormal.x, 'y': g.binormal.y, 'z': g.binormal.z},
            'width': g.width.data
        } for g in msg.grasps]
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(all_data, f, indent=4)
        rospy.loginfo("Saved all grasps to %s" % OUTPUT_JSON_PATH)
        grasps_saved = True

    grasps_received_event.set()
# -----------------------------------------------------------------------------

# --------------------------------- main --------------------------------------
def main():
    # Parse args: optional overrides
    env_path = ENV_PCD_FILE_PATH
    item_path = ITEM_PCD_FILE_PATH
    if len(sys.argv) >= 3:
        env_path, item_path = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        env_path = sys.argv[1]

    for path in (env_path, item_path):
        if not os.path.exists(path):
            rospy.logerr("PCD file not found: %s" % path)
            return

    rospy.init_node(NODE_NAME, anonymous=True)
    
    # Clear any previous state
    global grasps_received_event, grasps_saved, top_grasps
    grasps_received_event.clear()
    grasps_saved = False
    top_grasps = []

    env_cloud = load_pcd(env_path)
    item_cloud = load_pcd(item_path)
    if not env_cloud or not item_cloud:
        return

    # Simple merge without ICP alignment
    merged_cloud = merge_clouds(env_cloud, item_cloud)

    # Compute indices of item points within merged cloud
    env_np = np.array(merged_cloud.to_list())
    item_np = np.array(item_cloud.to_list())
    
    # Ensure arrays have consistent dimensions for comparison
    if env_np.shape[1] != item_np.shape[1]:
        # Get minimum dimension
        min_dim = min(env_np.shape[1], item_np.shape[1])
        # Use only the common dimensions (at least XYZ should be common)
        env_np = env_np[:, :min_dim]
        item_np = item_np[:, :min_dim]
    
    indices = compute_indices(env_np, item_np)
    if not indices:
        rospy.logerr("No matching indices found; aborting.")
        return

    # Create a PointCloud2 message from the item cloud instead of merged cloud
    # This follows the working script which only publishes the item cloud
    item_msg = create_point_cloud2_message(item_cloud, FRAME_ID)
    
    # Publishers - only need two publishers now
    cloud_pub = rospy.Publisher(ENV_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)
    combined_pub = rospy.Publisher(COMBINED_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)
    
    # Subscriber for the grasps (add subscription before publishing)
    grasp_sub = rospy.Subscriber(GRASP_TOPIC, GraspConfigList, grasp_callback)
    
    # Allow time for connections
    rospy.sleep(1.0)
    
    # Publish the point cloud (using the same approach as the working script)
    cloud_pub.publish(item_msg)
    rospy.loginfo("Published point cloud to %s" % ENV_CLOUD_TOPIC)

    # Wait for grasp detection with a timeout
    timeout_duration = 45.0  # Increased timeout
    rospy.loginfo("Waiting for grasp detection... (Timeout: %.1fs)" % timeout_duration)
    event_triggered = grasps_received_event.wait(timeout=timeout_duration)

    if event_triggered and top_grasps:
        rospy.loginfo("Grasps received. Generating visualization...")
        # Generate visualization
        grasp_viz_points, grasp_viz_colors = create_grasp_visualization_points(top_grasps)
        merged_colored_cloud = create_colored_merged_cloud(merged_cloud, grasp_viz_points, grasp_viz_colors)
        merged_cloud_msg = create_point_cloud2_message(merged_colored_cloud, FRAME_ID)
        
        # Publish the visualization
        combined_pub.publish(merged_cloud_msg)
        rospy.loginfo("Published combined cloud with grasp visualization to %s" % COMBINED_CLOUD_TOPIC)
        
        # Save the merged cloud for visualization
        try:
            output_ply_path = "/workspace/merged_cloud_for_rerun.ply"
            pcl.save(merged_colored_cloud, output_ply_path, format="ply", binary=False)
            rospy.loginfo("Saved merged cloud with grasp visualization to %s" % output_ply_path)
        except Exception as e:
            rospy.logerr("Failed to save merged cloud to %s: %s" % (output_ply_path, e))
    elif event_triggered:
        rospy.logwarn("Grasp callback finished, but no grasps were found or stored.")
    else:
        rospy.logwarn("Timeout waiting for grasps. No grasp visualization published.")
        rospy.logwarn("Check if the grasp detection node is properly processing the cloud data.")
        rospy.logwarn("You might need to restart the UR5 launch file or check ROS network configuration.")
    
    # Keep node alive briefly to ensure messages are sent, then shutdown
    rospy.loginfo("Processing complete. Shutting down in 3 seconds...")
    rospy.sleep(3.0)
    rospy.signal_shutdown("Finished processing and visualization.")

# --- Add visualization functions ---
def create_grasp_visualization_points(grasps):
    """
    Generates points representing the full 3D gripper structure for each grasp.
    Creates a dense point-based visualization of the gripper with:
    - Left and right fingers (as dense point clouds)
    - Hand base/palm
    - Approach vector
    - Score-based coloring
    """
    # Define gripper geometry parameters (in meters)
    finger_width = 0.01       # Width of each finger
    hand_depth = 0.06         # Depth of hand/fingers
    hand_height = 0.02        # Height of the hand
    base_depth = 0.02         # Depth of hand base
    approach_depth = 0.07     # Length of approach indicator
    
    grasp_points = []
    grasp_colors = []  # Store colors for each point
    
    # Number of points to generate for each component (higher = more detailed visualization)
    points_per_finger = 30    # Points per finger surface
    points_per_base = 30      # Points for the base
    points_per_approach = 15  # Points for the approach vector
    
    # Get min and max scores for color scaling
    scores = [grasp.score.data for grasp in grasps]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1
    score_range = max_score - min_score if max_score > min_score else 1.0
    
    for grasp_idx, grasp in enumerate(grasps):
        # Extract grasp parameters
        pos = np.array([grasp.position.x, grasp.position.y, grasp.position.z])
        approach = np.array([grasp.approach.x, grasp.approach.y, grasp.approach.z])
        binormal = np.array([grasp.binormal.x, grasp.binormal.y, grasp.binormal.z])
        axis = np.array([grasp.axis.x, grasp.axis.y, grasp.axis.z])
        width = grasp.width.data
        score = grasp.score.data
        
        # Color based on score (from blue=low to red=high)
        norm_score = (score - min_score) / score_range
        r = int(255 * norm_score)
        g = 0
        b = int(255 * (1 - norm_score))
        
        # Base colors for different parts of the gripper
        color_fingers = pack_rgb(r, g, b)           # Score-based color for fingers
        color_base = pack_rgb(max(0, r-50), g, min(255, b+50))  # Slightly modified for base
        color_approach = pack_rgb(255, 50, 50)      # Red for approach vector
        
        # Ensure vectors are normalized
        approach_norm = approach / np.linalg.norm(approach) if np.linalg.norm(approach) > 1e-6 else approach
        binormal_norm = binormal / np.linalg.norm(binormal) if np.linalg.norm(binormal) > 1e-6 else binormal
        axis_norm = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else axis
        
        # Calculate the half width for finger placement
        hw = 0.5 * width - 0.5 * finger_width
        
        # Calculate key positions
        left_bottom = pos - hw * binormal_norm
        right_bottom = pos + hw * binormal_norm
        left_top = left_bottom + hand_depth * approach_norm
        right_top = right_bottom + hand_depth * approach_norm
        
        # Create rotation matrix for the hand frame
        rot_matrix = np.column_stack((approach_norm, binormal_norm, axis_norm))
        
        # Generate dense points for the LEFT finger
        for i in range(points_per_finger):
            # Parametric surface coordinates (u,v,w)
            u = np.random.random()      # Along length (approach)
            v = np.random.random()      # Along width (binormal)
            w = np.random.random()      # Along height (axis)
            
            # Calculate position in the finger's local frame
            local_pos = np.array([
                u * hand_depth, 
                (v - 0.5) * finger_width,
                (w - 0.5) * hand_height
            ])
            
            # Transform to global coordinates
            global_pos = left_bottom + rot_matrix.dot(local_pos)
            grasp_points.append(list(global_pos))
            grasp_colors.append(color_fingers)
        
        # Generate dense points for the RIGHT finger
        for i in range(points_per_finger):
            u = np.random.random()
            v = np.random.random()
            w = np.random.random()
            
            local_pos = np.array([
                u * hand_depth, 
                (v - 0.5) * finger_width,
                (w - 0.5) * hand_height
            ])
            
            global_pos = right_bottom + rot_matrix.dot(local_pos)
            grasp_points.append(list(global_pos))
            grasp_colors.append(color_fingers)
        
        # Generate points for the BASE connecting the fingers
        base_center = left_bottom + 0.5 * (right_bottom - left_bottom) - 0.01 * approach_norm
        for i in range(points_per_base):
            u = np.random.random() * base_depth
            v = np.random.random() * width
            w = np.random.random() * hand_height
            
            local_pos = np.array([
                -u,  # Negative to go backward from the grasp center
                v - 0.5 * width,
                w - 0.5 * hand_height
            ])
            
            global_pos = base_center + rot_matrix.dot(local_pos)
            grasp_points.append(list(global_pos))
            grasp_colors.append(color_base)
        
        # Generate points for the APPROACH vector/arrow
        approach_center = base_center - 0.04 * approach_norm
        for i in range(points_per_approach):
            # For approach vector, create a thinner, tapered shape
            u = np.random.random() * approach_depth
            # Make the approach vector taper toward the end
            taper_factor = 1.0 - 0.8 * (u / approach_depth)
            v = (np.random.random() - 0.5) * finger_width * taper_factor
            w = (np.random.random() - 0.5) * 0.5 * hand_height * taper_factor
            
            local_pos = np.array([-u, v, w])  # Negative to go backward from center
            global_pos = approach_center + rot_matrix.dot(local_pos)
            
            grasp_points.append(list(global_pos))
            grasp_colors.append(color_approach)
        
        # Add extra points at key positions for better visibility
        # Add grasp center point
        grasp_points.append(list(pos))
        grasp_colors.append(pack_rgb(255, 255, 0))  # Yellow
        
        # Add fingertips
        grasp_points.append(list(left_top))
        grasp_colors.append(pack_rgb(255, 255, 255))  # White
        grasp_points.append(list(right_top))
        grasp_colors.append(pack_rgb(255, 255, 255))  # White

    rospy.loginfo("Generated %d points for detailed grasp visualization." % len(grasp_points))
    return grasp_points, grasp_colors

def pack_rgb(r, g, b):
    """Packs RGB bytes into a single uint32 suitable for PCL/ROS."""
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    return rgb_int

def create_colored_merged_cloud(original_pcl_cloud, grasp_points_xyz, grasp_colors=None):
    """Creates a new PCL PointCloud XYZRGB merging original and grasp points with colors."""
    merged_cloud = pcl.PointCloud_PointXYZRGB() # Create cloud with RGB field

    # Define colors (0-255)
    color_object = pack_rgb(255, 255, 255) # White
    color_default_grasp = pack_rgb(255, 0, 0)  # Red (default)

    # Add original points
    original_points = original_pcl_cloud.to_list()
    merged_points_list = []
    # Check if original cloud already has color
    has_rgb = len(original_points[0]) >= 4 if original_points else False
    
    for p in original_points:
        if has_rgb:
            # Keep existing color if available
            merged_points_list.append(p[:4])
        else:
            # Add white color
            merged_points_list.append([p[0], p[1], p[2], color_object])

    # Add grasp points with appropriate colors
    if grasp_colors and len(grasp_colors) == len(grasp_points_xyz):
        # Use provided colors 
        for p, color in zip(grasp_points_xyz, grasp_colors):
            merged_points_list.append([p[0], p[1], p[2], color])
    else:
        # Fall back to default color if no colors provided
        for p in grasp_points_xyz:
            merged_points_list.append([p[0], p[1], p[2], color_default_grasp])

    # Load data into the PCL cloud object
    merged_cloud.from_list(merged_points_list)
    rospy.loginfo("Created merged cloud with %d points (%d original + %d grasp)." % 
                 (merged_cloud.size, len(original_points), len(grasp_points_xyz)))
    return merged_cloud
# --- End visualization functions ---

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Unexpected error: %s" % e)
