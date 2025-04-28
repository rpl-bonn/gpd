#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publish point cloud to GPD for grasp detection and visualize the results.
Compatible with Python 2.x (tested on 2.7).
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
from std_msgs.msg import Header
from gpd_ros.msg import GraspConfigList

# ---------- Configuration ----------------------------------------------------
ENV_PCD_FILE_PATH    = "/workspace/env_cloud.pcd"   # Environment scene cloud 
ITEM_PCD_FILE_PATH   = "/workspace/item_cloud.pcd"  # Segmented object cloud
OUTPUT_JSON_PATH     = "/workspace/detected_grasps.json"
CLOUD_TOPIC          = "/cloud_stitched"            # Topic that GPD subscribes to
GRASP_TOPIC          = "/detect_grasps/clustered_grasps"
COMBINED_CLOUD_TOPIC = "/cloud_with_grasps"         # Colored grasp visualization
FRAME_ID             = "base_link"
NODE_NAME            = "cloud_publisher"
NUM_GRASPS_TO_VIS    = 3
# -----------------------------------------------------------------------------

# -------- Globals ------------------------------------------------------------
grasps_saved = False
top_grasps   = []
grasps_received_event = threading.Event()
# -----------------------------------------------------------------------------

def load_pcd(filepath):
    """Loads a PCD file using python-pcl."""
    rospy.loginfo("Loading PCD file: %s" % filepath)
    try:
        cloud = pcl.load(filepath)
        rospy.loginfo("Loaded %d points." % cloud.size)
        return cloud
    except Exception as e:
        rospy.logerr("Failed to load %s: %s" % (filepath, e))
        return None

def create_point_cloud2_message(pcl_cloud, frame_id):
    """Create a ROS PointCloud2 message from a PCL pointcloud."""
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

def grasp_callback(msg):
    """Callback to process grasp detection results."""
    global grasps_saved, top_grasps, grasps_received_event
    
    if grasps_received_event.is_set():
        return  # Already processed
    
    if not msg.grasps:
        rospy.logwarn("Received empty grasp list.")
        grasps_received_event.set()
        return

    # Sort grasps by score
    sorted_grasps = sorted(msg.grasps, key=lambda g: g.score.data, reverse=True)
    top_grasps = sorted_grasps[:NUM_GRASPS_TO_VIS]
    rospy.loginfo("Stored top %d grasps for visualization." % len(top_grasps))

    # Save all grasps to JSON 
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

    # Signal that we've received and processed grasps
    grasps_received_event.set()

def create_grasp_visualization_points(grasps):
    """Generates XYZ points representing the grasp structure (fingertips and approach)."""
    grasp_points = []
    arm_length = 0.05 # Visualize 5cm of the approach vector

    for grasp in grasps:
        pos = np.array([grasp.position.x, grasp.position.y, grasp.position.z])
        axis = np.array([grasp.axis.x, grasp.axis.y, grasp.axis.z])
        approach = np.array([grasp.approach.x, grasp.approach.y, grasp.approach.z])
        width = grasp.width.data

        # Normalize vectors
        axis_norm = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else axis
        approach_norm = approach / np.linalg.norm(approach) if np.linalg.norm(approach) > 1e-6 else approach

        # Calculate points for the two fingertips
        half_width_vec = axis_norm * (width / 2.0)
        p1 = pos + half_width_vec
        p2 = pos - half_width_vec

        # Calculate point representing the base along the approach vector
        p_base = pos - approach_norm * arm_length

        grasp_points.append(list(p1)) # Fingertip 1
        grasp_points.append(list(p2)) # Fingertip 2
        grasp_points.append(list(pos)) # Grasp center
        grasp_points.append(list(p_base)) # Point along approach vector

    rospy.loginfo("Generated %d points for grasp visualization." % len(grasp_points))
    return grasp_points

def pack_rgb(r, g, b):
    """Packs RGB bytes into a single uint32 suitable for PCL/ROS."""
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    return rgb_int

def create_colored_merged_cloud(original_pcl_cloud, grasp_points_xyz):
    """Creates a new PCL PointCloud XYZRGB merging original and grasp points."""
    merged_cloud = pcl.PointCloud_PointXYZRGB()

    # Define colors
    color_object = pack_rgb(255, 255, 255) # White
    color_grasp = pack_rgb(255, 0, 0)     # Red

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

    # Add grasp points
    for p in grasp_points_xyz:
        merged_points_list.append([p[0], p[1], p[2], color_grasp])

    # Load data into the PCL cloud object
    merged_cloud.from_list(merged_points_list)
    return merged_cloud

def main():
    """Main function to publish cloud and visualize grasps."""
    # Parse args: optional overrides
    item_path = ITEM_PCD_FILE_PATH
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            item_path = sys.argv[1]
        else:
            rospy.logwarn("Provided path %s not found. Using default: %s" % (sys.argv[1], item_path))

    if not os.path.exists(item_path):
        rospy.logerr("PCD file not found: %s" % item_path)
        return

    rospy.init_node(NODE_NAME, anonymous=True)
    
    # Reset state
    global grasps_received_event, grasps_saved, top_grasps
    grasps_received_event.clear()
    grasps_saved = False
    top_grasps = []

    # Load item point cloud
    item_cloud = load_pcd(item_path)
    if not item_cloud:
        return

    # Create ROS message from point cloud
    cloud_msg = create_point_cloud2_message(item_cloud, FRAME_ID)
    
    # Set up publishers and subscribers
    cloud_pub = rospy.Publisher(CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)
    combined_pub = rospy.Publisher(COMBINED_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)
    grasp_sub = rospy.Subscriber(GRASP_TOPIC, GraspConfigList, grasp_callback)
    
    # Allow time for connections to establish
    rospy.sleep(1.0)
    
    # Publish point cloud to trigger grasp detection
    cloud_pub.publish(cloud_msg)
    rospy.loginfo("Published point cloud to %s" % CLOUD_TOPIC)
    
    # Wait for grasp detection with a timeout
    timeout_duration = 30.0
    rospy.loginfo("Waiting for grasp detection... (Timeout: %.1fs)" % timeout_duration)
    event_triggered = grasps_received_event.wait(timeout=timeout_duration)
    
    if event_triggered and top_grasps:
        rospy.loginfo("Grasps received. Generating visualization...")
        # Generate visualization
        grasp_viz_points = create_grasp_visualization_points(top_grasps)
        merged_colored_cloud = create_colored_merged_cloud(item_cloud, grasp_viz_points)
        merged_cloud_msg = create_point_cloud2_message(merged_colored_cloud, FRAME_ID)
        
        # Publish the visualization
        combined_pub.publish(merged_cloud_msg)
        rospy.loginfo("Published visualization to %s" % COMBINED_CLOUD_TOPIC)
        
        # Save the merged cloud for later viewing
        try:
            output_ply_path = "/workspace/merged_cloud_for_rerun.ply"
            pcl.save(merged_colored_cloud, output_ply_path, format="ply", binary=False)
            rospy.loginfo("Saved merged cloud to %s" % output_ply_path)
        except Exception as e:
            rospy.logerr("Failed to save merged cloud: %s" % e)
    elif event_triggered:
        rospy.logwarn("Grasp callback finished, but no grasps were found.")
    else:
        rospy.logwarn("Timeout waiting for grasps. No visualization published.")
    
    # Allow time for final messages to be sent
    rospy.loginfo("Processing complete. Shutting down in 3 seconds...")
    rospy.sleep(3.0)
    rospy.signal_shutdown("Finished processing")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Unexpected error: %s" % e)
