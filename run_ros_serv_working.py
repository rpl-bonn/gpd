#!/usr/bin/env python

import rospy
import pcl
import struct
import ctypes
import json
import os
import sys
import threading # Added
import numpy as np # Added

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from gpd_ros.msg import GraspConfigList

# --- Configuration ---
PCD_FILE_PATH = "/workspace/item_cloud.pcd"  # Default PCD file path
OUTPUT_JSON_PATH = "/workspace/detected_grasps.json" # Where to save grasps
CLOUD_TOPIC = "/cloud_stitched"
GRASP_TOPIC = "/detect_grasps/clustered_grasps" # Adjust if your topic is different
COMBINED_CLOUD_TOPIC = "/cloud_with_grasps" # New topic for visualization
FRAME_ID = "base_link" # Ohor your relevant TF frame
NODE_NAME = "pcd_publisher_grasp_visualizer" # Updated node name
NUM_GRASPS_TO_VISUALIZE = 3 # How many top grasps to show
# --- End Configuration ---

# --- Globals ---
grasps_saved = False
top_grasps = []
grasps_received_event = threading.Event()
# --- End Globals ---

# def load_pcd(filepath):
#     """Loads a PCD file using python-pcl."""
#     rospy.loginfo("Loading PCD file: {}".format(filepath))
#     try:
#         cloud = pcl.load(filepath)
#         rospy.loginfo("Loaded cloud with {} points.".format(cloud.size))
#         return cloud
#     except Exception as e:
#         rospy.logerr("Failed to load PCD file {}: {}".format(filepath, e))
#         return None

# filepath: /home/user/azirar/docker_containers/grasp_pose_detection/gpd/run_ros_serv.py
def load_pcd(filepath):
    """Loads a PCD/PLY file using python-pcl."""
    rospy.loginfo("Loading point cloud file: {}".format(filepath))
    try:
        cloud = pcl.load(filepath)
        rospy.loginfo("Loaded cloud with {} points.".format(cloud.size))
        # --- Add this ---
        if cloud.size > 0:
            try:
                points_list = cloud.to_list()
                rospy.loginfo("First point data structure: {}".format(points_list[0]))
            except Exception as e:
                rospy.logwarn("Could not inspect first point: {}".format(e))
        # --- End Add ---
        return cloud
    except Exception as e:
        rospy.logerr("Failed to load point cloud file {}: {}".format(filepath, e))
        return None
    
def create_point_cloud2_message(pcl_cloud, frame_id):
    """Converts a PCL cloud object to a sensor_msgs/PointCloud2 message."""
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Assuming basic XYZ structure for simplicity
    # You might need to adapt this if your PCD has different fields (e.g., RGB)
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12 # 3 fields * 4 bytes/float32

    # Convert PCL data to a byte array
    points_list = pcl_cloud.to_list()
    data = bytearray()
    for point in points_list:
        # Pack x, y, z as float32
        # Ensure your PCD actually contains these fields in this order
        try:
            data.extend(struct.pack('fff', point[0], point[1], point[2]))
        except IndexError:
             rospy.logwarn_once("Point in PCD file does not seem to have XYZ data. Check PCD structure.")
             # Attempt packing with zeros if fields are missing (adjust as needed)
             xyz = list(point[:3]) + [0.0] * (3 - len(point))
             data.extend(struct.pack('fff', *xyz))


    cloud_msg = PointCloud2(
        header=header,
        height=1,
        width=pcl_cloud.size,
        is_dense=False, # Set to True if no NaNs/Infs
        is_bigendian=False,
        fields=fields,
        point_step=point_step,
        row_step=point_step * pcl_cloud.size,
        data=bytes(data) # Convert bytearray to bytes
    )
    rospy.loginfo("Created PointCloud2 message.")
    return cloud_msg

def create_colored_point_cloud2_message(pcl_cloud_xyzrgb, frame_id):
    """Converts a PCL XYZRGB cloud object to a sensor_msgs/PointCloud2 message."""
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1), # Changed to UINT32 for packed RGB
    ]
    point_step = 16 # 4 fields * 4 bytes/field (x,y,z are float32, rgb is uint32)

    # Convert PCL data to a byte array
    points_list = pcl_cloud_xyzrgb.to_list() # Assuming cloud is XYZRGB
    data = bytearray()
    for point in points_list:
        # Pack x, y, z as float32
        x, y, z, rgb_packed = point[0], point[1], point[2], point[3]
        # Pack RGB as a single uint32
        # The PCL library often stores RGB packed into a float. We need to handle this.
        # Let's assume rgb_packed is already in the correct uint32 format or can be cast.
        # If it's a float, conversion is needed:
        # s = struct.pack('>f', rgb_packed)
        # i = struct.unpack('>l', s)[0]
        # packed_int = i
        # For simplicity, assume it's directly usable or castable for now.
        # A common PCL format packs RGB into the float's bytes.
        try:
             # Pack x, y, z (float32), rgb (uint32)
             rgb_int = 0
             if isinstance(rgb_packed, float):
                 # Interpret the float's bytes as an int
                 s = struct.pack('>f', rgb_packed)
                 rgb_int = struct.unpack('>I', s)[0] # Use >I for big-endian uint32
             elif isinstance(rgb_packed, (int, np.uint32)):
                 rgb_int = int(rgb_packed)
             else:
                 rospy.logwarn_once("Unexpected RGB data type: {}".format(type(rgb_packed)))

             data.extend(struct.pack('<fffI', x, y, z, rgb_int)) # Use < for little-endian
        except Exception as e:
             rospy.logwarn_once("Packing point failed: {}. Point: {}".format(e, point))
             # Pack with default values if error occurs
             data.extend(struct.pack('<fffI', 0.0, 0.0, 0.0, 0))


    cloud_msg = PointCloud2(
        header=header,
        height=1,
        width=len(points_list), # Use length of list
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=point_step,
        row_step=point_step * len(points_list),
        data=bytes(data)
    )
    rospy.loginfo("Created colored PointCloud2 message.")
    return cloud_msg

def grasp_callback(msg):
    """Callback function to receive, save, and store top grasp messages."""
    global grasps_saved, top_grasps, grasps_received_event
    if grasps_received_event.is_set(): # Avoid processing again if already done
        return

    rospy.loginfo("Received {} grasps.".format(len(msg.grasps)))

    if not msg.grasps:
        rospy.logwarn("Received an empty grasp list.")
        grasps_received_event.set() # Signal completion even if empty
        return

    # Sort grasps by score (descending)
    sorted_grasps = sorted(msg.grasps, key=lambda g: g.score.data, reverse=True)

    # Store top N grasps
    top_grasps = sorted_grasps[:NUM_GRASPS_TO_VISUALIZE]
    rospy.loginfo("Stored top {} grasps for visualization.".format(len(top_grasps)))


    # --- Save all grasps to JSON (original functionality) ---
    if not grasps_saved:
        grasps_data = []
        for grasp in msg.grasps: # Save all received grasps, not just top N
            grasps_data.append({
                'score': grasp.score.data,
                'position': {'x': grasp.position.x, 'y': grasp.position.y, 'z': grasp.position.z},
                'orientation': {'x': grasp.approach.x, 'y': grasp.approach.y, 'z': grasp.approach.z}, # Using approach for orientation visualization
                'approach': {'x': grasp.approach.x, 'y': grasp.approach.y, 'z': grasp.approach.z},
                'binormal': {'x': grasp.binormal.x, 'y': grasp.binormal.y, 'z': grasp.binormal.z},
                'axis': {'x': grasp.axis.x, 'y': grasp.axis.y, 'z': grasp.axis.z},
                'width': grasp.width.data,
                'sample': {'x': grasp.sample.x, 'y': grasp.sample.y, 'z': grasp.sample.z},
            })
        try:
            with open(OUTPUT_JSON_PATH, 'w') as f:
                json.dump(grasps_data, f, indent=4)
            rospy.loginfo("Successfully saved {} grasps to {}".format(len(grasps_data), OUTPUT_JSON_PATH))
            grasps_saved = True
        except Exception as e:
            rospy.logerr("Failed to save grasps to {}: {}".format(OUTPUT_JSON_PATH, e))
    # --- End Save JSON ---

    grasps_received_event.set() # Signal that grasps have been processed


# --- New Function: Create Grasp Visualization Points ---
def create_grasp_visualization_points(grasps):
    """Generates XYZ points representing the grasp structure (fingertips and approach)."""
    grasp_points = []
    arm_length = 0.05 # Visualize 5cm of the approach vector

    for grasp in grasps:
        pos = np.array([grasp.position.x, grasp.position.y, grasp.position.z])
        axis = np.array([grasp.axis.x, grasp.axis.y, grasp.axis.z])
        approach = np.array([grasp.approach.x, grasp.approach.y, grasp.approach.z]) # Get approach vector
        width = grasp.width.data

        # Normalize vectors (important!)
        axis_norm = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else axis
        approach_norm = approach / np.linalg.norm(approach) if np.linalg.norm(approach) > 1e-6 else approach

        # Calculate points for the two fingertips
        half_width_vec = axis_norm * (width / 2.0)
        p1 = pos + half_width_vec
        p2 = pos - half_width_vec

        # Calculate point representing the base along the approach vector
        p_base = pos - approach_norm * arm_length # Extend backwards from center

        grasp_points.append(list(p1)) # Fingertip 1
        grasp_points.append(list(p2)) # Fingertip 2
        grasp_points.append(list(pos)) # Grasp center
        grasp_points.append(list(p_base)) # Point along approach vector

    rospy.loginfo("Generated {} points for grasp visualization.".format(len(grasp_points)))
    return grasp_points
# --- End New Function ---

# --- New Function: Merge Clouds with Color ---
def pack_rgb(r, g, b):
    """Packs RGB bytes into a single float or uint32 suitable for PCL/ROS."""
    # For PointCloud2 with UINT32 field:
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    return rgb_int
    # For PCL PointXYZRGB (which often uses a float):
    # rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    # s = struct.pack('>I', rgb_int) # Pack as big-endian uint32
    # return struct.unpack('>f', s)[0] # Unpack as big-endian float

def create_colored_merged_cloud(original_pcl_cloud, grasp_points_xyz):
    """Creates a new PCL PointCloud XYZRGB merging original and grasp points."""
    merged_cloud = pcl.PointCloud_PointXYZRGB() # Create cloud with RGB field

    # Define colors (0-255)
    color_object = pack_rgb(255, 255, 255) # White
    color_grasp = pack_rgb(255, 0, 0)     # Red

    # Add original points
    original_points = original_pcl_cloud.to_list()
    merged_points_list = []
    for p in original_points:
        # Assuming original cloud is XYZ, add color
        merged_points_list.append([p[0], p[1], p[2], color_object])

    # Add grasp points
    for p in grasp_points_xyz:
        merged_points_list.append([p[0], p[1], p[2], color_grasp])

    # Load data into the PCL cloud object
    merged_cloud.from_list(merged_points_list)
    rospy.loginfo("Created merged cloud with {} points ({} original + {} grasp).".format(
        merged_cloud.size, len(original_points), len(grasp_points_xyz)))
    return merged_cloud
# --- End New Function ---


def main():
    """Main function to load, publish, wait for grasps, visualize, and publish combined cloud."""
    global grasps_saved, top_grasps, grasps_received_event
    grasps_saved = False # Reset flags on start
    top_grasps = []
    grasps_received_event.clear()

    # --- Argument Parsing ---
    pcd_file = PCD_FILE_PATH
    if len(sys.argv) > 1:
        pcd_file_arg = sys.argv[1]
        if os.path.exists(pcd_file_arg):
            pcd_file = pcd_file_arg
            rospy.loginfo("Using PCD file from argument: {}".format(pcd_file))
        else:
            rospy.logwarn("Provided PCD file '{}' not found. Using default: {}".format(pcd_file_arg, pcd_file))

    if not os.path.exists(pcd_file):
        rospy.logerr("PCD file not found: {}. Exiting.".format(pcd_file))
        return

    rospy.init_node(NODE_NAME, anonymous=True)

    # Publisher for the *original* point cloud (for GPD)
    original_pub = rospy.Publisher(CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)

    # Publisher for the *combined* point cloud (with grasps visualized)
    combined_pub = rospy.Publisher(COMBINED_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)

    # Subscriber for the grasps
    sub = rospy.Subscriber(GRASP_TOPIC, GraspConfigList, grasp_callback)

    # Load PCD file
    cloud_data = load_pcd(pcd_file)
    if cloud_data is None:
        return

    # Convert *original* cloud to ROS message
    cloud_msg = create_point_cloud2_message(cloud_data, FRAME_ID)

    # Allow time for connections
    rospy.sleep(1.0)

    # Publish the *original* point cloud message
    original_pub.publish(cloud_msg)
    rospy.loginfo("Published original point cloud to {}".format(CLOUD_TOPIC))

    rospy.loginfo("Waiting for grasps on {}... (Timeout: 30s)".format(GRASP_TOPIC))

    # Wait for the grasp_callback to signal completion or timeout
    event_triggered = grasps_received_event.wait(timeout=30.0)

    if event_triggered and top_grasps:
        rospy.loginfo("Grasps received. Generating visualization...")
        # 1. Generate points for grasp visualization
        grasp_viz_points = create_grasp_visualization_points(top_grasps)

        # 2. Create a merged PCL cloud with colors
        merged_colored_cloud = create_colored_merged_cloud(cloud_data, grasp_viz_points)

        # 3. Convert merged cloud to ROS message
        merged_cloud_msg = create_colored_point_cloud2_message(merged_colored_cloud, FRAME_ID)

        # 4. Publish the merged cloud
        combined_pub.publish(merged_cloud_msg)
        rospy.loginfo("Published combined cloud with grasp visualization to {}".format(COMBINED_CLOUD_TOPIC))

        # --- Add this: Save merged cloud to PLY file ---
        try:
            output_ply_path = "/workspace/merged_cloud_for_rerun.ply"
            pcl.save(merged_colored_cloud, output_ply_path, format="ply", binary=False) # Save as ASCII PLY
            rospy.loginfo("Successfully saved merged cloud with grasp visualization to {}".format(output_ply_path))
        except Exception as e:
            rospy.logerr("Failed to save merged cloud to {}: {}".format(output_ply_path, e))
        # --- End Add ---

    elif event_triggered: # Event triggered, but top_grasps is empty
         rospy.logwarn("Grasp callback finished, but no grasps were found or stored. No visualization published.")
    else:
        rospy.logwarn("Timeout waiting for grasps. No grasp visualization published.")

    # Keep node alive briefly to ensure messages are sent, then shutdown
    rospy.loginfo("Processing complete. Shutting down in 3 seconds...")
    rospy.sleep(3.0)
    rospy.signal_shutdown("Finished processing and visualization.")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("An unexpected error occurred: {}".format(e))
