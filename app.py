#!/usr/bin/env python
"""
Simple ROS-based service for GPD (Grasp Pose Detection)
This MVP allows sending point clouds and receiving grasp candidates
Python 3.5 compatible version
"""

import os
import sys
import time
import numpy as np
from flask import Flask, request, jsonify
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Int64
from gpd_ros.msg import CloudSources, CloudIndexed, GraspConfigList
import tempfile
import rospy

# Set up Flask application
app = Flask(__name__)

# Global variables
last_grasp_config_list = None
grasp_result_received = False

# ROS callback for grasp configurations
def grasp_callback(grasp_config_list):
    global last_grasp_config_list, grasp_result_received
    print("Received grasp config list with {} grasps".format(len(grasp_config_list.grasps)))
    last_grasp_config_list = grasp_config_list
    grasp_result_received = True

# Convert Open3D point cloud to ROS PointCloud2
def o3d_to_ros_cloud(o3d_cloud, frame_id="base_link"):
    points = np.asarray(o3d_cloud.points)
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    cloud_msg = pc2.create_cloud_xyz32(header, points)
    return cloud_msg

# Create CloudIndexed message for GPD
def create_cloud_indexed(cloud_msg, indices=None):
    # Create CloudSources
    cloud_sources = CloudSources()
    cloud_sources.cloud = cloud_msg
    
    # Set up camera source
    cloud_sources.camera_source.resize(1)
    cloud_sources.camera_source[0] = 0  # Single camera source
    
    # Set up view point (assuming (0,0,0) for simplicity)
    cloud_sources.view_points.resize(1)
    cloud_sources.view_points[0].x = 0
    cloud_sources.view_points[0].y = 0
    cloud_sources.view_points[0].z = 0
    
    # Create CloudIndexed message
    cloud_indexed = CloudIndexed()
    cloud_indexed.cloud_sources = cloud_sources
    
    # If indices are provided, use them, otherwise use all points
    if indices is not None:
        cloud_indexed.indices = indices
    
    return cloud_indexed

@app.route('/detect_grasps', methods=['POST'])
def detect_grasps():
    global grasp_result_received, last_grasp_config_list
    
    # Reset grasp result flag
    grasp_result_received = False
    
    # Check that a point cloud file was provided
    if 'cloud' not in request.files:
        return jsonify({"error": "No point cloud file provided"}), 400
    
    cloud_file = request.files['cloud']
    print("Received point cloud file: {}".format(cloud_file.filename))
    
    # Save the file temporarily
    temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    cloud_file.save(temp_path)
    
    # Load the point cloud
    cloud = o3d.io.read_point_cloud(temp_path)
    print("Loaded point cloud with {} points".format(len(cloud.points)))
    
    # If indices were provided, parse them
    indices = None
    if 'indices' in request.form:
        indices = list(map(int, request.form['indices'].split(',')))
        print("Using {} sample indices".format(len(indices)))
    
    # Convert to ROS message
    cloud_msg = o3d_to_ros_cloud(cloud)
    
    # Create CloudIndexed message
    cloud_indexed = create_cloud_indexed(cloud_msg, indices)
    
    # Publish cloud for processing
    print("Publishing cloud for grasp detection")
    cloud_pub.publish(cloud_indexed)
    
    # Wait for results
    timeout = int(request.form.get('timeout', 30))
    start_time = time.time()
    
    while not grasp_result_received and time.time() - start_time < timeout:
        time.sleep(0.1)
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    if not grasp_result_received:
        return jsonify({"error": "Timeout waiting for grasp detection"}), 408
    
    # Process results
    grasps_list = []
    for grasp in last_grasp_config_list.grasps:
        grasp_data = {
            "position": {
                "x": grasp.position.x,
                "y": grasp.position.y,
                "z": grasp.position.z
            },
            "orientation": {
                "x": grasp.approach.x,
                "y": grasp.approach.y,
                "z": grasp.approach.z
            },
            "width": grasp.width.data,
            "score": grasp.score.data
        }
        grasps_list.append(grasp_data)
    
    # Return results
    return jsonify({
        "grasps": grasps_list,
        "count": len(grasps_list)
    })

if __name__ == "__main__":

    # Initialize ROS node
    print("Initializing ROS node")
    rospy.init_node('gpd_mvp_server', anonymous=True)
    
    # Create publishers and subscribers
    print("Setting up ROS publishers and subscribers")
    cloud_pub = rospy.Publisher('/cloud_indexed', CloudIndexed, queue_size=1)
    grasp_sub = rospy.Subscriber('/detect_grasps/clustered_grasps', 
                                    GraspConfigList, grasp_callback)
    
    # Give ROS time to set up connections
    print("Waiting for ROS connections")
    time.sleep(2)
    
    # Start Flask server
    print("Starting server on port 5000")
    app.run(host='0.0.0.0', port=5000,debug=True)
        
