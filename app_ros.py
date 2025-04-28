#!/usr/bin/env python3
"""
ROS-based server for GPD (Grasp Pose Detection) system.
This module serves as the main server application for GPD using ROS.

Usage:
    1. Run this inside the Docker container where GPD and ROS are set up
    
The server receives point clouds and optional sample indexes, 
processes them using the GPD ROS package, and returns grasp poses.
"""

import os
import time
import logging
import numpy as np
import tempfile
from flask import Flask, request, jsonify
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from gpd_ros.msg import CloudSources, CloudIndexed, GraspConfigList, GraspConfig
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header, Float64
import rospy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)

# Global variable to store the last received grasp list
last_grasp_list = None
grasp_received = False

def grasp_callback(grasp_list_msg):
    """Callback function for received grasp poses"""
    global last_grasp_list, grasp_received
    logger.info("Received grasp list with {} grasps".format(len(grasp_list_msg.grasps)))
    last_grasp_list = grasp_list_msg
    grasp_received = True

def convert_o3d_to_ros_cloud(o3d_cloud, frame_id="base_link"):
    """Convert Open3D point cloud to ROS PointCloud2 message"""
    points = np.asarray(o3d_cloud.points)
    
    # Create header
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    
    # Create PointCloud2 message
    cloud_msg = pc2.create_cloud_xyz32(header, points)
    return cloud_msg

def create_cloud_indexed_msg(cloud_msg, indices=None):
    """Create CloudIndexed message from PointCloud2"""
    if indices is None:
        # If no indices provided, use all points
        indices = list(range(cloud_msg.width * cloud_msg.height))
    
    # Create CloudSources message
    cloud_sources = CloudSources()
    cloud_sources.cloud = cloud_msg
    # Set camera_source as a list of zeros (single camera for all points)
    num_points = cloud_msg.width * cloud_msg.height
    cloud_sources.camera_source = [0] * num_points
    
    # Create a view point for the camera
    from geometry_msgs.msg import Point
    view_point = Point()
    view_point.x = 0
    view_point.y = 0
    view_point.z = 0
    cloud_sources.view_points = [view_point]
    
    # Create CloudIndexed message
    cloud_indexed = CloudIndexed()
    cloud_indexed.cloud_sources = cloud_sources
    cloud_indexed.indices = indices
    
    return cloud_indexed

def wait_for_grasp_results(timeout=30):
    """Wait for grasp results with timeout"""
    global grasp_received
    start_time = time.time()
    
    while not grasp_received and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    if not grasp_received:
        logger.warning("Timed out waiting for grasp results after {} seconds".format(timeout))
        return False
    
    return True

def process_grasp_results():
    """Process and format grasp results for API response"""
    global last_grasp_list, grasp_received
    
    if not grasp_received or last_grasp_list is None:
        return {"error": "No grasp poses detected"}
    
    grasps = []
    
    for grasp in last_grasp_list.grasps:
        grasp_pose = {
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
        grasps.append(grasp_pose)
    
    # Reset for next request
    grasp_received = False
    
    return {
        "grasps": grasps,
        "count": len(grasps)
    }

@app.route('/detect_grasps', methods=['POST'])
def detect_grasps():
    """Simple API endpoint to detect grasps in a point cloud using GPD ROS."""
    global grasp_received, last_grasp_list
    
    logger.info("Received grasp detection request")
    
    # Check if files were uploaded
    if 'cloud' not in request.files:
        logger.error("No point cloud file received")
        return jsonify({"error": "No point cloud file provided"}), 400
    
    # Save cloud to temporary file
    cloud_file = request.files['cloud']
    logger.info("Received cloud file: {}".format(cloud_file.filename))
    
    cloud_temp_file = tempfile.NamedTemporaryFile(prefix='cloud_', suffix='.ply', delete=False)
    cloud_temp_path = cloud_temp_file.name
    cloud_temp_file.close()
    
    cloud_file.save(cloud_temp_path)
    logger.debug("Point cloud saved to: {}".format(cloud_temp_path))
    
    # Load the cloud with Open3D
    cloud = o3d.io.read_point_cloud(cloud_temp_path)
    
    # Check if we have sample indices
    sample_indices = None
    if 'indices' in request.form:
        sample_indices = [int(i) for i in request.form['indices'].split(',')]
        logger.info("Received {} sample indices".format(len(sample_indices)))
    
    # Get timeout parameter
    timeout = int(request.form.get('timeout', '30'))
    
    # Clean up temporary file
    os.remove(cloud_temp_path)
    
    # Convert Open3D point cloud to ROS PointCloud2
    cloud_msg = convert_o3d_to_ros_cloud(cloud)
    
    # Create CloudIndexed message
    cloud_indexed_msg = create_cloud_indexed_msg(cloud_msg, sample_indices)
    
    # Reset grasp received flag
    grasp_received = False
    
    # Publish cloud to GPD
    cloud_pub.publish(cloud_indexed_msg)
    logger.info("Published point cloud to GPD")
    
    # Wait for results with the specified timeout
    if wait_for_grasp_results(timeout):
        # Process and return results
        result = process_grasp_results()
        return jsonify(result)
    else:
        # Simple failure if no response is received within timeout
        return jsonify({"error": "Timed out waiting for grasp detection results"}), 408

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify ROS connectivity"""
    # Check if ROS is still running
    ros_ok = not rospy.is_shutdown()
    
    if ros_ok:
        # Check subscribers to our topic
        num_subscribers = cloud_pub.get_num_connections()
        
        return jsonify({
            "status": "ok",
            "ros_ok": True,
            "subscribers": num_subscribers,
            "ready": num_subscribers > 0,
            "grasp_callback_registered": grasp_sub is not None,
            "last_grasp_received": grasp_received
        })
    else:
        logger.error("ROS core is shutdown")
        return jsonify({
            "status": "error",
            "message": "ROS core is shutdown", 
            "ros_ok": False
        }), 500

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('gpd_ros_server', anonymous=True)
    logger.info("Initialized ROS node 'gpd_ros_server'")
    
    # Create publisher for cloud indexed
    cloud_pub = rospy.Publisher('/cloud_indexed', CloudIndexed, queue_size=1)
    
    # Subscribe to grasp detections
    grasp_sub = rospy.Subscriber('/detect_grasps/clustered_grasps', GraspConfigList, grasp_callback)
    
    # Wait for subscribers to connect
    time.sleep(1)
    
    logger.info("Starting Flask server...")
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000,debug=True)

