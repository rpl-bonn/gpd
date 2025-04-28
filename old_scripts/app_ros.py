#!/usr/bin/env python3
"""
ROS-based server for GraspNet grasp detection system.
This module serves as the main server application for the GraspNet
system using the ROS interface instead of the HTTP-based approach.

Usage:
    1. Start the ROS master: roscore
    2. Start the GPD ROS node: roslaunch gpd_ros detect_grasps.launch
    3. Run this server: python app_ros.py
"""

import os
import time
import logging
import rospy
import numpy as np
import open3d as o3d
import tempfile
from flask import Flask, request, jsonify
from graspnet_ros_interface import predict_grasps_ros, get_best_grasp_ros

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)
app.config['DEBUG'] = True

# Initialize ROS node
rospy.init_node('graspnet_ros_server', anonymous=True, disable_signals=True)
logger.info("Initialized ROS node 'graspnet_ros_server'")

@app.route('/detect_grasps', methods=['POST'])
def detect_grasps():
    """API endpoint to detect grasps in a point cloud using ROS service."""
    logger.info("Received grasp detection request")
    
    try:
        # Check if files were uploaded
        if 'item_cloud' not in request.files:
            logger.error("No item point cloud file received")
            return jsonify({"error": "No item point cloud file provided"}), 400
            
        if 'env_cloud' not in request.files:
            logger.warning("No environment point cloud file received, using empty environment")
            # Create empty environment cloud
            env_cloud = o3d.geometry.PointCloud()
        else:
            # Save the environment cloud to a temporary file
            env_file = request.files['env_cloud']
            logger.info(f"Received environment cloud file: {env_file.filename}")
            env_temp_file = tempfile.NamedTemporaryFile(prefix='env_cloud_', suffix='.ply', delete=False)
            env_temp_path = env_temp_file.name
            env_temp_file.close()
            
            env_file.save(env_temp_path)
            logger.debug(f"Environment point cloud saved to: {env_temp_path}")
            
            # Load the environment cloud
            env_cloud = o3d.io.read_point_cloud(env_temp_path)
            
            # Clean up temporary file
            os.remove(env_temp_path)
        
        # Save the item cloud to a temporary file
        item_file = request.files['item_cloud']
        logger.info(f"Received item cloud file: {item_file.filename}")
        item_temp_file = tempfile.NamedTemporaryFile(prefix='item_cloud_', suffix='.ply', delete=False)
        item_temp_path = item_temp_file.name
        item_temp_file.close()
        
        item_file.save(item_temp_path)
        logger.debug(f"Item point cloud saved to: {item_temp_path}")
        
        # Load the item cloud
        item_cloud = o3d.io.read_point_cloud(item_temp_path)
        
        # Get parameters from request
        get_best_only = request.form.get('get_best_only', 'false').lower() == 'true'
        timeout = int(request.form.get('timeout', '90'))
        
        # Clean up temporary file
        os.remove(item_temp_path)
        
        # Start timing
        start_time = time.time()
        
        if get_best_only:
            # Get only the best grasp
            result = get_best_grasp_ros(item_cloud, env_cloud, timeout=timeout)
            execution_time = time.time() - start_time
            logger.info(f"Best grasp detection completed in {execution_time:.2f} seconds")
            return jsonify(result)
        else:
            # Get all grasps
            tf_matrices, widths, scores = predict_grasps_ros(item_cloud, env_cloud, timeout=timeout)
            
            execution_time = time.time() - start_time
            logger.info(f"Grasp detection completed in {execution_time:.2f} seconds")
            
            # Check if any grasps were found
            if len(scores) == 0:
                return jsonify({
                    "success": False,
                    "message": "No grasps found"
                })
            
            # Convert numpy arrays to lists for JSON serialization
            result = {
                "success": True,
                "tf_matrices": tf_matrices.tolist(),
                "widths": widths.tolist(),
                "scores": scores.tolist()
            }
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error during grasp detection: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    # Check if ROS services are available
    try:
        service_name = '/detect_grasps'
        available = rospy.wait_for_service(service_name, timeout=1)
        return jsonify({
            "status": "healthy",
            "ros_service": "available"
        })
    except rospy.ROSException:
        return jsonify({
            "status": "degraded",
            "ros_service": "unavailable",
            "message": f"ROS service {service_name} is not available"
        }), 503

if __name__ == '__main__':
    # Log info
    logger.info("Starting Flask server on port 5000")
    logger.info("Ensuring ROS services are available...")
    
    # Check for ROS services
    try:
        service_name = '/detect_grasps'
        logger.info(f"Waiting for ROS service: {service_name}")
        rospy.wait_for_service(service_name, timeout=5)
        logger.info(f"ROS service {service_name} is available")
    except rospy.ROSException as e:
        logger.warning(f"Warning: {service_name} service not available. {str(e)}")
        logger.warning("Make sure to run 'roslaunch gpd_ros detect_grasps.launch' first")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
