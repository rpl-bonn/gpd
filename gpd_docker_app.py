"""
GPD ROS Server Application (runs inside Docker)
This script starts a ROS node and HTTP server inside the Docker container
to process grasp detection requests.
"""
import os
import sys
import json
import time
import numpy as np
import open3d as o3d
import rospy
import rospkg
from flask import Flask, request, jsonify
import threading

# Add the necessary path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ROS interface module
from graspnet_ros_interface import o3d_to_ros_cloud, predict_grasps_ros, get_best_grasp_ros, visualize_grasps

app = Flask(__name__)

# Global cache to store the latest detection results
detection_cache = {
    "latest_item_cloud": None,
    "latest_env_cloud": None,
    "latest_results": None,
    "latest_timestamp": 0
}

# Initialize ROS node
def init_ros():
    """Initialize the ROS node"""
    rospy.init_node('gpd_http_server', anonymous=True, disable_signals=True)
    rospy.loginfo("GPD HTTP Server initialized")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the server is running"""
    return jsonify({
        "status": "ok", 
        "message": "GPD Docker App is running",
        "timestamp": time.time()
    })

@app.route('/detect_grasps', methods=['POST'])
def detect_grasps_endpoint():
    """HTTP endpoint to detect grasps on a point cloud"""
    try:
        # Check if we received data
        if 'item_cloud' not in request.files and 'item_cloud_path' not in request.form:
            return jsonify({"success": False, "message": "No item cloud provided"}), 400
        
        if 'item_cloud' in request.files:
            # Load point cloud from uploaded file
            item_cloud_file = request.files['item_cloud']
            item_cloud_path = f"/tmp/{time.time()}_item.ply"
            item_cloud_file.save(item_cloud_path)
            item_cloud = o3d.io.read_point_cloud(item_cloud_path)
            # Clean up temp file
            os.remove(item_cloud_path)
        else:
            # Load from specified path (must be accessible inside the container)
            item_cloud_path = request.form['item_cloud_path']
            if not os.path.exists(item_cloud_path):
                return jsonify({"success": False, "message": f"Item cloud file not found: {item_cloud_path}"}), 404
            item_cloud = o3d.io.read_point_cloud(item_cloud_path)
        
        # Handle environment cloud (optional)
        env_cloud = o3d.geometry.PointCloud()
        if 'env_cloud' in request.files:
            env_cloud_file = request.files['env_cloud']
            env_cloud_path = f"/tmp/{time.time()}_env.ply"
            env_cloud_file.save(env_cloud_path)
            env_cloud = o3d.io.read_point_cloud(env_cloud_path)
            os.remove(env_cloud_path)
        elif 'env_cloud_path' in request.form:
            env_cloud_path = request.form['env_cloud_path']
            if os.path.exists(env_cloud_path):
                env_cloud = o3d.io.read_point_cloud(env_cloud_path)
        
        # Get options
        best_only = request.form.get('best_only', 'false').lower() == 'true'
        visualize = request.form.get('visualize', 'false').lower() == 'true'
        visualization_path = request.form.get('visualization_path', '/tmp/grasp_visualization.ply')
        
        # Start timing
        start_time = time.time()
        
        # Detect grasps
        if best_only:
            result = get_best_grasp_ros(item_cloud, env_cloud)
            
            # Cache the result
            detection_cache["latest_item_cloud"] = item_cloud
            detection_cache["latest_env_cloud"] = env_cloud
            detection_cache["latest_results"] = result
            detection_cache["latest_timestamp"] = time.time()
            
            if result["success"] and visualize:
                tf_matrices = np.array([result["transform"]])
                widths = np.array([result["width"]])
                scores = np.array([result["score"]])
                
                visualize_grasps(item_cloud, env_cloud, tf_matrices, widths, scores, visualization_path)
                result["visualization_path"] = visualization_path
            
            result["processing_time"] = time.time() - start_time
            return jsonify(result)
        else:
            tf_matrices, widths, scores = predict_grasps_ros(item_cloud, env_cloud)
            
            # Create result dictionary
            result = {
                "success": len(scores) > 0,
                "num_grasps": len(scores),
                "transforms": tf_matrices.tolist() if len(scores) > 0 else [],
                "widths": widths.tolist() if len(scores) > 0 else [],
                "scores": scores.tolist() if len(scores) > 0 else [],
                "processing_time": time.time() - start_time
            }
            
            # Cache the result
            detection_cache["latest_item_cloud"] = item_cloud
            detection_cache["latest_env_cloud"] = env_cloud
            detection_cache["latest_results"] = result
            detection_cache["latest_timestamp"] = time.time()
            
            # Visualize if requested
            if visualize and len(scores) > 0:
                visualize_grasps(item_cloud, env_cloud, tf_matrices, widths, scores, visualization_path)
                result["visualization_path"] = visualization_path
            
            return jsonify(result)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

def run_flask():
    """Run Flask server"""
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    # Initialize ROS node
    init_ros()
    
    # Give ROS some time to initialize
    time.sleep(1)
    
    print("Starting GPD Docker App...")
    print("HTTP server will be available at http://localhost:5000")
    print("Use /detect_grasps endpoint to detect grasps")
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Keep the main thread alive
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
