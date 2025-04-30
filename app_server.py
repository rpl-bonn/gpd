#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask server for GPD interface.
This script creates a Flask server that wraps the GPD interface and makes it accessible via HTTP.
"""
from __future__ import division, print_function

import os
import sys
import json
import numpy as np
import tempfile
import argparse
from io import BytesIO


# Try to import PCL for point cloud handling
try:
    import pcl
except ImportError:
    print("PCL Python bindings not found. Will try Open3D as fallback.")
    pcl = None
    # Try to import Open3D as fallback
    try:
        import open3d as o3d
        print("Open3D found, will use as fallback for point cloud handling.")
        has_o3d = True
    except ImportError:
        raise ImportError("Neither PCL nor Open3D found. Cannot handle point clouds.")
# Import app.py for the grasp prediction functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import PointCloud, Config, Logger, predict_full_grasp

# Set up Flask
try:
    from flask import Flask, request, jsonify
    has_flask = True
except ImportError:
    print("Flask not found. Web API will not be available.")
    has_flask = False

# Create Flask app if available
if has_flask:
    app = Flask(__name__)
    logger = Logger("GPDServer")
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({'status': 'ok'})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Predict grasp poses from uploaded point clouds.
        
        Expected input:
        - item_cloud: Binary PCD file contents of item point cloud
        - env_cloud: Binary PCD file contents of environment point cloud
        - rotation_resolution: Number of rotation angles to try (default: 24)
        - top_n: Number of grasps per angle (default: 3)
        - n_best: Number of best grasps to return (default: 1)
        
        Returns:
        - tf_matrices: List of transformation matrices
        - widths: List of grasp widths
        - scores: List of grasp scores
        """
        # Check if request has the required files
        if 'item_cloud' not in request.files or 'env_cloud' not in request.files:
            return jsonify({'error': 'Missing point cloud files'}), 400
        
        # Parse parameters
        rotation_resolution = int(request.form.get('rotation_resolution', 24))
        top_n = int(request.form.get('top_n', 3))
        n_best = int(request.form.get('n_best', 1))
        
        # Load the point clouds
        item_file = request.files['item_cloud']
        env_file = request.files['env_cloud']
        
        # Create temporary files and save the uploads
        item_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pcd')
        env_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pcd')
        
        item_file.save(item_temp.name)
        env_file.save(env_temp.name)
        
        # Load point clouds with PCL
        if pcl is None:
            return jsonify({'error': 'PCL module not available'}), 500
        
        item_cloud_pcl = pcl.load(item_temp.name)
        env_cloud_pcl = pcl.load(env_temp.name)
        
        # Convert to our PointCloud type
        item_cloud = PointCloud(np.array(item_cloud_pcl.to_array()))
        env_cloud = PointCloud(np.array(env_cloud_pcl.to_array()))
        
        # Create a simple configuration
        # config = Config(
        #     # Add any configuration parameters here
        #     gripper_width=float(request.form.get('gripper_width', 0.08)),
        #     finger_depth=float(request.form.get('finger_depth', 0.05)),
        #     hand_depth=float(request.form.get('hand_depth', 0.10)),
        #     object_min_height=float(request.form.get('object_min_height', 0.005)),
        # )
        config = None
        # Call the predict_full_grasp function
        logger.info("Calling predict_full_grasp")
        tf_matrices, widths, scores = predict_full_grasp(
            item_cloud=item_cloud,
            env_cloud=env_cloud,
            config=config,
            logger=logger,
            rotation_resolution=rotation_resolution,
            top_n=top_n,
            n_best=n_best,
            vis_block=False
        )
        
        # Convert numpy arrays to lists for JSON serialization
        # Use list comprehension instead of tolist() for Python 2 compatibility
        result = {
            'tf_matrices': [matrix.tolist() for matrix in tf_matrices],
            'widths': widths.tolist() if len(widths) > 0 else [],
            'scores': scores.tolist() if len(scores) > 0 else []
        }
        
        return jsonify(result)
            
    
    # Add a simple web interface for testing
    @app.route('/', methods=['GET'])
    def index():
        """Simple web interface for testing."""
        html_content = '''
        <html>
        <head>
            <title>GPD Grasp Detection</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #333; }
                form { margin-top: 20px; }
                label { display: block; margin-top: 10px; }
                input, button { margin-top: 5px; }
                button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
                #result { margin-top: 20px; white-space: pre; background-color: #f5f5f5; padding: 10px; }
            </style>
        </head>
        <body>
            <h1>GPD Grasp Detection</h1>
            <form id="grasp-form" enctype="multipart/form-data" method="post" action="/predict">
                <label for="item-cloud">Item Point Cloud (PCD file):</label>
                <input type="file" id="item-cloud" name="item_cloud" accept=".pcd">
                
                <label for="env-cloud">Environment Point Cloud (PCD file):</label>
                <input type="file" id="env-cloud" name="env_cloud" accept=".pcd">
                
                <label for="rotation-resolution">Rotation Resolution:</label>
                <input type="number" id="rotation-resolution" name="rotation_resolution" value="24" min="1" max="100">
                
                <label for="top-n">Top N Grasps per Angle:</label>
                <input type="number" id="top-n" name="top_n" value="3" min="1" max="10">
                
                <label for="n-best">N Best Grasps to Return:</label>
                <input type="number" id="n-best" name="n_best" value="1" min="1" max="10">
                
                <button type="submit">Detect Grasps</button>
            </form>
            
            <div id="result"></div>
            
            <script>
                // Simple form submission with fetch API but with fallback for older browsers
                document.getElementById('grasp-form').addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    var resultDiv = document.getElementById('result');
                    resultDiv.textContent = 'Processing... Please wait.';
                    
                    var formData = new FormData(this);
                    
                    // Check if fetch API is available
                    if (window.fetch) {
                        fetch('/predict', {
                            method: 'POST',
                            body: formData
                        })
                        .then(function(response) {
                            return response.json();
                        })
                        .then(function(data) {
                            if (data.error) {
                                resultDiv.textContent = 'Error: ' + data.error;
                            } else {
                                resultDiv.textContent = JSON.stringify(data, null, 2);
                            }
                        })
                        .catch(function(error) {
                            resultDiv.textContent = 'Error: ' + error.message;
                        });
                    } else {
                        // Fallback for older browsers - just submit the form normally
                        this.submit();
                    }
                });
            </script>
        </body>
        </html>
        '''
        return html_content

def main():
    """Main function to start the Flask server."""
    if not has_flask:
        print("Cannot start server: Flask not installed")
        return 1
    
    parser = argparse.ArgumentParser(description="GPD Flask Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the server to")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    args = parser.parse_args()
    
    print("Starting GPD Flask server on {}:{}".format(args.host, args.port))
    
    # Check if we're in Python 2 or 3 and use appropriate API
    import sys
    if sys.version_info[0] >= 3:
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        # Python 2 compatibility - some older Flask versions have different API
        try:
            app.run(host=args.host, port=args.port, debug=args.debug)
        except TypeError:
            # Fall back to older Flask API without named arguments
            app.run(args.host, args.port, args.debug)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
