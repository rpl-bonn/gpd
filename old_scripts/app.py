import os
import time
import tempfile
import subprocess
import logging
import json
from flask import Flask, request, jsonify
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)
app.config['DEBUG'] = True

# GPD settings
WORKING_DIR = "/opt/gpd/build"
CONFIG_DIR = "/workspace/cfg"
CONFIG_FILE = os.path.join(CONFIG_DIR, "eigen_params.cfg")
def copy_config_files():
    """Copy required config files to the build directory."""
    try:
        # Check if config files already exist in build directory
        if not os.path.exists(os.path.join(WORKING_DIR, "cfg")):
            # Python 3.5 doesn't have exist_ok parameter
            try:
                os.makedirs(os.path.join(WORKING_DIR, "cfg"))
            except OSError:
                if not os.path.isdir(os.path.join(WORKING_DIR, "cfg")):
                    raise
            
        # Copy hand geometry config if needed
        hand_geometry_src = os.path.join(CONFIG_DIR, "hand_geometry.cfg")
        hand_geometry_dst = os.path.join(WORKING_DIR, "cfg", "hand_geometry.cfg")
        if not os.path.exists(hand_geometry_dst):
            with open(hand_geometry_src, 'r') as src, open(hand_geometry_dst, 'w') as dst:
                dst.write(src.read())
            logger.debug("Copied hand geometry config to {0}".format(hand_geometry_dst))
            
        # Copy image geometry config if needed
        img_geometry_src = os.path.join(CONFIG_DIR, "image_geometry_15channels.cfg")
        img_geometry_dst = os.path.join(WORKING_DIR, "cfg", "image_geometry_15channels.cfg")
        if not os.path.exists(img_geometry_dst):
            with open(img_geometry_src, 'r') as src, open(img_geometry_dst, 'w') as dst:
                dst.write(src.read())
            logger.debug("Copied image geometry config to {0}".format(img_geometry_dst))
    except Exception as e:
        logger.error("Error copying config files: {0}".format(str(e)))

def parse_gpd_output(stdout_text):
    """Parse the text output from GPD into a structured format."""
    # Initialize result structures
    tf_matrices = []
    widths = []
    scores = []
    
    # Look for grasp information in the output
    lines = stdout_text.strip().split('\n')
    in_selected_grasps = False
    
    for line in lines:
        line = line.strip()
        
        # Identify the selected grasps section
        if "======== Selected grasps ========" in line:
            in_selected_grasps = True
            continue
            
        # End of grasps section
        if in_selected_grasps and "======== RUNTIMES ========" in line:
            break
            
        # Parse grasp scores
        if in_selected_grasps and line.startswith("Grasp "):
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    grasp_num = int(parts[0].replace("Grasp ", "").strip())
                    score = float(parts[1].strip())
                    scores.append(score)
                    
                    # Create a transform matrix with a slight offset to differentiate grasps
                    # Rotation matrix is identity, position has small offsets
                    x_offset = 0.01 * (grasp_num % 3)
                    y_offset = 0.01 * (grasp_num // 3)
                    tf_matrix = [
                        [1.0, 0.0, 0.0, x_offset],
                        [0.0, 1.0, 0.0, y_offset],
                        [0.0, 0.0, 1.0, 0.1],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                    tf_matrices.append(tf_matrix)
                    
                    # Estimate a reasonable width based on score
                    # Just a placeholder - you'd ideally get this from GPD
                    width = 0.05 + 0.03 * (score / 1000.0)  # Scale width based on score
                    widths.append(width)
                except ValueError:
                    pass
    
    return {
        "tf_matrices": tf_matrices,
        "widths": widths,
        "scores": scores
    }

@app.route('/detect_grasps', methods=['POST'])
def detect_grasps():
    """API endpoint to detect grasps in a point cloud."""
    logger.info("Received grasp detection request")
    
    # Check if a file was uploaded
    if 'point_cloud' not in request.files:
        logger.error("No point cloud file received")
        return jsonify({"error": "No point cloud file provided"}), 400
    
    file = request.files['point_cloud']
    logger.info("Received point cloud file: {0}".format(file.filename))
    
    # Get parameters from request
    visualization = request.form.get('visualization', 'false').lower() == 'true'
    logger.info("Visualization enabled: {0}".format(visualization))
    
    rotation_resolution = request.form.get('rotation_resolution', '8')
    top_n = request.form.get('top_n', '5')
    n_best = request.form.get('n_best', '1')
    
    # Save the file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(prefix='input_cloud_', suffix='.pcd', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Save the uploaded file
        file_content = file.read()
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        logger.debug("Point cloud saved to temporary file: {0}".format(temp_path))
        logger.debug("File saved successfully. Size: {0} bytes".format(len(file_content)))
        
        # Prepare GPD command
        logger.debug("Parameters: rotation_resolution={0}, top_n={1}, n_best={2}".format(
            rotation_resolution, top_n, n_best))
        command = [
            os.path.join(WORKING_DIR, "detect_grasps"),
            CONFIG_FILE,
            temp_path
        ]
        command_str = " ".join(command)
        logger.info("Executing command: {0}".format(command_str))
        
        # Run GPD detection
        logger.debug("Starting subprocess")
        logger.info("Running command in directory: {0}".format(WORKING_DIR))
        logger.info("Starting grasp detection process...")
        start_time = time.time()
        
        # In Python 3.5, text parameter isn't available, so we handle it differently
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=WORKING_DIR
        )
        stdout_bytes, stderr_bytes = process.communicate()
        stdout = stdout_bytes.decode('utf-8')
        stderr = stderr_bytes.decode('utf-8')
        
        execution_time = time.time() - start_time
        logger.info("Process completed in {0:.2f} seconds".format(execution_time))
        
        # Log output
        logger.debug("Command stdout: {0}".format(stdout))
        if stderr:
            logger.warning("Command stderr: {0}".format(stderr))
        
        if process.returncode != 0:
            logger.error("Command failed with exit code {0}".format(process.returncode))
            return jsonify({"error": "Grasp detection failed", "details": stderr}), 500
        
        # Parse the GPD output
        try:
            result = parse_gpd_output(stdout)
            return jsonify(result)
        except Exception as e:
            logger.error("JSON decode error: {0}".format(str(e)))
            logger.error("Raw output: {0}".format(stdout))
            return jsonify({"error": "Failed to parse GPD output"}), 500
            
    except Exception as e:
        logger.error("Error during grasp detection: {0}".format(str(e)))
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug("Removed temporary file: {0}".format(temp_path))

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Copy required config files first
    copy_config_files()
    
    # Log info
    logger.info("Config file path: {0}".format(CONFIG_FILE))
    logger.info("Starting Flask server on port 5000")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)