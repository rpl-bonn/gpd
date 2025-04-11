#!/usr/bin/env python3
import os
import tempfile
import subprocess
import json
import logging
import traceback
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Change these paths if necessary.
# Path to the GPD executable
# WORKSPACE_PATH = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_PATH = os.path.abspath("opt/gpd")
EXECUTABLE = os.path.join(WORKSPACE_PATH, "build", "detect_grasps")
# Configuration file paths for GPD
CONFIG_FILE = os.path.join(WORKSPACE_PATH, "cfg", "eigen_params.cfg")
# Additional required config files
HAND_GEOMETRY_CFG = os.path.join(WORKSPACE_PATH, "cfg", "hand_geometry.cfg")
IMAGE_GEOMETRY_CFG = os.path.join(WORKSPACE_PATH, "cfg", "image_geometry_15channels.cfg")

if not os.path.exists(EXECUTABLE):
    logger.error("GPD executable not found at {}".format(EXECUTABLE))
    logger.warning("The executable needs to be built. Run 'mkdir build && cd build && cmake .. && make' in the GPD directory")
if not os.path.exists(CONFIG_FILE):
    logger.error("GPD config file not found at {}".format(CONFIG_FILE))
else:
    logger.info("Config file path: {}".format(CONFIG_FILE))

@app.route('/detect_grasps', methods=['POST'])
def detect_grasps():
    logger.info("Received grasp detection request")
    
    # Validate that a file is included
    if 'point_cloud' not in request.files:
        logger.warning("No point cloud file provided in request")
        return jsonify({'error': 'No point cloud file provided. Use key "point_cloud".'}), 400
    
    point_cloud_file = request.files['point_cloud']
    logger.info("Received point cloud file: {}".format(point_cloud_file.filename))
    
    # Save the incoming point cloud to a temporary file.
    temp_dir = tempfile.gettempdir()
    input_file_path = os.path.join(temp_dir, "input_cloud.pcd")
    try:
        point_cloud_file.save(input_file_path)
        logger.debug("Point cloud saved to temporary file: {}".format(input_file_path))
        
        # Check if file exists and report its size
        if os.path.exists(input_file_path):
            file_size = os.path.getsize(input_file_path)
            logger.debug("File saved successfully. Size: {} bytes".format(file_size))
        else:
            logger.error("File not found after saving: {}".format(input_file_path))
            return jsonify({'error': 'Failed to save input file properly'}), 500
    except Exception as e:
        logger.error("Exception when saving input file: {}".format(str(e)))
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to save input file', 'details': str(e)}), 500

    # Retrieve extra parameters from the form data. Provide defaults if not given.
    try:
        rotation_resolution = int(request.form.get('rotation_resolution', '24'))
        top_n = int(request.form.get('top_n', '3'))
        n_best = int(request.form.get('n_best', '1'))
        logger.debug("Parameters: rotation_resolution={}, top_n={}, n_best={}".format(rotation_resolution, top_n, n_best))
    except Exception as e:
        logger.error("Exception parsing parameters: {}".format(str(e)))
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Invalid parameter format', 'details': str(e)}), 400

    # (Optional) You can expand this dictionary with more parameters if needed.
    additional_params = {
        "rotation_resolution": rotation_resolution,
        "top_n": top_n,
        "n_best": n_best
    }

    # Check additional config files
    if not os.path.exists(HAND_GEOMETRY_CFG):
        logger.error("Hand geometry config file not found: {}".format(HAND_GEOMETRY_CFG))
        return jsonify({"error": "Hand geometry config file not found", "path": HAND_GEOMETRY_CFG}), 500
        
    if not os.path.exists(IMAGE_GEOMETRY_CFG):
        logger.error("Image geometry config file not found: {}".format(IMAGE_GEOMETRY_CFG))
        return jsonify({"error": "Image geometry config file not found", "path": IMAGE_GEOMETRY_CFG}), 500
    
    # Create symbolic links or copy config files to expected locations
    gpd_dir = os.path.dirname(EXECUTABLE)
    expected_cfg_dir = os.path.join(gpd_dir, "..", "cfg")
    os.makedirs(expected_cfg_dir, exist_ok=True)
    
    # Copy or link hand geometry config
    hand_geometry_dest = os.path.join(expected_cfg_dir, "hand_geometry.cfg")
    if not os.path.exists(hand_geometry_dest):
        with open(HAND_GEOMETRY_CFG, 'r') as src, open(hand_geometry_dest, 'w') as dst:
            dst.write(src.read())
        logger.debug("Copied hand geometry config to {}".format(hand_geometry_dest))
            
    # Copy or link image geometry config
    image_geometry_dest = os.path.join(expected_cfg_dir, "image_geometry_15channels.cfg")
    if not os.path.exists(image_geometry_dest):
        with open(IMAGE_GEOMETRY_CFG, 'r') as src, open(image_geometry_dest, 'w') as dst:
            dst.write(src.read())
        logger.debug("Copied image geometry config to {}".format(image_geometry_dest))
    
    # Build the command.
    # Here we use the executable, CONFIG_FILE, and the saved point cloud file.
    # If your GPD executable supports passing extra parameters, add them to the command list.
    command = [EXECUTABLE, CONFIG_FILE, input_file_path]
    # Extend command with additional parameters
    command.extend([str(rotation_resolution), str(top_n), str(n_best)])
    
    logger.info("Executing command: {}".format(' '.join(command)))
    
    try:
        # Check if executable exists
        if not os.path.exists(EXECUTABLE):
            logger.error("Executable not found: {}".format(EXECUTABLE))
            return jsonify({"error": "GPD executable not found", "path": EXECUTABLE}), 500
        
        # Check if config file exists
        if not os.path.exists(CONFIG_FILE):
            logger.error("Config file not found: {}".format(CONFIG_FILE))
            return jsonify({"error": "Config file not found", "path": CONFIG_FILE}), 500
            
        # Execute the GPD command.
        # We assume that the executable returns a JSON string on stdout.
        logger.debug("Starting subprocess")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate(timeout=60)
        
        # Log both stdout and stderr for debugging
        logger.debug("Command stdout: {}".format(stdout))
        if stderr:
            logger.warning("Command stderr: {}".format(stderr))
            
        if process.returncode != 0:
            logger.error("Command failed with exit code {}".format(process.returncode))
            return jsonify({
                "error": "GPD command failed",
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr
            }), 500
            
        # Decode output; it must be formatted as JSON.
        result_json = stdout.strip()
        logger.debug("Attempting to parse JSON result: {}...".format(result_json[:200]))  # Log first 200 chars
        
        # Parse JSON. (Ensure that your GPD executable is modified to output JSON!)
        try:
            result = json.loads(result_json)
            logger.info("Successfully parsed JSON result")
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: {}".format(str(e)))
            logger.error("Raw output: {}".format(result_json))
            return jsonify({
                "error": "Failed to parse GPD output as JSON",
                "details": str(e),
                "raw_output": result_json[:1000]  # Include part of the raw output for debugging
            }), 500
            
    except subprocess.CalledProcessError as e:
        logger.error("Subprocess error: {}".format(str(e)))
        if hasattr(e, 'output'):
            logger.error("Process output: {}".format(e.output))
        return jsonify({
            "error": "GPD command failed",
            "details": str(e),
            "output": e.output.decode("utf-8") if hasattr(e, 'output') else "No output"
        }), 500
    except subprocess.TimeoutExpired as e:
        logger.error("Command timed out: {}".format(str(e)))
        return jsonify({
            "error": "GPD command timed out",
            "details": str(e),
            "command": ' '.join(command)
        }), 500
    except Exception as e:
        logger.error("Unexpected error: {}".format(str(e)))
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to process GPD output",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500
    finally:
        # Remove the temporary point cloud file.
        if os.path.exists(input_file_path):
            try:
                os.remove(input_file_path)
                logger.debug("Removed temporary file: {}".format(input_file_path))
            except Exception as e:
                logger.warning("Failed to remove temporary file: {}".format(str(e)))
    
    # Return the result as JSON.
    logger.info("Returning successful response")
    return jsonify(result)

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any uncaught exception"""
    logger.error("Uncaught exception: {}".format(str(e)))
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal server error",
        "details": str(e),
        "traceback": traceback.format_exc()
    }), 500

if __name__ == '__main__':
    logger.info("Starting Flask server on port 5000")
    # Listen on all interfaces (0.0.0.0) on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)