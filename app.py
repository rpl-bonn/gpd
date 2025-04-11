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
# Assume the GPD executable (e.g., "detect_grasps") is built in the same directory as app.py.
EXECUTABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect_grasps")
# Configuration file path for GPD (e.g., located under a "cfg" subdirectory)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", "eigen_params.cfg")
if not os.path.exists(EXECUTABLE):
    logger.error(f"GPD executable not found at {EXECUTABLE}")
if not os.path.exists(CONFIG_FILE):
    logger.error(f"GPD config file not found at {CONFIG_FILE}")
else:
    logger.info(f"Config file path: {CONFIG_FILE}")

@app.route('/detect_grasps', methods=['POST'])
def detect_grasps():
    logger.info("Received grasp detection request")
    
    # Validate that a file is included
    if 'point_cloud' not in request.files:
        logger.warning("No point cloud file provided in request")
        return jsonify({'error': 'No point cloud file provided. Use key "point_cloud".'}), 400
    
    point_cloud_file = request.files['point_cloud']
    logger.info(f"Received point cloud file: {point_cloud_file.filename}")
    
    # Save the incoming point cloud to a temporary file.
    temp_dir = tempfile.gettempdir()
    input_file_path = os.path.join(temp_dir, "input_cloud.pcd")
    try:
        point_cloud_file.save(input_file_path)
        logger.debug(f"Point cloud saved to temporary file: {input_file_path}")
        
        # Check if file exists and report its size
        if os.path.exists(input_file_path):
            file_size = os.path.getsize(input_file_path)
            logger.debug(f"File saved successfully. Size: {file_size} bytes")
        else:
            logger.error(f"File not found after saving: {input_file_path}")
            return jsonify({'error': 'Failed to save input file properly'}), 500
    except Exception as e:
        logger.error(f"Exception when saving input file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to save input file', 'details': str(e)}), 500

    # Retrieve extra parameters from the form data. Provide defaults if not given.
    try:
        rotation_resolution = int(request.form.get('rotation_resolution', '24'))
        top_n = int(request.form.get('top_n', '3'))
        n_best = int(request.form.get('n_best', '1'))
        logger.debug(f"Parameters: rotation_resolution={rotation_resolution}, top_n={top_n}, n_best={n_best}")
    except Exception as e:
        logger.error(f"Exception parsing parameters: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Invalid parameter format', 'details': str(e)}), 400

    # (Optional) You can expand this dictionary with more parameters if needed.
    additional_params = {
        "rotation_resolution": rotation_resolution,
        "top_n": top_n,
        "n_best": n_best
    }

    # Build the command.
    # Here we use the executable, CONFIG_FILE, and the saved point cloud file.
    # If your GPD executable supports passing extra parameters, add them to the command list.
    command = [EXECUTABLE, CONFIG_FILE, input_file_path]
    # Optionally, you could extend the command by appending extra parameters.
    # For example, if your executable accepts them as command-line arguments:
    # command.extend([str(rotation_resolution), str(top_n), str(n_best)])
    
    logger.info(f"Executing command: {' '.join(command)}")
    
    try:
        # Check if executable exists
        if not os.path.exists(EXECUTABLE):
            logger.error(f"Executable not found: {EXECUTABLE}")
            return jsonify({"error": "GPD executable not found", "path": EXECUTABLE}), 500
        
        # Check if config file exists
        if not os.path.exists(CONFIG_FILE):
            logger.error(f"Config file not found: {CONFIG_FILE}")
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
        logger.debug(f"Command stdout: {stdout}")
        if stderr:
            logger.warning(f"Command stderr: {stderr}")
            
        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            return jsonify({
                "error": "GPD command failed",
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr
            }), 500
            
        # Decode output; it must be formatted as JSON.
        result_json = stdout.strip()
        logger.debug(f"Attempting to parse JSON result: {result_json[:200]}...")  # Log first 200 chars
        
        # Parse JSON. (Ensure that your GPD executable is modified to output JSON!)
        try:
            result = json.loads(result_json)
            logger.info("Successfully parsed JSON result")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Raw output: {result_json}")
            return jsonify({
                "error": "Failed to parse GPD output as JSON",
                "details": str(e),
                "raw_output": result_json[:1000]  # Include part of the raw output for debugging
            }), 500
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess error: {str(e)}")
        if hasattr(e, 'output'):
            logger.error(f"Process output: {e.output}")
        return jsonify({
            "error": "GPD command failed",
            "details": str(e),
            "output": e.output.decode("utf-8") if hasattr(e, 'output') else "No output"
        }), 500
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out: {str(e)}")
        return jsonify({
            "error": "GPD command timed out",
            "details": str(e),
            "command": ' '.join(command)
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
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
                logger.debug(f"Removed temporary file: {input_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    # Return the result as JSON.
    logger.info("Returning successful response")
    return jsonify(result)

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any uncaught exception"""
    logger.error(f"Uncaught exception: {str(e)}")
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