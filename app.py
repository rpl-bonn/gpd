#!/usr/bin/env python3
import os
import tempfile
import subprocess
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Change these paths if necessary.
# Assume the GPD executable (e.g., "detect_grasps") is built in the same directory as app.py.
EXECUTABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect_grasps")
# Configuration file path for GPD (e.g., located under a "cfg" subdirectory)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", "eigen_params.cfg")

@app.route('/detect_grasps', methods=['POST'])
def detect_grasps():
    # Validate that a file is included
    if 'point_cloud' not in request.files:
        return jsonify({'error': 'No point cloud file provided. Use key "point_cloud".'}), 400
    
    point_cloud_file = request.files['point_cloud']
    
    # Save the incoming point cloud to a temporary file.
    temp_dir = tempfile.gettempdir()
    input_file_path = os.path.join(temp_dir, "input_cloud.pcd")
    try:
        point_cloud_file.save(input_file_path)
    except Exception as e:
        return jsonify({'error': 'Failed to save input file', 'details': str(e)}), 500

    # Retrieve extra parameters from the form data. Provide defaults if not given.
    try:
        rotation_resolution = int(request.form.get('rotation_resolution', '24'))
        top_n = int(request.form.get('top_n', '3'))
        n_best = int(request.form.get('n_best', '1'))
    except Exception as e:
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
    
    try:
        # Execute the GPD command.
        # We assume that the executable returns a JSON string on stdout.
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=60)
        # Decode output; it must be formatted as JSON.
        result_json = output.decode("utf-8").strip()
        # Parse JSON. (Ensure that your GPD executable is modified to output JSON!)
        result = json.loads(result_json)
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "GPD command failed",
            "details": e.output.decode("utf-8")
        }), 500
    except subprocess.TimeoutExpired as e:
        return jsonify({
            "error": "GPD command timed out",
            "details": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "error": "Failed to parse GPD output",
            "details": str(e)
        }), 500
    finally:
        # Remove the temporary point cloud file.
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
    
    # Return the result as JSON.
    return jsonify(result)

if __name__ == '__main__':
    # Listen on all interfaces (0.0.0.0) on port 5000
    app.run(host='0.0.0.0', port=5000)