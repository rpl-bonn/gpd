#!/bin/bash
# Make all GPD scripts executable

# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# List of scripts to make executable
SCRIPTS=(
    "launch_gpd.sh"
    "run_docker_new.sh"
    "run_gpd_docker.sh"
    "start_gpd_services.sh"
    "test_gpd_client.sh"
    "use_gpd.sh"
    "app_server.py"
    "direct_client.py"
    "gpd_client_api.py"
    "gpd_example.py"
    "gpd_example_external.py"
    "gpd_external_client.py"
    "test_gpd.py"
)

echo "Making GPD scripts executable..."
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        echo "  Made executable: $script"
    else
        echo "  Warning: Script not found: $script"
    fi
done

echo "Done."
