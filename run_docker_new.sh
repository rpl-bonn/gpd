#!/bin/bash
# Run Docker container with ROS environment setup and our Flask server
docker run --gpus all -it -p 5000:5000 \
  -v /home/user/azirar/docker_containers/grasp_pose_detection/gpd:/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=:2 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e MESA_GL_VERSION_OVERRIDE=3.3 \
  --net=host \
  gpd \
  bash -c "Xvfb :2 -ac -screen 0 1024x768x24 -nolisten tcp > /dev/null 2>&1 & \
           sleep 2; \
           export DISPLAY=:2; \
           source /opt/ros/*/setup.bash; \
           roscore & \
           sleep 5; \
           cd; \
           cd .. ; \
           cp /workspace/gpd_ros/launch/ur5.launch opt/catkin_ws/src/gpd_ros/launch/ur5.launch; \
           sleep 2; \
           cd /opt/catkin_ws && source devel/setup.bash; \\
           # start the grasp-detection node
           roslaunch gpd_ros ur5.launch & \
           sleep 3; \
           cd /workspace; \
           python3 -m pip install flask numpy rospkg rospy; \
           python3 -m pip install open3d==0.8.0; \
           # Run our Flask server
           python3 app_server.py --host 0.0.0.0 --port 5000"
