docker run --gpus all -it -p 5000:5000 \
  -v /home/user/azirar/docker_containers/grasp_pose_detection/gpd:/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=:2 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e MESA_GL_VERSION_OVERRIDE=3.3 \
  --net=host \
  gpd \
  bash  

