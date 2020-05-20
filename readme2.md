# Issue on installation on Ubuntu 18.04, remove -O3 compiler optimization flag in CMakeLists.txt of gpd and gpd_ros 
https://github.com/atenpas/gpd/issues/88#issuecomment-610466113

rosrun pcl_ros pcd_to_pointcloud Scene_1.pcd 0.1

rostopic echo /cloud_pcd | rostopic pub /camera/depth/points sensor_msgs/PointCloud2

# error while loading shared libraries: libpcl_features.so.1.9: cannot open shared object file: No such file or director
find library location; here its /usr/local/lib; then

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
check with: echo $LD_LIBRARY_PATH
sudo ldconfig