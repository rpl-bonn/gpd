import open3d as o3d
import numpy as np
import rerun as rr

# Load point clouds
env_pcd = o3d.io.read_point_cloud("env_cloud.ply")
item_pcd = o3d.io.read_point_cloud("item_cloud.ply")

# Convert to numpy arrays
env_points = np.asarray(env_pcd.points)
env_colors = np.asarray(env_pcd.colors) if env_pcd.has_colors() else None

item_points = np.asarray(item_pcd.points)
item_colors = np.asarray(item_pcd.colors) if item_pcd.has_colors() else None

# Initialize rerun viewer
rr.init("pointcloud_viewer", spawn=True)

# Log both clouds to different paths in the viewer
rr.log("environment", rr.Points3D(env_points, colors=env_colors))
rr.log("item", rr.Points3D(item_points, colors=item_colors))
