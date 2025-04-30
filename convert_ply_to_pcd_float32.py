# filepath: convert_ply_to_pcd_float32.py
import open3d as o3d
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python convert_ply_to_pcd_float32.py <input_ply_file> <output_pcd_file>")
    sys.exit(1)

ply_path = sys.argv[1]
pcd_path = sys.argv[2]

print(f"Loading PLY: {ply_path}")
try:
    pcd = o3d.io.read_point_cloud(ply_path)
except Exception as e:
    print(f"Error loading PLY file: {e}")
    sys.exit(1)

if not pcd.has_points():
    print("Failed to load points from PLY.")
    sys.exit(1)

print(f"Loaded {len(pcd.points)} points.")

# --- Convert points to float32 ---
points_np = np.asarray(pcd.points).astype(np.float32)
pcd.points = o3d.utility.Vector3dVector(points_np)
print("Converted points to float32.")

# --- Convert normals to float32 (if they exist) ---
if pcd.has_normals():
    normals_np = np.asarray(pcd.normals).astype(np.float32)
    pcd.normals = o3d.utility.Vector3dVector(normals_np)
    print("Converted normals to float32.")
else:
    print("No normals found in input.")
    # Ensure normals field is removed if points had it but conversion failed somehow
    if 'normals' in pcd.point_field_names:
         pcd.remove_point_field('normals')


# --- Handle Colors (PCD often uses uint32 RGB) ---
# Open3D uses float64[0-1]. For simplicity, we'll let Open3D save them as floats.
# If the grasp detection requires packed uint32 RGB, this needs more complex handling.
if pcd.has_colors():
     print("Colors found. Saving as float colors.")
else:
     print("No colors found in input.")
     # Ensure colors field is removed if points had it but conversion failed somehow
     if 'colors' in pcd.point_field_names:
         pcd.remove_point_field('colors')


# --- Save as ASCII PCD ---
print(f"Saving as ASCII PCD: {pcd_path}")
try:
    o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=True)
    print("Save complete.")
except Exception as e:
    print(f"Error saving PCD file: {e}")
    sys.exit(1)

# --- Verify Header ---
print("\nVerifying header of saved PCD:")
try:
    with open(pcd_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 15: # Print first 15 lines
                break
            print(line.strip())
except Exception as e:
    print(f"Could not read back saved PCD header: {e}")
