#!/usr/bin/env python3
"""
Simple command–line test script for interface.predict_full_grasp

Example
-------
python test_graspnet_predict.py \
       --item item_cloud.ply \
       --env  env_cloud.pcd \
       --config graspnet_cfg.yaml
"""

from __future__ import annotations
import argparse
import sys
import pathlib
from typing import Tuple, Any

import open3d as o3d                       # pip install open3d
import numpy as np
import yaml                                # PyYAML, pip install pyyaml

# Import the GPD client API
from gpd_client_api import predict_full_grasp, gpd_client

# ────────────────────────────── I/O HELPERS ────────────────────────────── #

def load_cloud(path: str | pathlib.Path) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    """
    Load a .ply or .pcd file with Open3D and return (Nx3 numpy array, o3d cloud).

    Exits with a helpful message if the file cannot be read or is empty.
    """
    path = pathlib.Path(path)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        sys.exit(f"[ERROR] Point cloud at {path} is empty or unreadable.")

    pts = np.asarray(cloud.points, dtype=np.float32)
    return pts, cloud

def load_config(path: str | pathlib.Path | None) -> Any:
    """
    Load a YAML config file if provided, else return an empty dict.

    Your `predict_full_grasp` may accept either a dict or a custom Config
    object—adjust this loader if that’s the case.
    """
    if path is None:
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

# ─────────────────────────────── MAIN LOGIC ────────────────────────────── #

def main(args: argparse.Namespace) -> None:
    # 0. Make sure the Docker‐backed GPD server is up

    # 1. Read the item and environment clouds
    item_pts, _item_cloud_o3d = load_cloud(args.item)
    env_pts,  _env_cloud_o3d  = load_cloud(args.env)

    # 2. (Optional) crop or filter env cloud here if you like
    lim_env_cloud = env_pts     # just pass through for now

    # 3. Load YAML config (ignored by server but kept for compatibility)
    config = load_config(args.config)

    # 4. Call the drop-in interface
    print("Connecting to GPD server and requesting grasp poses…")
    tf_matrices, widths, grasp_scores = print("Item cloud points:", _item_cloud_o3d.points)
print("Env cloud points:", _env_cloud_o3d.points)
predict_full_grasp(
        _item_cloud_o3d,        # Open3D PointCloud
        _env_cloud_o3d,         # Open3D PointCloud (or None)
        config,                 # keeps signature compat
        rotation_resolution=args.rotation_resolution,
        top_n=args.top_n,
        n_best=args.n_best,
        vis_block=args.vis,
    )

    # 5. Pretty-print the results
    print(f"\nReturned {tf_matrices.shape[0]} grasp candidates:\n" + "-" * 60)
    for i, (tf, w, s) in enumerate(zip(tf_matrices, widths, grasp_scores)):
        print(f"#{i:02d} | score={s:6.3f} | width={w:7.4f} m")
        np.set_printoptions(precision=4, suppress=True)
        print(tf, "\n")

# ────────────────────────────── CLI INTERFACE ──────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test your interface.predict_full_grasp with a pair of point clouds."
    )
    parser.add_argument("--item", required=True,
                        help="Path to the *item* point cloud (.ply or .pcd).")
    parser.add_argument("--env",  required=True,
                        help="Path to the *environment* point cloud (.ply or .pcd).")
    parser.add_argument("--config", default=None,
                        help="Optional YAML configuration file for the grasp planner.")
    parser.add_argument("--rotation_resolution", type=int, default=24,
                        help="Number of discrete yaw angles to sample (default: 24).")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Number of grasp clusters to return (default: 3).")
    parser.add_argument("--n_best", type=int, default=60,
                        help="Number of raw grasps to keep before clustering (default: 60).")
    parser.add_argument("--vis", action="store_true",
                        help="This option is currently ignored by the server API.")
    args = parser.parse_args()

    main(args)