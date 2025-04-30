#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light-weight client for the GPD docker service (app.py listening on
http://localhost:5000/predict).

Key points
----------
*  Accepts **Open3D PointCloud**, **NumPy (N×3)** or a **path** to .ply/.pcd.
*  No hard dependency on PCL any more.
*  Adds  drop-in-replacement helper `predict_full_grasp(…)`
   that mirrors the signature of utils.graspnet_interface.predict_full_grasp.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Tuple, Any

import numpy as np

# ─────────────────────── optional imports ─────────────────────────────── #
# HTTP
import requests
# Open3D – choose CPU vs CUDA depending on utils.recursive_config.Config
import open3d as o3d
# if _conf["device"] == "cuda":                     # use CUDA build if present
#     import open3d.cuda.pybind as o3d              # type: ignore
# else:
#     import open3d.cpu.pybind as o3d               # type: ignore
import open3d as o3d                              # type: ignore

# ──────────────────────────  client class ──────────────────────────────── #
class GPDClient:
    def __init__(
        self,
        server_url: str = "http://localhost:5000/predict",
        container_name: str = "gpd",
        docker_script: str = "run_docker_new.sh",
    ) -> None:
        self.server_url: str = server_url
        self.container_name: str = container_name
        self.docker_script: str = docker_script


    # ----- public API ----------------------------------------------------- #
    def predict_grasps(
        self,
        item_cloud: Any,
        env_cloud: Any | None = None,
        rotation_resolution: int = 24,
        top_n: int = 3,
        n_best: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Low-level function that really talks to the HTTP endpoint.
        Accepts Open3D cloud / NumPy array / file path.
        """


        item_path = self._serialise_cloud(item_cloud, "item_cloud")
        env_path  = self._serialise_cloud(env_cloud,  "env_cloud")

        files = {
            "item_cloud": open(item_path, "rb"),
            "env_cloud":  open(env_path,  "rb"),
        }
        data = {
            "rotation_resolution": rotation_resolution,
            "top_n":               top_n,
            "n_best":              n_best,
        }
        print(f"[gpd-client] → POST {self.server_url}")
        r = requests.post(self.server_url, files=files, data=data, timeout=300)
        for f in files.values():
            f.close()                            # close our handles

        if r.status_code != 200:
            raise RuntimeError(
                f"GPD server error {r.status_code}: {r.text[:200]}"
            )

        out = r.json()
        return (
            np.asarray(out.get("tf_matrices", []), dtype=np.float32),
            np.asarray(out.get("widths",      []), dtype=np.float32),
            np.asarray(out.get("scores",      []), dtype=np.float32),
        )

    # ----- drop-in wrapper ------------------------------------------------ #
    def predict_full_grasp(       # noqa: n802   (keep original CamelCase name)
        self,
        item_cloud: Any,
        env_cloud: Any | None,
        _config: Any | None = None,
        rotation_resolution: int = 24,
        top_n: int = 3,
        n_best: int = 60,
        vis_block: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Same signature as utils.graspnet_interface.predict_full_grasp.
        The *_config* and *vis_block* arguments are accepted for API
        compatibility but are **ignored** by the server.
        """
        return self.predict_grasps(
            item_cloud,
            env_cloud,
            rotation_resolution,
            top_n,
            n_best,
        )

    # --------------------------------------------------------------------- #
    #                           internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _empty_cpu_cloud() -> "o3d.geometry.PointCloud":
        return o3d.geometry.PointCloud()

    def _serialise_cloud(self, cloud: Any, tag: str) -> str:
        """
        Turn *cloud* into a .pcd file on disk → return path.
        Supports:  path str • Open3D PointCloud • NumPy (N×3)
        """
        # 1) already a path
        if isinstance(cloud, (str, os.PathLike)):
            path = os.fspath(cloud)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return path

        # 2) NumPy array
        if isinstance(cloud, np.ndarray):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cloud.astype(np.float32))
            cloud = pc                                  # fallthrough

        # 3) Open3D PointCloud
        if isinstance(cloud, o3d.geometry.PointCloud):
            fd, path = tempfile.mkstemp(prefix=f"{tag}_", suffix=".pcd")
            os.close(fd)                                # no need to keep it open
            o3d.io.write_point_cloud(path, cloud, write_ascii=False, compressed=True)
            return path

        # 4) None  →  empty cloud
        if cloud is None:
            return self._serialise_cloud(self._empty_cpu_cloud(), tag)

        raise TypeError(f"Unsupported point-cloud type: {type(cloud)}")


# ─────────────────────────── module-level helpers ──────────────────────── #
gpd_client = GPDClient()

def predict_grasps(*args, **kwds):          # legacy helper
    return gpd_client.predict_grasps(*args, **kwds)

def predict_full_grasp(*args, **kwds):      # ✨ drop-in helper
    return gpd_client.predict_full_grasp(*args, **kwds)