#!/usr/bin/env python3
"""Convert a maploc runtime map (map.npz, written by the host-side map builder)
into the flat binary map.bin that ov_maploc (C++) loads.  numpy only.

  map_npz_to_bin.py map.npz [map.bin]

Format (little endian):
  "OVMAPLOC"  u32 version=1  u32 n_arrays
  n_arrays x { u32 name_len, name, u8 dtype, u8 ndim, u16 0, u64 shape[ndim], raw C-order data }
  dtype: 1 u8, 2 u16, 3 i32, 4 f32, 5 f64

Arrays: the npz arrays unchanged (points_*, desc*, kf_*), plus the fields the
C++ side needs from the JSON meta as numbers (start_T_map_imu, view,
orb, orb_n_features_map, rig_fisheye, rig_T_imu_cam, T_world_map) and the
meta itself as meta_json (u8).
"""
import json
import struct
import sys

import numpy as np

MAGIC = b"OVMAPLOC"
VERSION = 1
DTYPES = {np.dtype(np.uint8): 1, np.dtype(np.uint16): 2, np.dtype(np.int32): 3,
          np.dtype(np.float32): 4, np.dtype(np.float64): 5}


def convert(npz_path, bin_path):
    z = np.load(npz_path)
    meta = json.loads(bytes(z["meta"]).decode())
    arrays = {k: z[k] for k in z.files if k != "meta"}
    rig = meta["rig"]
    orb = meta["orb"]
    if orb.get("type", "ORB") != "ORB":
        raise ValueError("only ORB maps are supported")
    arrays["start_T_map_imu"] = np.array(meta["start_T_map_imu"], np.float64)
    arrays["view"] = np.array([rig["view"]["size"], rig["view"]["fov_deg"]], np.float64)
    arrays["orb"] = np.array([orb["scale_factor"], orb["n_levels"], orb["edge_threshold"], orb["patch_size"],
                              orb["fast_threshold"], orb["grid"]], np.float64)
    arrays["orb_n_features_map"] = np.array([meta["orb_n_features_map"]], np.int32)
    arrays["rig_fisheye"] = np.array([[c["f"], c["cx"], c["cy"], c["width"], c["height"]] for c in rig["cams"]], np.float64)
    arrays["rig_T_imu_cam"] = np.array([c["T_imu_cam"] for c in rig["cams"]], np.float64)
    if meta.get("T_world_map") is not None:
        arrays["T_world_map"] = np.array(meta["T_world_map"], np.float64)
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)

    with open(bin_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(arrays)))
        for name, a in arrays.items():
            a = np.ascontiguousarray(a)
            if a.dtype not in DTYPES:
                raise TypeError(f"{name}: unsupported dtype {a.dtype}")
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<BBH", DTYPES[a.dtype], a.ndim, 0))
            f.write(struct.pack(f"<{a.ndim}Q", *a.shape))
            f.write(a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes())
    return len(arrays)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src[:-4] + ".bin" if src.endswith(".npz") else src + ".bin"
    n = convert(src, dst)
    print(f"wrote {dst} ({n} arrays)")
