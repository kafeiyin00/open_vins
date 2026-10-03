# ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS

Relocalizes a drone with four 220° fisheyes against an ORB landmark map built
offline from an earlier flight, starting from a known take-off pose.
Loosely coupled: OpenVINS runs unmodified in its own `global` frame; this
package only estimates `T_map_odom` (map → global) in 4 DoF (x, y, z, yaw;
both frames are gravity aligned) and publishes it as TF.

C++ only (OpenCV + Eigen, no pycolmap), for deployment on an RK3588.

## Pipeline

**Map (offline, on the host).** Two builders:

- `maploc_build` (this package, C++, needs Ceres): keyframes of a mapping run
  (VIO pose + one image per camera, `index.txt`) → each fisheye rendered into
  the 120° virtual pinhole view with the localizer's own `VirtualView` and ORB
  → loop closure (`--loop 1`, `src/loop_closure.cpp`): a flat bag of words
  (1000 words, k-majority on the run's own ORB, TF-IDF, DBoW's L1 score)
  proposes older keyframes (≥ 20 s) that look alike; each candidate is
  verified by the generalized PnP of the keyframe's four views against the
  landmarks triangulated around the older one (≥ 40 inliers, tilt within
  3°, a plausible correction); a 4-DoF pose graph (x, y, z, yaw, VINS-Mono
  style) of the VIO's relative motion and the loops corrects the drift, the
  loop it agrees with least dropped until the rest fit (0.3 m / 3°)
  → image pairs chosen from the corrected poses (close camera centres and
  optical axes, any time apart: revisits) → Hamming + ratio + mutual
  matching verified by an essential-matrix RANSAC → union-find tracks,
  multi-view triangulation → Ceres bundle adjustment (consecutive keyframes
  held to the VIO's relative motion, every keyframe to the VIO's tilt, the
  first one fixed) → `map.bin` version 2, `report.json`, `preview.json`, and a
  self test (the Localizer against the new map on the run's own keyframes).
  The RK3588_SLAM dev PC runs it from its calibration page on an uploaded bag
  (`ros/pc/web`: open_vins on the bag, keyframes, then this).
- NV_SIM (`scripts/40_maploc_poses.sh`, `scripts/41_maploc_build_map.py`):
  the same steps with COLMAP rig triangulation and bundle adjustment with
  weak VIO position priors → `map.npz`, converted here with
  `scripts/map_npz_to_bin.py` to `map.bin` version 1 (ideal r = f θ
  fisheyes only).

Camera model: OpenVINS' equidistant fisheye (Kannala-Brandt, Kalibr's
pinhole-equi: fx fy cx cy k1–k4), which the map stores per camera (version
2; version 1 maps load as k = 0, fx = fy).

**Relocalization (online, `Localizer`, ~1 Hz).**
1. predict the IMU pose in the map: `T_map_imu = T_map_odom · T_odom_imu`;
2. candidate landmarks: those seen by the map keyframes nearest to the
   prediction (radius 4 m, ×1.5 per miss up to 12 m);
3. per camera: render the same virtual view, ORB, keep candidates in front of
   the camera, Hamming matching with a ratio test between distinct landmarks;
4. generalized 4-camera absolute pose: RANSAC over a linear 3-point solver
   with gravity from the VIO (unknowns yaw + translation), re-fit on inliers,
   then 6-DoF Levenberg-Marquardt (Huber);
5. gates: ≥ 25 inliers, refined tilt within 3° of the VIO's, and once locked
   a correction jump ≤ 1.5 m / 10°; the fix is blended into `T_map_odom`
   (first fix replaces it).

Before the first fix the start pose is not trusted (the rig may not stand
where it was told to): no orientation gate (the pose solver needs only the
VIO's gravity), and from the third miss on the whole map is walked, `kf_max`
consecutive keyframes per attempt; such a fix needs ≥ 40 inliers
(`search_global`, `search_min_inliers`). The tilt gate compares the gravity
direction in the IMU frame, which the yaw does not change.

Known start: by default the drone starts where the mapping flight started,
`T_map_odom = start_T_map_imu · inv(first VIO pose)`; or give `start:=x,y,yaw_deg`.

## Build and run

ROS 1 (`src/ros1_node.cpp`) or ROS 2 (`src/ros_node.cpp`), picked by the build
like the other OpenVINS packages; the two nodes have the same topics and
parameters (ROS 1: private `~` parameters).

```bash
python3 ov_maploc/scripts/map_npz_to_bin.py map.npz map.bin

# ROS 1 (catkin_make / catkin build)
roslaunch ov_maploc maploc.launch map:=/abs/map.bin \
    calib:=/abs/kalibr_imucam_chain.yaml masks:=/abs/masks [transport:=compressed]

# ROS 2
colcon build --packages-select ov_maploc
ros2 run ov_maploc maploc_node --ros-args -p map:=/abs/map.bin \
    -p calib:=/abs/kalibr_imucam_chain.yaml -p masks:=/abs/masks
```

ROS 1 only: `transport:=compressed` subscribes to `<topic>/compressed`
(`sensor_msgs/CompressedImage`) and decodes just the frames it relocalizes
(~1 Hz) instead of taking every raw frame of every camera; and a frame's
cameras are joined within 1 ms rather than by exact stamp (on the RK3588_SLAM
board one trigger's stamps differ by up to ~1 us between cameras).

| topic / frame | |
|---|---|
| in: camera topics from `calib` (`rostopic`), `sensor_msgs/Image` rgb8/bgr8/mono8 | the four fisheyes |
| in: `/ov_msckf/poseimu` | OpenVINS pose after each update |
| out: TF `map → global` | `T_map_odom` |
| out: `/maploc/pose` | `PoseWithCovarianceStamped` in `map`, at the VIO rate |
| out: `/maploc/status` | JSON per attempt: ok, reason, inliers, timings |
| out: `/maploc/map_points` | landmarks (transient local) |

Parameters: `reloc_hz` (1.0), `orb_n` (1000), `threads` (1; >1 processes the
cameras in parallel), `kf_radius`, `kf_radius_max`, `kf_max`, `ratio`,
`ransac_px`, `min_inliers`, `max_tilt_deg`, `max_jump_m`, `max_jump_deg`,
`alpha`, `start`, `map_frame` (`map`), `odom_frame` (`global`).

In scripts, start the installed executable (`$(ros2 pkg prefix ov_maploc)/lib/ov_maploc/maploc_node`)
rather than a backgrounded `ros2 run`: a background job of a non-interactive
shell ignores SIGINT and the `ros2 run` wrapper does not pass it on.

The node refuses a map built with a different calibration (fisheye intrinsics
or extrinsics); view size/FOV and ORB parameters are taken from the map.
OpenVINS stamps `poseimu` at `t_img + td` with an online time offset; the node
tracks `td` when pairing images with poses.

## Benchmark without ROS (e.g. on the board)

```bash
maploc_bench --map map.bin --calib kalibr_imucam_chain.yaml --masks masks/ --frames frames/ [--threads 4]
```

`frames/` is exported on the host from a query flight (NV_SIM
`scripts/45_maploc_export_frames.py`): `index.txt` with the known start and,
per frame, the VIO pose and ground truth in the map frame, plus
`cam<c>_<i>.png`.  Prints the fix rate, error against ground truth and
per-stage timing.

## Results (simulation, NV_SIM; Xeon host, single thread)

| map → query | fixes | fix error, median / p95 | known start only | time / fix |
|---|---|---|---|---|
| indoor office, flight 1 → 2 | 134/134 | 0.021 / 0.035 m | 0.49 m | 56 ms |
| outdoor campus, flight 1 → 5 | 164/166 | 0.219 / 0.360 m | 0.82 m | 84 ms (52 ms, 4 threads) |
| warehouse figure-8, laps 1–2 → 3–5 | 99/105 | 0.241 / 0.352 m | 1.02 m | 46 ms |

Live, OpenVINS + `maploc_node` on a replayed flight (position RMSE in the map
frame, no alignment): indoor 319/320 fixes, 0.037 m (known start only
0.48 m); outdoor 533/540, 0.244 m (0.43 m); 54–64 ms per fix.

The outdoor and warehouse errors are dominated by the map's scale, inherited
from the mapping flight's OpenVINS (off by 2.5–3.4 % in those simulated
flights); within the map frame the relocalization itself is at 0.06–0.1 m.
Same numbers as the Python reference implementation within 0.01 m.
RK3588 timing: not measured yet — run `maploc_bench` on the board.
