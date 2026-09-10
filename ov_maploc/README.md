# ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS

Relocalizes a drone with four 220° fisheyes against an ORB landmark map built
offline from an earlier flight, starting from a known take-off pose.
Loosely coupled: OpenVINS runs unmodified in its own `global` frame; this
package only estimates `T_map_odom` (map → global) in 4 DoF (x, y, z, yaw;
both frames are gravity aligned) and publishes it as TF.

C++ only (OpenCV + Eigen, no pycolmap), for deployment on an RK3588.

## Pipeline

**Map (offline, on the host; not in this package).** The map builder lives in
the NV_SIM repository (`scripts/40_maploc_poses.sh`, `scripts/41_maploc_build_map.py`):
OpenVINS poses of a mapping flight → keyframes → each fisheye resampled into a
120° virtual pinhole view (COLMAP's fisheye models return NaN past 90°) → ORB
→ pairs chosen from the VIO poses → COLMAP rig triangulation and bundle
adjustment with weak VIO position priors → `map.npz`, converted here with
`scripts/map_npz_to_bin.py` to `map.bin`.

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

Known start: by default the drone starts where the mapping flight started,
`T_map_odom = start_T_map_imu · inv(first VIO pose)`; or give `start:=x,y,yaw_deg`.

## Build and run

```bash
colcon build --packages-select ov_maploc
python3 ov_maploc/scripts/map_npz_to_bin.py map.npz map.bin

ros2 run ov_maploc maploc_node --ros-args -p map:=/abs/map.bin \
    -p calib:=/abs/kalibr_imucam_chain.yaml -p masks:=/abs/masks
```

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
