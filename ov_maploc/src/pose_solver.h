/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#ifndef OV_MAPLOC_POSE_SOLVER_H
#define OV_MAPLOC_POSE_SOLVER_H

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace ov_maploc {

/// One 2D-3D correspondence seen by camera `cam` of the rig.
struct Obs {
  int cam = 0;
  Eigen::Vector2d uv;  ///< pixel, OpenCV convention, in the camera's pinhole view
  Eigen::Vector3d X;   ///< landmark in the map
};

/// Rig of pinhole views sharing one intrinsic (all virtual views have the same spec).
struct RigModel {
  std::vector<Eigen::Matrix3d> R_imu_cam;
  std::vector<Eigen::Vector3d> p_imu_cam;
  double f = 0, cx = 0, cy = 0;
};

struct PoseOptions {
  double max_px = 4.0;        ///< inlier reprojection threshold
  int max_trials = 1000;      ///< RANSAC upper bound (adaptive)
  double confidence = 0.999;
  int refine_iters = 15;
  double huber_px = 2.0;
  uint32_t seed = 42;
};

struct PoseResult {
  bool ok = false;
  Eigen::Matrix4d T_map_imu = Eigen::Matrix4d::Identity();
  int inliers = 0;
  int trials = 0;
  std::vector<uint8_t> inlier;
};

/// Generalized (multi-camera) absolute pose.
///
/// Hypotheses come from a linear 3-point solver with the gravity direction
/// fixed to R_tilt (the tilt of the VIO prediction; unknowns yaw and
/// translation), scored by reprojection error.  The winner is re-fitted on
/// its inliers and then refined in all 6 DoF (Levenberg-Marquardt, Huber), so
/// the caller can still compare the refined tilt with the VIO's.
PoseResult solve_generalized_pose(const std::vector<Obs> &obs, const RigModel &rig, const Eigen::Matrix3d &R_tilt,
                                  const PoseOptions &opt);

} // namespace ov_maploc

#endif
