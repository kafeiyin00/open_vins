/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#ifndef OV_MAPLOC_LOCALIZER_H
#define OV_MAPLOC_LOCALIZER_H

#include "orb_extractor.h"
#include "map.h"
#include "pose_solver.h"
#include "rig.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ov_maploc {

struct LocParams {
  int orb_n = 1000;              ///< online ORB budget per view
  double kf_radius = 4.0;        ///< m, map keyframes searched around the prediction ...
  double kf_radius_max = 12.0;   ///< ... growing x1.5 per miss up to this
  int kf_max = 12;               ///< nearest keyframes used
  double gate_margin_deg = 25.0; ///< frustum gate beyond the view's corner angle
  double ratio = 0.85;           ///< best / second best (distinct landmark) Hamming
  int max_hamming = 64;
  double ransac_px = 4.0;
  int ransac_trials = 1000;
  int min_inliers = 25;
  double max_tilt_deg = 3.0;     ///< PnP tilt vs VIO tilt
  double max_jump_m = 1.5;       ///< once locked: correction change per fix
  double max_jump_deg = 10.0;
  double alpha = 0.5;            ///< blend of an accepted fix into T_map_odom (first fix: 1) ...
  int good_inliers = 200;        ///< ... scaled down for fixes with fewer inliers (x0.2 at min_inliers)
  double jump_rate = 0.05;       ///< m/s: the jump gate grows by this per second since the last accepted fix
  int reanchor_n = 3;            ///< re-anchor: this many consecutive gate-rejected fixes that agree with each other ...
  int reanchor_inliers = 50;     ///< ... each with at least this many inliers ...
  double reanchor_tol_m = 0.3;   ///< ... and within this of the newest one (T_map_odom translation) ...
  double reanchor_tol_deg = 3.0; ///< ... and yaw: the VIO slipped, T_map_odom is reset to them (0: off)
  int threads = 1;               ///< cameras processed in parallel (RK3588: up to the free A76 cores)
};

struct LocResult {
  bool ok = false;
  bool reanchor = false; ///< accepted by resetting T_map_odom (consistent fixes beyond the jump gate)
  std::string why;
  int n_kf = 0, n_landmarks = 0, n_query = 0, n_corr = 0, inliers = 0;
  double radius = 0, tilt_deg = 0, jump_m = 0, jump_deg = 0;
  Eigen::Matrix4d T_pred = Eigen::Matrix4d::Identity(); ///< predicted IMU pose in the map
  Eigen::Matrix4d T_meas = Eigen::Matrix4d::Identity(); ///< measured IMU pose in the map (when PnP ran)
  double t_select = 0, t_orb = 0, t_match = 0, t_pnp = 0, t_total = 0; ///< seconds (t_orb/t_match summed over cameras)
};

/// Relocalization against a RuntimeMap with a known start (see ov_maploc/README.md).
/// Loosely coupled: OpenVINS keeps its own frame, this estimates T_map_odom in
/// 4 DoF from occasional absolute fixes.  localize() is meant for one worker
/// thread; T_map_odom()/predict() may be called from any thread.
class Localizer {
public:
  /// Throws if the calibration (fisheye intrinsics, extrinsics) differs from the map's.
  Localizer(std::shared_ptr<const RuntimeMap> map, const std::vector<FisheyeCam> &cams, const LocParams &p);

  void set_start(const Eigen::Matrix4d &T_map_odom);
  bool started() const;
  Eigen::Matrix4d T_map_odom() const;
  Eigen::Matrix4d predict(const Eigen::Matrix4d &T_odom_imu) const { return T_map_odom() * T_odom_imu; }

  /// One fix from the four fisheye grayscale images and the VIO pose of the same instant.
  /// stamp (s, image time) lets the jump gate grow with the time since the last accepted fix; <0: fixed gate
  LocResult localize(const std::vector<cv::Mat> &fisheye_grays, const Eigen::Matrix4d &T_odom_imu, double stamp = -1.0);

  const RuntimeMap &map() const { return *map_; }
  const LocParams &params() const { return p_; }

private:
  std::vector<Obs> match_camera(int ci, const cv::Mat &gray, const Eigen::Matrix4d &T_pred, const std::vector<int> &lm,
                                double &t_orb, int &n_query) const;
  LocResult miss(LocResult r, const std::string &why);

  std::shared_ptr<const RuntimeMap> map_;
  LocParams p_;
  std::vector<VirtualView> views_;
  std::vector<OrbExtractor> orbs_;
  RigModel rig_;
  std::vector<double> cos_gate_;
  mutable std::mutex mtx_;
  Eigen::Matrix4d T_map_odom_ = Eigen::Matrix4d::Identity();
  bool started_ = false, locked_ = false;
  int misses_ = 0;
  double last_ok_stamp_ = -1.0;
  struct Cand { Eigen::Matrix4d T_mo; double stamp; int inliers; };
  std::vector<Cand> cands_; ///< gate-rejected fixes, for the re-anchor vote
};

} // namespace ov_maploc

#endif
