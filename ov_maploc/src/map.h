/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#ifndef OV_MAPLOC_MAP_H
#define OV_MAPLOC_MAP_H

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ov_maploc {

/// Runtime map: ORB landmarks + keyframe visibility, as written by
/// scripts/map_npz_to_bin.py.  All poses are T_a_b (points in b -> a).
struct RuntimeMap {
  // landmarks (N)
  std::vector<Eigen::Vector3d> points_xyz;
  std::vector<float> points_err;
  std::vector<uint16_t> points_track;
  // descriptors (M x 32), rows of landmark p: point_desc_rows[point_desc_ptr[p] .. point_desc_ptr[p+1])
  cv::Mat desc;
  std::vector<int32_t> desc_point;
  std::vector<int32_t> point_desc_ptr, point_desc_rows;
  // keyframes (K)
  std::vector<double> kf_stamp;
  std::vector<Eigen::Matrix4d> kf_T_map_imu;
  std::vector<int32_t> kf_vis_ptr, kf_vis_idx;
  // how the map was built (the localizer must reproduce it exactly)
  Eigen::Matrix4d start_T_map_imu = Eigen::Matrix4d::Identity();
  int view_size = 0;
  double view_fov_deg = 0;
  double orb_scale_factor = 1.2;
  int orb_n_levels = 8, orb_edge_threshold = 31, orb_patch_size = 31, orb_fast_threshold = 10, orb_grid = 8;
  int orb_n_features_map = 0;
  std::vector<std::array<double, 10>> rig_fisheye; ///< per camera: FisheyeCam::model() (fx fy cx cy k1-k4 width height)
  std::vector<Eigen::Matrix4d> rig_T_imu_cam;
  // optional, for display only (the localizer does not load them): each keyframe view as a JPEG of
  // thumb_size^2 px, image k*C + c (keyframe k, camera c) in kf_thumbs[kf_thumbs_ptr[i] .. kf_thumbs_ptr[i+1])
  int thumb_size = 0;
  std::vector<uint8_t> kf_thumbs;
  std::vector<int32_t> kf_thumbs_ptr;
  bool has_T_world_map = false;
  Eigen::Matrix4d T_world_map = Eigen::Matrix4d::Identity(); ///< simulation only, for evaluation
  std::string meta_json;

  /// Throws std::runtime_error on a missing/garbled file. Reads version 1
  /// (map_npz_to_bin.py: ideal fisheyes, f cx cy per camera) and 2 (save()).
  static RuntimeMap load(const std::string &path);
  /// Writes version 2: the full fisheye model per camera (rig_fisheye_kb4).
  void save(const std::string &path) const;

  size_t num_points() const { return points_xyz.size(); }
  size_t num_keyframes() const { return kf_stamp.size(); }
};

} // namespace ov_maploc

#endif
