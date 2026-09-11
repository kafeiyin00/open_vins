/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#include "localizer.h"

#include "geometry.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <future>
#include <stdexcept>
#include <unordered_map>

namespace ov_maploc {

namespace {
double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

Localizer::Localizer(std::shared_ptr<const RuntimeMap> map, const std::vector<FisheyeCam> &cams, const LocParams &p)
    : map_(std::move(map)), p_(p) {
  const RuntimeMap &m = *map_;
  if (cams.size() != m.rig_T_imu_cam.size() || cams.size() != m.rig_fisheye.size())
    throw std::runtime_error("localizer: map has " + std::to_string(m.rig_T_imu_cam.size()) + " cameras, calibration " +
                             std::to_string(cams.size()));
  for (size_t i = 0; i < cams.size(); i++) {
    const auto &f = m.rig_fisheye[i];
    const bool intr_ok = std::abs(f[0] - cams[i].f) < 1e-6 && std::abs(f[1] - cams[i].cx) < 1e-6 &&
                         std::abs(f[2] - cams[i].cy) < 1e-6 && (int)f[3] == cams[i].width && (int)f[4] == cams[i].height;
    if (!intr_ok || (m.rig_T_imu_cam[i] - cams[i].T_imu_cam).cwiseAbs().maxCoeff() > 1e-6)
      throw std::runtime_error("localizer: " + cams[i].name + " calibration differs from the one the map was built with");
  }
  OrbSpec spec;
  spec.n_features = p.orb_n;
  spec.scale_factor = m.orb_scale_factor;
  spec.n_levels = m.orb_n_levels;
  spec.edge_threshold = m.orb_edge_threshold;
  spec.patch_size = m.orb_patch_size;
  spec.fast_threshold = m.orb_fast_threshold;
  spec.grid = m.orb_grid;
  const double corner_deg = std::atan(std::sqrt(2.0) * std::tan(m.view_fov_deg * M_PI / 360.0)) * 180.0 / M_PI;
  for (const auto &c : cams) {
    views_.emplace_back(c, m.view_size, m.view_fov_deg);
    orbs_.emplace_back(spec);
    rig_.R_imu_cam.push_back(c.T_imu_cam.block<3, 3>(0, 0));
    rig_.p_imu_cam.push_back(c.T_imu_cam.block<3, 1>(0, 3));
    cos_gate_.push_back(std::cos(std::min(89.0, corner_deg + p.gate_margin_deg) * M_PI / 180.0));
  }
  rig_.f = views_[0].f;
  rig_.cx = rig_.cy = views_[0].cx_cv;
}

void Localizer::set_start(const Eigen::Matrix4d &T_map_odom) {
  std::lock_guard<std::mutex> lk(mtx_);
  T_map_odom_ = to_4dof(T_map_odom);
  started_ = true;
}

bool Localizer::started() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return started_;
}

Eigen::Matrix4d Localizer::T_map_odom() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return T_map_odom_;
}

LocResult Localizer::miss(LocResult r, const std::string &why) {
  misses_++;
  r.ok = false;
  r.why = why;
  return r;
}

std::vector<Obs> Localizer::match_camera(int ci, const cv::Mat &gray, const Eigen::Matrix4d &T_pred,
                                         const std::vector<int> &lm, double &t_orb, int &n_query) const {
  const RuntimeMap &m = *map_;
  std::vector<Obs> out;
  const double ta = now_s();
  const cv::Mat vg = views_[ci].render(gray);
  std::vector<cv::KeyPoint> kps;
  cv::Mat desc;
  orbs_[ci].extract(vg, views_[ci].mask, kps, desc);
  t_orb = now_s() - ta;
  n_query = (int)kps.size();
  if (kps.size() < 10)
    return out;

  // landmarks in front of this camera at the predicted pose
  const Eigen::Matrix4d T_cam_map = T_inv(T_pred * views_[ci].T_imu_cam);
  const Eigen::Matrix3d R = T_cam_map.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T_cam_map.block<3, 1>(0, 3);
  std::vector<int> rows_lid;
  std::vector<int> rows;
  for (int lid : lm) {
    const Eigen::Vector3d pc = R * m.points_xyz[lid] + t;
    const double rng = pc.norm();
    if (pc.z() > 0.1 && pc.z() / std::max(rng, 1e-9) > cos_gate_[ci]) {
      for (int k = m.point_desc_ptr[lid]; k < m.point_desc_ptr[lid + 1]; k++) {
        rows.push_back(m.point_desc_rows[k]);
        rows_lid.push_back(lid);
      }
    }
  }
  if (rows.size() < 10)
    return out;
  cv::Mat dm((int)rows.size(), 32, CV_8U);
  for (size_t k = 0; k < rows.size(); k++)
    std::memcpy(dm.ptr((int)k), m.desc.ptr(rows[k]), 32);

  cv::BFMatcher bf(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn;
  bf.knnMatch(desc, dm, knn, std::min(4, dm.rows));
  // ratio test against the best match of a DIFFERENT landmark (a landmark has up to 3 descriptors)
  std::unordered_map<int, std::pair<int, float>> best_for_lm;
  for (size_t qi = 0; qi < knn.size(); qi++) {
    const auto &ms = knn[qi];
    if (ms.empty())
      continue;
    const cv::DMatch &b = ms[0];
    const int lid = rows_lid[b.trainIdx];
    float second = 256.f;
    for (size_t j = 1; j < ms.size(); j++) {
      if (rows_lid[ms[j].trainIdx] != lid) {
        second = ms[j].distance;
        break;
      }
    }
    if (b.distance > p_.max_hamming || b.distance >= p_.ratio * second)
      continue;
    auto it = best_for_lm.find(lid);
    if (it == best_for_lm.end() || b.distance < it->second.second)
      best_for_lm[lid] = {(int)qi, b.distance};
  }
  out.reserve(best_for_lm.size());
  for (const auto &kv : best_for_lm) {
    Obs o;
    o.cam = ci;
    o.uv = Eigen::Vector2d(kps[kv.second.first].pt.x, kps[kv.second.first].pt.y);
    o.X = m.points_xyz[kv.first];
    out.push_back(o);
  }
  return out;
}

LocResult Localizer::localize(const std::vector<cv::Mat> &grays, const Eigen::Matrix4d &T_odom_imu, double stamp) {
  const RuntimeMap &m = *map_;
  LocResult r;
  const double t0 = now_s();
  if (!started())
    return miss(r, "no start pose");
  if (grays.size() != views_.size())
    return miss(r, "wrong number of images");
  const Eigen::Matrix4d T_mo = T_map_odom();
  r.T_pred = T_mo * T_odom_imu;
  const Eigen::Vector3d p_pred = r.T_pred.block<3, 1>(0, 3);

  // candidate landmarks: seen by the map keyframes nearest to the prediction
  r.radius = std::min(p_.kf_radius * std::pow(1.5, misses_), p_.kf_radius_max);
  std::vector<std::pair<double, int>> near;
  for (size_t k = 0; k < m.kf_T_map_imu.size(); k++) {
    const double d = (m.kf_T_map_imu[k].block<3, 1>(0, 3) - p_pred).norm();
    if (d <= r.radius)
      near.emplace_back(d, (int)k);
  }
  if (near.size() > (size_t)p_.kf_max) {
    std::partial_sort(near.begin(), near.begin() + p_.kf_max, near.end());
    near.resize(p_.kf_max);
  }
  r.n_kf = (int)near.size();
  if (near.empty())
    return miss(r, "no map keyframe near the prediction");
  std::vector<int> lm;
  for (const auto &kv : near)
    for (int j = m.kf_vis_ptr[kv.second]; j < m.kf_vis_ptr[kv.second + 1]; j++)
      lm.push_back(m.kf_vis_idx[j]);
  std::sort(lm.begin(), lm.end());
  lm.erase(std::unique(lm.begin(), lm.end()), lm.end());
  r.n_landmarks = (int)lm.size();
  const double t1 = now_s();
  r.t_select = t1 - t0;

  // per camera: ORB, frustum gate, matching
  const int C = (int)views_.size();
  std::vector<std::vector<Obs>> per(C);
  std::vector<double> t_orb(C, 0);
  std::vector<int> nq(C, 0);
  if (p_.threads > 1) {
    std::vector<std::future<void>> fut;
    for (int ci = 0; ci < C; ci++)
      fut.push_back(std::async(std::launch::async, [&, ci] { per[ci] = match_camera(ci, grays[ci], r.T_pred, lm, t_orb[ci], nq[ci]); }));
    for (auto &f : fut)
      f.get();
  } else {
    for (int ci = 0; ci < C; ci++)
      per[ci] = match_camera(ci, grays[ci], r.T_pred, lm, t_orb[ci], nq[ci]);
  }
  std::vector<Obs> obs;
  for (int ci = 0; ci < C; ci++) {
    obs.insert(obs.end(), per[ci].begin(), per[ci].end());
    r.t_orb += t_orb[ci];
    r.n_query += nq[ci];
  }
  const double t2 = now_s();
  r.t_match = std::max(0.0, (t2 - t1) - (p_.threads > 1 ? 0.0 : r.t_orb));
  r.n_corr = (int)obs.size();
  if (r.n_corr < p_.min_inliers) {
    r.t_total = now_s() - t0;
    return miss(r, "too few matches");
  }

  // generalized absolute pose: gravity from the VIO for the hypotheses, 6-DoF refinement
  const Eigen::Matrix3d R_pred = r.T_pred.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_tilt = Rz(-yaw_of(R_pred)) * R_pred;
  PoseOptions po;
  po.max_px = p_.ransac_px;
  po.max_trials = p_.ransac_trials;
  const PoseResult pr = solve_generalized_pose(obs, rig_, R_tilt, po);
  const double t3 = now_s();
  r.t_pnp = t3 - t2;
  r.t_total = t3 - t0;
  if (!pr.ok)
    return miss(r, "RANSAC failed");
  r.inliers = pr.inliers;
  r.T_meas = pr.T_map_imu;
  if (r.inliers < p_.min_inliers)
    return miss(r, "too few inliers");
  r.tilt_deg = tilt_deg(r.T_meas.block<3, 3>(0, 0), R_pred);
  if (r.tilt_deg > p_.max_tilt_deg)
    return miss(r, "tilt disagrees with VIO");

  const Eigen::Matrix4d T_mo_meas = to_4dof(r.T_meas * T_inv(T_odom_imu));
  const double yaw_old = yaw_of(T_mo.block<3, 3>(0, 0)), yaw_new = yaw_of(T_mo_meas.block<3, 3>(0, 0));
  const double dyaw = wrap_rad(yaw_new - yaw_old);
  r.jump_m = (T_mo_meas.block<3, 1>(0, 3) - T_mo.block<3, 1>(0, 3)).norm();
  r.jump_deg = std::abs(dyaw) * 180.0 / M_PI;
  // drift budget: the longer since the last accepted fix, the larger the correction the VIO may need
  const double since = (stamp >= 0.0 && last_ok_stamp_ >= 0.0) ? std::max(0.0, stamp - last_ok_stamp_) : 0.0;
  const double gate_m = p_.max_jump_m + p_.jump_rate * since;
  if (locked_ && (r.jump_m > gate_m || r.jump_deg > p_.max_jump_deg + 2.0 * p_.jump_rate * since))
    return miss(r, "jump too large");
  // weak fixes (few inliers) move T_map_odom less
  const double q = std::min(1.0, std::max(0.2, double(r.inliers - p_.min_inliers) / std::max(1, p_.good_inliers - p_.min_inliers)));
  const double a = locked_ ? p_.alpha * q : 1.0;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    T_map_odom_ = T_from(Rz(yaw_old + a * dyaw), (1 - a) * T_mo.block<3, 1>(0, 3) + a * T_mo_meas.block<3, 1>(0, 3));
  }
  locked_ = true;
  misses_ = 0;
  if (stamp >= 0.0)
    last_ok_stamp_ = stamp;
  r.ok = true;
  return r;
}

} // namespace ov_maploc
