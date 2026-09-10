/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#ifndef OV_MAPLOC_GEOMETRY_H
#define OV_MAPLOC_GEOMETRY_H

#include <Eigen/Dense>
#include <cmath>

namespace ov_maploc {

inline Eigen::Matrix3d Rz(double yaw) {
  const double c = std::cos(yaw), s = std::sin(yaw);
  Eigen::Matrix3d R;
  R << c, -s, 0, s, c, 0, 0, 0, 1;
  return R;
}

inline double yaw_of(const Eigen::Matrix3d &R) { return std::atan2(R(1, 0), R(0, 0)); }

inline Eigen::Matrix4d T_from(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;
  return T;
}

inline Eigen::Matrix4d T_inv(const Eigen::Matrix4d &T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  return T_from(R.transpose(), -R.transpose() * T.block<3, 1>(0, 3));
}

/// Keep x, y, z and yaw: both OpenVINS' frame and the map are gravity aligned.
inline Eigen::Matrix4d to_4dof(const Eigen::Matrix4d &T) {
  return T_from(Rz(yaw_of(T.block<3, 3>(0, 0))), T.block<3, 1>(0, 3));
}

/// Angle between the z axes (gravity direction) of two orientations, degrees.
inline double tilt_deg(const Eigen::Matrix3d &Ra, const Eigen::Matrix3d &Rb) {
  const double c = std::max(-1.0, std::min(1.0, Ra.col(2).dot(Rb.col(2))));
  return std::acos(c) * 180.0 / M_PI;
}

inline double wrap_rad(double a) { return std::atan2(std::sin(a), std::cos(a)); }

inline Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

/// Rotation vector -> matrix (Rodrigues).
inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d &w) {
  const double th = w.norm();
  if (th < 1e-12)
    return Eigen::Matrix3d::Identity() + skew(w);
  return Eigen::AngleAxisd(th, w / th).toRotationMatrix();
}

} // namespace ov_maploc

#endif
