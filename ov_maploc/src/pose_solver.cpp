/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#include "pose_solver.h"

#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace ov_maploc {

namespace {

/// Body-frame bearing and origin of an observation, rotated by the tilt:
/// B = R_tilt * R_imu_cam * K^-1 [u v 1],  O = R_tilt * p_imu_cam.
struct Ray {
  Eigen::Vector3d B, O;
};

/// Linear yaw + translation from >= 3 observations, gravity fixed.
///
/// With R_map_imu = Rz(th) R_tilt and u = Rz(th)^T t, every observation gives
///   Rz(th)^T X - u = O + lambda B  =>  [B]x (A [c s u]^T + d) = 0,
/// A = [[X.x X.y -1 0 0], [X.y -X.x 0 -1 0], [0 0 0 0 -1]], d = (0 0 X.z) - O,
/// linear in (c, s, u) = (cos th, sin th, u).  (c, s) is normalised, then u
/// re-solved with th fixed.
bool solve_linear(const std::vector<Obs> &obs, const std::vector<Ray> &rays, const std::vector<int> &idx,
                  const Eigen::Matrix3d &R_tilt, Eigen::Matrix3d &R, Eigen::Vector3d &t) {
  const int n = (int)idx.size();
  if (n < 3)
    return false;
  Eigen::MatrixXd M(3 * n, 5);
  Eigen::VectorXd rhs(3 * n);
  for (int k = 0; k < n; k++) {
    const Eigen::Vector3d &X = obs[idx[k]].X;
    const Ray &r = rays[idx[k]];
    Eigen::Matrix<double, 3, 5> A;
    A << X.x(), X.y(), -1, 0, 0, X.y(), -X.x(), 0, -1, 0, 0, 0, 0, 0, -1;
    const Eigen::Vector3d d = Eigen::Vector3d(0, 0, X.z()) - r.O;
    const Eigen::Matrix3d S = skew(r.B);
    M.block<3, 5>(3 * k, 0) = S * A;
    rhs.segment<3>(3 * k) = -S * d;
  }
  const Eigen::VectorXd x = M.colPivHouseholderQr().solve(rhs);
  double c = x(0), s = x(1);
  const double nrm = std::hypot(c, s);
  if (!std::isfinite(nrm) || nrm < 1e-9)
    return false;
  c /= nrm;
  s /= nrm;
  const Eigen::VectorXd r2 = rhs - M.leftCols(2) * Eigen::Vector2d(c, s);
  const Eigen::Vector3d u = M.rightCols(3).colPivHouseholderQr().solve(r2);
  if (!u.allFinite())
    return false;
  const Eigen::Matrix3d Rzt = Rz(std::atan2(s, c));
  R = Rzt * R_tilt;
  t = Rzt * u;
  return true;
}

inline bool project(const Obs &o, const RigModel &rig, const Eigen::Matrix3d &Rt, const Eigen::Vector3d &t,
                    Eigen::Vector2d &uv) {
  const Eigen::Vector3d Xb = Rt * (o.X - t); // Rt = R_map_imu^T
  const Eigen::Vector3d Xc = rig.R_imu_cam[o.cam].transpose() * (Xb - rig.p_imu_cam[o.cam]);
  if (Xc.z() < 0.05)
    return false;
  uv = Eigen::Vector2d(rig.f * Xc.x() / Xc.z() + rig.cx, rig.f * Xc.y() / Xc.z() + rig.cy);
  return true;
}

int count_inliers(const std::vector<Obs> &obs, const RigModel &rig, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                  double max_px, std::vector<uint8_t> *mask) {
  const Eigen::Matrix3d Rt = R.transpose();
  const double thr2 = max_px * max_px;
  int n = 0;
  if (mask)
    mask->assign(obs.size(), 0);
  Eigen::Vector2d uv;
  for (size_t i = 0; i < obs.size(); i++) {
    if (project(obs[i], rig, Rt, t, uv) && (uv - obs[i].uv).squaredNorm() < thr2) {
      n++;
      if (mask)
        (*mask)[i] = 1;
    }
  }
  return n;
}

/// 6-DoF Levenberg-Marquardt on the inliers, Huber loss, numeric Jacobian
/// (6 parameters, a few hundred to a few thousand points: cheap).
void refine(const std::vector<Obs> &obs, const std::vector<int> &idx, const RigModel &rig, const PoseOptions &opt,
            Eigen::Matrix3d &R, Eigen::Vector3d &t) {
  const double k = opt.huber_px;
  auto residual = [&](const Eigen::Matrix3d &Rc, const Eigen::Vector3d &tc, int i) -> Eigen::Vector2d {
    Eigen::Vector2d uv;
    if (!project(obs[i], rig, Rc.transpose(), tc, uv))
      return Eigen::Vector2d(1e3, 1e3);
    return uv - obs[i].uv;
  };
  auto cost = [&](const Eigen::Matrix3d &Rc, const Eigen::Vector3d &tc) {
    double c = 0;
    for (int i : idx) {
      const double e = residual(Rc, tc, i).norm();
      c += e <= k ? 0.5 * e * e : k * (e - 0.5 * k);
    }
    return c;
  };
  auto apply = [&](const Eigen::Matrix<double, 6, 1> &d, Eigen::Matrix3d &Ro, Eigen::Vector3d &to) {
    Ro = R * exp_so3(d.head<3>());
    to = t + d.tail<3>();
  };
  double lambda = 1e-3, c0 = cost(R, t);
  const double eps = 1e-6;
  for (int it = 0; it < opt.refine_iters; it++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    for (int i : idx) {
      const Eigen::Vector2d r = residual(R, t, i);
      const double e = r.norm();
      const double w = e <= k ? 1.0 : k / e;
      Eigen::Matrix<double, 2, 6> J;
      for (int j = 0; j < 6; j++) {
        Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix3d Rp, Rm;
        Eigen::Vector3d tp, tm;
        d(j) = eps;
        apply(d, Rp, tp);
        d(j) = -eps;
        apply(d, Rm, tm);
        J.col(j) = (residual(Rp, tp, i) - residual(Rm, tm, i)) / (2 * eps);
      }
      H += w * J.transpose() * J;
      g += w * J.transpose() * r;
    }
    bool improved = false;
    for (int tries = 0; tries < 6 && !improved; tries++) {
      Eigen::Matrix<double, 6, 6> A = H;
      A.diagonal() += lambda * H.diagonal() + Eigen::Matrix<double, 6, 1>::Constant(1e-9);
      const Eigen::Matrix<double, 6, 1> d = A.ldlt().solve(-g);
      Eigen::Matrix3d Rn;
      Eigen::Vector3d tn;
      apply(d, Rn, tn);
      const double c1 = cost(Rn, tn);
      if (std::isfinite(c1) && c1 < c0) {
        const double rel = (c0 - c1) / std::max(c0, 1e-12);
        R = Rn;
        t = tn;
        c0 = c1;
        lambda = std::max(lambda / 10, 1e-9);
        improved = true;
        if (rel < 1e-8)
          return;
      } else {
        lambda *= 10;
      }
    }
    if (!improved)
      return;
  }
}

} // namespace

PoseResult solve_generalized_pose(const std::vector<Obs> &obs, const RigModel &rig, const Eigen::Matrix3d &R_tilt,
                                  const PoseOptions &opt) {
  PoseResult res;
  const int n = (int)obs.size();
  if (n < 3)
    return res;
  std::vector<Ray> rays(n);
  for (int i = 0; i < n; i++) {
    const Eigen::Vector3d rc((obs[i].uv.x() - rig.cx) / rig.f, (obs[i].uv.y() - rig.cy) / rig.f, 1.0);
    rays[i].B = (R_tilt * rig.R_imu_cam[obs[i].cam] * rc).normalized();
    rays[i].O = R_tilt * rig.p_imu_cam[obs[i].cam];
  }

  std::mt19937 rng(opt.seed);
  std::uniform_int_distribution<int> pick(0, n - 1);
  int best = 0;
  Eigen::Matrix3d R_best;
  Eigen::Vector3d t_best;
  int needed = opt.max_trials;
  int trial = 0;
  for (; trial < std::min(needed, opt.max_trials); trial++) {
    std::vector<int> s(3);
    s[0] = pick(rng);
    do {
      s[1] = pick(rng);
    } while (s[1] == s[0]);
    do {
      s[2] = pick(rng);
    } while (s[2] == s[0] || s[2] == s[1]);
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    if (!solve_linear(obs, rays, s, R_tilt, R, t))
      continue;
    const int c = count_inliers(obs, rig, R, t, opt.max_px, nullptr);
    if (c > best) {
      best = c;
      R_best = R;
      t_best = t;
      // adaptive trial count, computed in double: with a poor first hypothesis
      // (w ~ 1/n) it is ~1e9 and overflowed int, which ended RANSAC at once
      const double w = (double)best / n;
      const double denom = std::log(std::max(1e-12, 1.0 - w * w * w));
      const double k = denom < 0 ? std::ceil(std::log(1.0 - opt.confidence) / denom) : (double)opt.max_trials;
      needed = (int)std::min<double>(std::max(k, 1.0), (double)opt.max_trials);
    }
  }
  res.trials = trial;
  if (best < 3)
    return res;

  // local optimisation: re-fit the linear model on the inliers while it grows
  std::vector<uint8_t> mask;
  count_inliers(obs, rig, R_best, t_best, opt.max_px, &mask);
  for (int lo = 0; lo < 3; lo++) {
    std::vector<int> in;
    for (int i = 0; i < n; i++)
      if (mask[i])
        in.push_back(i);
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    if (!solve_linear(obs, rays, in, R_tilt, R, t))
      break;
    std::vector<uint8_t> m2;
    const int c = count_inliers(obs, rig, R, t, opt.max_px, &m2);
    if (c <= best)
      break;
    best = c;
    R_best = R;
    t_best = t;
    mask.swap(m2);
  }

  // 6-DoF refinement on the inliers
  std::vector<int> in;
  for (int i = 0; i < n; i++)
    if (mask[i])
      in.push_back(i);
  Eigen::Matrix3d R = R_best;
  Eigen::Vector3d t = t_best;
  refine(obs, in, rig, opt, R, t);
  std::vector<uint8_t> m2;
  const int c = count_inliers(obs, rig, R, t, opt.max_px, &m2);
  // keep the refined pose unless it lost many inliers: it is more accurate, and
  // only it carries an independent tilt the caller can check against the VIO
  if (c >= 0.95 * best) {
    best = c;
    R_best = R;
    t_best = t;
    mask.swap(m2);
  }
  res.ok = true;
  res.inliers = best;
  res.inlier = mask;
  res.T_map_imu = T_from(R_best, t_best);
  return res;
}

} // namespace ov_maploc
