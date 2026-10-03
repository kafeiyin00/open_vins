/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#include "loop_closure.h"

#include "geometry.h"

#include <algorithm>
#include <atomic>
#include <ceres/ceres.h>
#include <cmath>
#include <map>
#include <random>
#include <thread>

namespace ov_maploc {

namespace {

template <class F> void par_for(int n, int threads, F f) {
  std::atomic<int> next{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < std::max(1, threads); t++)
    ts.emplace_back([&] {
      for (int i; (i = next++) < n;)
        f(i);
    });
  for (auto &t : ts)
    t.join();
}

/// ZYX Euler angles of R = Rz(yaw) Ry(pitch) Rx(roll)
void ypr(const Eigen::Matrix3d &R, double &yaw, double &pitch, double &roll) {
  yaw = std::atan2(R(1, 0), R(0, 0));
  pitch = std::atan2(-R(2, 0), std::hypot(R(2, 1), R(2, 2)));
  roll = std::atan2(R(2, 1), R(2, 2));
}

Eigen::Matrix3d R_ypr(double yaw, double pitch, double roll) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

/// VINS-Mono's 4-DoF error: r_t = R(yaw_a, pitch_a, roll_a)^T (t_b - t_a) - t_ab, r_yaw = yaw_b - yaw_a - dyaw.
struct FourDofCost {
  FourDofCost(const PoseEdge &e, double pitch_a, double roll_a) : e(e) {
    // R(yaw_a, pitch_a, roll_a)^T = Rx(roll)^T Ry(pitch)^T Rz(yaw_a)^T: the constant part
    P = (Eigen::AngleAxisd(pitch_a, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(roll_a, Eigen::Vector3d::UnitX())).toRotationMatrix().transpose();
  }
  template <class T> bool operator()(const T *ta, const T *ya, const T *tb, const T *yb, T *r) const {
    const T d[3] = {tb[0] - ta[0], tb[1] - ta[1], tb[2] - ta[2]};
    const T c = cos(ya[0]), s = sin(ya[0]);
    const T dz[3] = {c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]}; // Rz(yaw_a)^T d
    for (int i = 0; i < 3; i++)
      r[i] = (T(P(i, 0)) * dz[0] + T(P(i, 1)) * dz[1] + T(P(i, 2)) * dz[2] - T(e.t_ab(i))) / T(e.sigma_t);
    T dy = yb[0] - ya[0] - T(e.dyaw);
    dy = atan2(sin(dy), cos(dy));
    r[3] = dy / T(e.sigma_yaw);
    return true;
  }
  PoseEdge e;
  Eigen::Matrix3d P;
};

} // namespace

void Vocabulary::train(const std::vector<Bits256> &sample, int k, int iters, int threads, uint32_t seed) {
  words_.clear();
  if (sample.empty())
    return;
  k = std::min<int>(k, (int)sample.size());
  std::mt19937 rng(seed);
  std::vector<int> idx(sample.size());
  for (size_t i = 0; i < idx.size(); i++)
    idx[i] = (int)i;
  std::shuffle(idx.begin(), idx.end(), rng);
  for (int i = 0; i < k; i++)
    words_.push_back(sample[idx[i]]);
  std::vector<int> assign(sample.size(), 0);
  for (int it = 0; it < iters; it++) {
    par_for((int)sample.size(), threads, [&](int i) { assign[i] = word(sample[i]); });
    // each word becomes the bitwise majority of its members
    std::vector<std::array<int, 256>> ones(k);
    std::vector<int> n(k, 0);
    for (auto &o : ones)
      o.fill(0);
    for (size_t i = 0; i < sample.size(); i++) {
      auto &o = ones[assign[i]];
      n[assign[i]]++;
      for (int b = 0; b < 256; b++)
        o[b] += (int)((sample[i][b >> 6] >> (b & 63)) & 1);
    }
    std::uniform_int_distribution<size_t> any(0, sample.size() - 1);
    for (int w = 0; w < k; w++) {
      if (!n[w]) { // an empty word restarts from a random descriptor
        words_[w] = sample[any(rng)];
        continue;
      }
      Bits256 c{{0, 0, 0, 0}};
      for (int b = 0; b < 256; b++)
        if (2 * ones[w][b] > n[w])
          c[b >> 6] |= (uint64_t)1 << (b & 63);
      words_[w] = c;
    }
  }
}

int Vocabulary::word(const Bits256 &d) const {
  int best = 0, bd = 1 << 30;
  for (size_t w = 0; w < words_.size(); w++) {
    const int h = hamming(d, words_[w]);
    if (h < bd) {
      bd = h;
      best = (int)w;
    }
  }
  return best;
}

std::vector<BowVector> bow_vectors(const std::vector<std::vector<int>> &docs, int n_words) {
  std::vector<int> df(n_words, 0);
  for (const auto &d : docs) {
    std::vector<int> u(d);
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    for (int w : u)
      df[w]++;
  }
  std::vector<BowVector> out(docs.size());
  for (size_t i = 0; i < docs.size(); i++) {
    std::map<int, float> tf;
    for (int w : docs[i])
      tf[w] += 1.0f;
    double sum = 0;
    for (auto &kv : tf) {
      kv.second *= (float)std::log((double)docs.size() / std::max(1, df[kv.first]));
      sum += kv.second;
    }
    for (const auto &kv : tf)
      if (kv.second > 0)
        out[i].emplace_back(kv.first, (float)(kv.second / std::max(sum, 1e-12)));
  }
  return out;
}

double bow_score(const BowVector &a, const BowVector &b) {
  // |a - b|_1 = |a|_1 + |b|_1 + sum over common words (|a_w - b_w| - a_w - b_w)
  double l1 = 2.0;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i].first < b[j].first)
      i++;
    else if (b[j].first < a[i].first)
      j++;
    else {
      l1 += std::abs(a[i].second - b[j].second) - a[i].second - b[j].second;
      i++;
      j++;
    }
  }
  return 1.0 - 0.5 * l1;
}

PoseEdge relative_4dof(const Eigen::Matrix4d &T_a, const Eigen::Matrix4d &T_b, int a, int b) {
  PoseEdge e;
  e.a = a;
  e.b = b;
  e.t_ab = T_a.block<3, 3>(0, 0).transpose() * (T_b.block<3, 1>(0, 3) - T_a.block<3, 1>(0, 3));
  e.dyaw = wrap_rad(yaw_of(T_b.block<3, 3>(0, 0)) - yaw_of(T_a.block<3, 3>(0, 0)));
  return e;
}

std::vector<std::pair<double, double>> optimize_pose_graph(std::vector<Eigen::Matrix4d> &T, const std::vector<PoseEdge> &edges,
                                                           int threads, int iters) {
  const int n = (int)T.size();
  std::vector<std::array<double, 3>> t(n);
  std::vector<double> yaw(n), pitch(n), roll(n);
  for (int k = 0; k < n; k++) {
    ypr(T[k].block<3, 3>(0, 0), yaw[k], pitch[k], roll[k]);
    for (int i = 0; i < 3; i++)
      t[k][i] = T[k](i, 3);
  }
  ceres::Problem problem;
  for (const auto &e : edges)
    if (e.w > 0)
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<FourDofCost, 4, 3, 1, 3, 1>(new FourDofCost(e, pitch[e.a], roll[e.a])),
                               e.w == 1.0 ? nullptr : new ceres::ScaledLoss(nullptr, e.w, ceres::TAKE_OWNERSHIP), t[e.a].data(), &yaw[e.a],
                               t[e.b].data(), &yaw[e.b]);
  if (problem.HasParameterBlock(t[0].data())) {
    problem.SetParameterBlockConstant(t[0].data());
    problem.SetParameterBlockConstant(&yaw[0]);
  }
  ceres::Solver::Options opt;
  opt.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  opt.max_num_iterations = iters;
  opt.num_threads = threads;
  ceres::Solver::Summary sum;
  ceres::Solve(opt, &problem, &sum);
  for (int k = 0; k < n; k++)
    T[k] = T_from(R_ypr(yaw[k], pitch[k], roll[k]), Eigen::Vector3d(t[k][0], t[k][1], t[k][2]));
  std::vector<std::pair<double, double>> res;
  for (const auto &e : edges) {
    const PoseEdge now = relative_4dof(T[e.a], T[e.b], e.a, e.b);
    res.emplace_back((now.t_ab - e.t_ab).norm(), std::abs(wrap_rad(now.dyaw - e.dyaw)));
  }
  return res;
}

double edge_r2(const std::vector<Eigen::Matrix4d> &T, const PoseEdge &e) {
  const PoseEdge now = relative_4dof(T[e.a], T[e.b], e.a, e.b);
  const double rt = (now.t_ab - e.t_ab).norm() / e.sigma_t, ry = wrap_rad(now.dyaw - e.dyaw) / e.sigma_yaw;
  return rt * rt + ry * ry;
}

std::vector<double> robust_pose_graph(std::vector<Eigen::Matrix4d> &T, const std::vector<PoseEdge> &odo, const std::vector<PoseEdge> &loops,
                                      int threads, double c_bar) {
  std::vector<double> w(loops.size(), 1.0), r2(loops.size(), 0.0);
  auto solve = [&] {
    std::vector<PoseEdge> edges = odo;
    for (size_t l = 0; l < loops.size(); l++) {
      edges.push_back(loops[l]);
      edges.back().w = w[l];
    }
    optimize_pose_graph(T, edges, threads);
    for (size_t l = 0; l < loops.size(); l++)
      r2[l] = edge_r2(T, loops[l]);
  };
  solve(); // least squares: every loop at full weight
  const double c2 = c_bar * c_bar;
  const double r2_max = loops.empty() ? 0.0 : *std::max_element(r2.begin(), r2.end());
  if (r2_max <= c2)
    return w;
  // TLS weights for the surrogate with parameter mu (convex at mu -> 0, TLS at mu -> inf)
  for (double mu = c2 / (2 * r2_max - c2); mu < 1e4; mu *= 1.4) {
    double unsure = 0;
    for (size_t l = 0; l < loops.size(); l++) {
      if (r2[l] >= (mu + 1) / mu * c2)
        w[l] = 0;
      else if (r2[l] <= mu / (mu + 1) * c2)
        w[l] = 1;
      else
        w[l] = c_bar * std::sqrt(mu * (mu + 1) / r2[l]) - mu;
      unsure += w[l] * (1 - w[l]);
    }
    solve();
    if (unsure < 1e-6)
      break;
  }
  for (double &x : w)
    x = x > 0.5 ? 1.0 : 0.0;
  solve();
  return w;
}

} // namespace ov_maploc
