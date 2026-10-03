/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 *
 * Loop closure for maploc_build: place recognition with a bag of words trained
 * on the run's own ORB descriptors (no vocabulary file), and a 4-DoF pose
 * graph (x, y, z, yaw; roll and pitch stay the VIO's, which gravity makes
 * observable) in the manner of VINS-Mono.
 */
#ifndef OV_MAPLOC_LOOP_CLOSURE_H
#define OV_MAPLOC_LOOP_CLOSURE_H

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace ov_maploc {

using Bits256 = std::array<uint64_t, 4>; ///< one ORB descriptor

inline Bits256 bits_of(const uint8_t *d) {
  Bits256 b;
  for (int i = 0; i < 4; i++) {
    uint64_t v = 0;
    for (int k = 0; k < 8; k++)
      v |= (uint64_t)d[8 * i + k] << (8 * k);
    b[i] = v;
  }
  return b;
}

inline int hamming(const Bits256 &a, const Bits256 &b) {
  return __builtin_popcountll(a[0] ^ b[0]) + __builtin_popcountll(a[1] ^ b[1]) + __builtin_popcountll(a[2] ^ b[2]) +
         __builtin_popcountll(a[3] ^ b[3]);
}

/// A flat visual vocabulary: k binary words from k-majority clustering.
class Vocabulary {
public:
  /// Clusters (a sample of) the descriptors into k words, iters rounds.
  void train(const std::vector<Bits256> &sample, int k, int iters, int threads, uint32_t seed = 7);
  int word(const Bits256 &d) const;
  int size() const { return (int)words_.size(); }

private:
  std::vector<Bits256> words_;
};

/// A TF-IDF bag of words, L1-normalized, sorted by word.
using BowVector = std::vector<std::pair<int, float>>;

/// Documents (each a list of word ids) -> their BoW vectors, with the IDF of this set of documents.
std::vector<BowVector> bow_vectors(const std::vector<std::vector<int>> &docs, int n_words);

/// DBoW's L1 score in [0, 1]: 1 - |a - b|_1 / 2.
double bow_score(const BowVector &a, const BowVector &b);

/// One constraint of the 4-DoF pose graph: node b relative to node a (b's position in a's frame, b's yaw
/// minus a's), weighted by sigma_t (m) and sigma_yaw (rad).
struct PoseEdge {
  int a = 0, b = 0;
  Eigen::Vector3d t_ab = Eigen::Vector3d::Zero();
  double dyaw = 0;
  double sigma_t = 0.05, sigma_yaw = 0.01;
  bool loop = false;
  double w = 1.0; ///< weight of the squared residual (0: left out)
};

/// The 4-DoF relative measurement of b from a, from their full poses.
PoseEdge relative_4dof(const Eigen::Matrix4d &T_a, const Eigen::Matrix4d &T_b, int a, int b);

/// Optimizes the poses' x, y, z and yaw over the edges, node 0 fixed, roll and pitch kept. Returns the
/// edges' residuals after the optimization (metres, and radians for the yaw). No robust loss: a loop
/// far off the VIO would hardly pull under one, so the caller drops the loops that disagree instead.
std::vector<std::pair<double, double>> optimize_pose_graph(std::vector<Eigen::Matrix4d> &T, const std::vector<PoseEdge> &edges,
                                                           int threads, int iters = 100);

/// The squared residual of an edge at these poses, normalized by its sigmas.
double edge_r2(const std::vector<Eigen::Matrix4d> &T, const PoseEdge &e);

/// Robust pose graph: the odometry edges as they are, the loop edges by GNC-TLS (Yang, Antonante, Tzoumas,
/// Carlone: "Graduated Non-Convexity for Robust Spatial Perception", RA-L 2020) -- least squares first, then
/// the truncated least squares cost graduated from convex to its own shape, each loop re-weighted between
/// solves; a loop whose normalized residual stays beyond c_bar ends at weight 0. T: the start, then the
/// result with the loops of weight 1. Returns each loop's weight (0 or 1).
std::vector<double> robust_pose_graph(std::vector<Eigen::Matrix4d> &T, const std::vector<PoseEdge> &odo, const std::vector<PoseEdge> &loops,
                                      int threads, double c_bar);

} // namespace ov_maploc

#endif
