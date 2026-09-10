/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#ifndef OV_MAPLOC_ORB_EXTRACTOR_H
#define OV_MAPLOC_ORB_EXTRACTOR_H

#include <opencv2/features2d.hpp>
#include <vector>

namespace ov_maploc {

/// ORB as the map builder runs it (maploc/features.py).  Only n_features may
/// differ from the map's: it changes how many keypoints are kept, not what a
/// descriptor looks like.
struct OrbSpec {
  int n_features = 1000;
  double scale_factor = 1.2;
  int n_levels = 8, edge_threshold = 31, patch_size = 31, fast_threshold = 10;
  int grid = 8;       ///< grid x grid cells to spread keypoints
  int oversample = 3; ///< detect oversample * n_features, then bucket
};

/// Not thread safe: one extractor per thread.
class OrbExtractor {
public:
  explicit OrbExtractor(const OrbSpec &spec);
  void extract(const cv::Mat &gray, const cv::Mat &mask, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) const;

private:
  OrbSpec spec_;
  cv::Ptr<cv::ORB> orb_;
};

} // namespace ov_maploc

#endif
