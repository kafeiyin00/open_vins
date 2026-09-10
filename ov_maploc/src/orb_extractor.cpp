/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#include "orb_extractor.h"

#include <algorithm>

namespace ov_maploc {

OrbExtractor::OrbExtractor(const OrbSpec &spec) : spec_(spec) {
  orb_ = cv::ORB::create(spec.n_features * spec.oversample, (float)spec.scale_factor, spec.n_levels, spec.edge_threshold,
                         0, 2, cv::ORB::HARRIS_SCORE, spec.patch_size, spec.fast_threshold);
}

void OrbExtractor::extract(const cv::Mat &gray, const cv::Mat &mask, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) const {
  kps.clear();
  orb_->detect(gray, kps, mask);
  const int n = spec_.n_features;
  if ((int)kps.size() > n) {
    // spread over a grid: ORB's own top-N by score clusters on the strongest texture
    const int g = spec_.grid;
    std::vector<std::vector<cv::KeyPoint>> cells(g * g);
    for (const auto &k : kps) {
      const int cy = std::min((int)(k.pt.y * g / gray.rows), g - 1);
      const int cx = std::min((int)(k.pt.x * g / gray.cols), g - 1);
      cells[cy * g + cx].push_back(k);
    }
    for (auto &c : cells)
      std::sort(c.begin(), c.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });
    std::vector<cv::KeyPoint> kept;
    kept.reserve(n);
    for (size_t rank = 0; (int)kept.size() < n; rank++) {
      bool added = false;
      for (const auto &c : cells) {
        if (rank < c.size()) {
          kept.push_back(c[rank]);
          added = true;
          if ((int)kept.size() == n)
            break;
        }
      }
      if (!added)
        break;
    }
    kps.swap(kept);
  }
  orb_->compute(gray, kps, desc);
}

} // namespace ov_maploc
