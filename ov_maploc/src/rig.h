/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#ifndef OV_MAPLOC_RIG_H
#define OV_MAPLOC_RIG_H

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ov_maploc {

/// One equidistant fisheye (r = f * theta, OpenCV pixel convention as in OpenVINS).
struct FisheyeCam {
  std::string name, topic;
  double f = 0, cx = 0, cy = 0;
  int width = 0, height = 0;
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity(); ///< optical frame in the IMU frame
  cv::Mat mask;                                             ///< uint8, 255 = do not use (OpenVINS convention)
};

/// Reads the same kalibr imu-camera chain OpenVINS reads (pure equidistant
/// fisheyes only) and <mask_dir>/camN.png when mask_dir is not empty.
std::vector<FisheyeCam> load_ring(const std::string &kalibr_yaml, const std::string &mask_dir);

/// Pinhole view rendered from one fisheye with the same optical axis.
/// Must match maploc/rig.py VirtualView of the map builder exactly.
class VirtualView {
public:
  VirtualView(const FisheyeCam &cam, int size, double fov_deg);
  cv::Mat render(const cv::Mat &gray) const;
  int size = 0;
  double f = 0;       ///< px
  double cx_cv = 0;   ///< principal point, OpenCV pixel convention (= size/2 - 0.5)
  cv::Mat mask;       ///< uint8, 255 = usable
  Eigen::Matrix4d T_imu_cam;

private:
  cv::Mat mapx_, mapy_;
};

} // namespace ov_maploc

#endif
