/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#ifndef OV_MAPLOC_RIG_H
#define OV_MAPLOC_RIG_H

#include <Eigen/Dense>
#include <array>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ov_maploc {

/// One fisheye in OpenVINS' equidistant model (Kannala-Brandt, Kalibr's
/// pinhole-equi), OpenCV pixel convention: theta = angle off the optical axis,
/// theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8),
/// u = cx + fx theta_d x / r, v = cy + fy theta_d y / r. k = 0, fx = fy is the
/// ideal r = f theta of the simulated rigs.
struct FisheyeCam {
  std::string name, topic;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  std::array<double, 4> k{{0, 0, 0, 0}};
  int width = 0, height = 0;
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity(); ///< optical frame in the IMU frame
  cv::Mat mask;                                             ///< uint8, 255 = do not use (OpenVINS convention)

  /// fx fy cx cy k1 k2 k3 k4 width height: what a map stores and is checked against
  std::array<double, 10> model() const { return {fx, fy, cx, cy, k[0], k[1], k[2], k[3], (double)width, (double)height}; }
};

/// Reads the same kalibr imu-camera chain OpenVINS reads (equidistant fisheyes)
/// and, when mask_dir is not empty, each camera's mask: <mask_dir>/camN.png, or
/// the OpenVINS config's own mask_<label>.png (label from the topic, /cn2/... -> cn2).
std::vector<FisheyeCam> load_ring(const std::string &kalibr_yaml, const std::string &mask_dir);

/// Pinhole view rendered from one fisheye with the same optical axis. The map
/// builder (maploc_build) and the localizer both render with this class.
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
