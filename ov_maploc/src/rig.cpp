/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#include "rig.h"

#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace ov_maploc {

std::vector<FisheyeCam> load_ring(const std::string &kalibr_yaml, const std::string &mask_dir) {
  cv::FileStorage fs(kalibr_yaml, cv::FileStorage::READ);
  if (!fs.isOpened())
    throw std::runtime_error("rig: cannot open " + kalibr_yaml);
  std::vector<FisheyeCam> cams;
  for (int i = 0;; i++) {
    cv::FileNode n = fs["cam" + std::to_string(i)];
    if (n.empty())
      break;
    FisheyeCam c;
    c.name = "cam" + std::to_string(i);
    std::string model;
    n["distortion_model"] >> model;
    std::vector<double> intr, dist;
    std::vector<int> res;
    n["intrinsics"] >> intr;
    n["distortion_coeffs"] >> dist;
    n["resolution"] >> res;
    n["rostopic"] >> c.topic;
    if (model != "equidistant" || intr.size() != 4 || res.size() != 2)
      throw std::runtime_error("rig: " + c.name + " must be a pure equidistant fisheye");
    for (double k : dist)
      if (k != 0.0)
        throw std::runtime_error("rig: " + c.name + " has distortion coefficients, only r = f*theta is supported");
    if (std::abs(intr[0] - intr[1]) > 1e-9)
      throw std::runtime_error("rig: " + c.name + " fu != fv");
    c.f = intr[0];
    c.cx = intr[2];
    c.cy = intr[3];
    c.width = res[0];
    c.height = res[1];
    cv::FileNode T = n["T_imu_cam"];
    int r = 0;
    for (auto it = T.begin(); it != T.end() && r < 4; ++it, ++r) {
      std::vector<double> row;
      (*it) >> row;
      if (row.size() != 4)
        throw std::runtime_error("rig: " + c.name + " T_imu_cam is not 4x4");
      for (int k = 0; k < 4; k++)
        c.T_imu_cam(r, k) = row[k];
    }
    if (r != 4)
      throw std::runtime_error("rig: " + c.name + " T_imu_cam is not 4x4");
    if (!mask_dir.empty()) {
      c.mask = cv::imread(mask_dir + "/" + c.name + ".png", cv::IMREAD_GRAYSCALE);
      if (c.mask.empty() || c.mask.cols != c.width || c.mask.rows != c.height)
        throw std::runtime_error("rig: bad mask " + mask_dir + "/" + c.name + ".png");
    }
    cams.push_back(c);
  }
  if (cams.empty())
    throw std::runtime_error("rig: no cameras in " + kalibr_yaml);
  return cams;
}

VirtualView::VirtualView(const FisheyeCam &cam, int n, double fov_deg) : size(n), T_imu_cam(cam.T_imu_cam) {
  f = (n / 2.0) / std::tan(fov_deg * M_PI / 360.0);
  const double c = n / 2.0; // COLMAP convention, pixel centres at +0.5
  cx_cv = c - 0.5;
  mapx_.create(n, n, CV_32F);
  mapy_.create(n, n, CV_32F);
  cv::Mat usable(n, n, CV_8U);
  for (int v = 0; v < n; v++) {
    for (int u = 0; u < n; u++) {
      const double x = (u + 0.5 - c) / f, y = (v + 0.5 - c) / f;
      const double rho = std::hypot(x, y);
      const double scale = rho > 1e-12 ? cam.f * std::atan(rho) / rho : cam.f;
      const float mx = (float)(cam.cx + x * scale), my = (float)(cam.cy + y * scale);
      mapx_.at<float>(v, u) = mx;
      mapy_.at<float>(v, u) = my;
      usable.at<uint8_t>(v, u) = (mx >= 0 && mx <= cam.width - 1 && my >= 0 && my <= cam.height - 1) ? 255 : 0;
    }
  }
  if (!cam.mask.empty()) {
    cv::Mat fish_ok = (cam.mask < 128);
    cv::Mat ok;
    cv::remap(fish_ok, ok, mapx_, mapy_, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
    usable &= (ok > 0);
  }
  // keep ORB's 31 px patch off the airframe edge
  cv::erode(usable, mask, cv::Mat::ones(9, 9, CV_8U));
}

cv::Mat VirtualView::render(const cv::Mat &gray) const {
  cv::Mat out;
  cv::remap(gray, out, mapx_, mapy_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
  return out;
}

} // namespace ov_maploc
