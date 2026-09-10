/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 *
 * Offline benchmark without ROS: run the localizer over exported query frames
 * and report fix rate, accuracy (when ground truth is given) and timing.
 * Meant for measuring the RK3588.
 *
 *   maploc_bench --map map.bin --calib kalibr_imucam_chain.yaml --masks masks/ --frames frames/
 *                [--orb-n 1000] [--threads 1] [--repeat 1]
 *
 * frames/index.txt:
 *   start tx ty tz qx qy qz qw                       first VIO pose of the flight (known start)
 *   frame i stamp tx ty tz qx qy qz qw gx gy gz      VIO pose T_odom_imu at the frame, ground-truth
 *                                                    IMU position in the map frame (nan if unknown)
 * frames/cam<c>_<i:05d>.png: the fisheye images, 8-bit gray.
 */
#include "geometry.h"
#include "localizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <sstream>

using namespace ov_maploc;

namespace {

struct Frame {
  int idx = 0;
  double stamp = 0;
  Eigen::Matrix4d T_odom_imu;
  Eigen::Vector3d gt;
};

Eigen::Matrix4d pose7(const double *v) {
  const Eigen::Quaterniond q(v[6], v[3], v[4], v[5]);
  return T_from(q.normalized().toRotationMatrix(), Eigen::Vector3d(v[0], v[1], v[2]));
}

double median(std::vector<double> v) {
  if (v.empty())
    return NAN;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

double pct(std::vector<double> v, double p) {
  if (v.empty())
    return NAN;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(p / 100.0 * (v.size() - 1) + 0.5))];
}

} // namespace

int main(int argc, char **argv) {
  std::map<std::string, std::string> a{{"--orb-n", "1000"}, {"--threads", "1"}, {"--repeat", "1"}, {"--masks", ""}};
  for (int i = 1; i + 1 < argc; i += 2)
    a[argv[i]] = argv[i + 1];
  for (const char *k : {"--map", "--calib", "--frames"})
    if (!a.count(k)) {
      std::fprintf(stderr, "usage: %s --map map.bin --calib kalibr.yaml --masks dir --frames dir [--orb-n N] [--threads N] [--repeat N]\n",
                   argv[0]);
      return 2;
    }
  try {
    auto map = std::make_shared<RuntimeMap>(RuntimeMap::load(a["--map"]));
    const auto cams = load_ring(a["--calib"], a["--masks"]);
    LocParams p;
    p.orb_n = std::stoi(a["--orb-n"]);
    p.threads = std::stoi(a["--threads"]);
    const int repeat = std::max(1, std::stoi(a["--repeat"]));
    std::printf("map: %zu landmarks, %d descriptors, %zu keyframes\n", map->num_points(), map->desc.rows, map->num_keyframes());

    std::ifstream idx(a["--frames"] + "/index.txt");
    if (!idx)
      throw std::runtime_error("cannot open " + a["--frames"] + "/index.txt");
    Eigen::Matrix4d T_first = Eigen::Matrix4d::Identity();
    bool have_start = false;
    std::vector<Frame> frames;
    std::string line;
    while (std::getline(idx, line)) {
      std::istringstream ss(line);
      std::string tag;
      ss >> tag;
      double v[12];
      if (tag == "start") {
        for (int k = 0; k < 7; k++)
          ss >> v[k];
        T_first = pose7(v);
        have_start = true;
      } else if (tag == "frame") {
        Frame f;
        ss >> f.idx >> f.stamp;
        for (int k = 0; k < 10; k++) {
          std::string s;
          ss >> s;
          v[k] = (s == "nan") ? NAN : std::stod(s);
        }
        f.T_odom_imu = pose7(v);
        f.gt = Eigen::Vector3d(v[7], v[8], v[9]);
        frames.push_back(f);
      }
    }
    if (!have_start || frames.empty())
      throw std::runtime_error("index.txt needs a 'start' line and 'frame' lines");

    std::vector<double> t_tot, t_orb, t_match, t_pnp, fix_err, cor_err, raw_err;
    std::map<std::string, int> why;
    int n_ok = 0, n_try = 0;
    for (int rep = 0; rep < repeat; rep++) {
      Localizer loc(map, cams, p);
      const Eigen::Matrix4d T_mo0 = to_4dof(map->start_T_map_imu * T_inv(T_first));
      loc.set_start(T_mo0);
      for (const auto &f : frames) {
        std::vector<cv::Mat> grays;
        for (size_t c = 0; c < cams.size(); c++) {
          char name[64];
          std::snprintf(name, sizeof(name), "/cam%zu_%05d.png", c, f.idx);
          grays.push_back(cv::imread(a["--frames"] + name, cv::IMREAD_GRAYSCALE));
          if (grays.back().empty())
            throw std::runtime_error(std::string("missing ") + name);
        }
        const LocResult r = loc.localize(grays, f.T_odom_imu);
        n_try++;
        t_tot.push_back(r.t_total * 1e3);
        t_orb.push_back(r.t_orb * 1e3);
        t_match.push_back(r.t_match * 1e3);
        t_pnp.push_back(r.t_pnp * 1e3);
        if (r.ok) {
          n_ok++;
        } else {
          why[r.why]++;
        }
        if (rep == 0 && std::isfinite(f.gt.x())) {
          if (r.ok)
            fix_err.push_back((r.T_meas.block<3, 1>(0, 3) - f.gt).norm());
          cor_err.push_back(((loc.T_map_odom() * f.T_odom_imu).block<3, 1>(0, 3) - f.gt).norm());
          raw_err.push_back(((T_mo0 * f.T_odom_imu).block<3, 1>(0, 3) - f.gt).norm());
        }
        if (rep == 0)
          std::printf("frame %4d t=%8.2f ok=%d inl=%4d corr=%4d lm=%5d tilt=%.2f ms=%6.1f %s\n", f.idx, f.stamp, (int)r.ok,
                      r.inliers, r.n_corr, r.n_landmarks, r.tilt_deg, r.t_total * 1e3, r.why.c_str());
      }
    }
    std::printf("\nfixes %d/%d  rejections:", n_ok, n_try);
    for (const auto &kv : why)
      std::printf(" [%s: %d]", kv.first.c_str(), kv.second);
    std::printf("\ntiming ms (median / p95): total %.1f / %.1f  orb %.1f  match %.1f  pnp %.1f  (threads %d)\n", median(t_tot),
                pct(t_tot, 95), median(t_orb), median(t_match), median(t_pnp), p.threads);
    if (!cor_err.empty())
      std::printf("error vs ground truth, m: fix median %.3f p95 %.3f | corrected median %.3f | known start only median %.3f\n",
                  median(fix_err), pct(fix_err, 95), median(cor_err), median(raw_err));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
