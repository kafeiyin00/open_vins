/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 *
 * maploc_build: an ov_maploc map from the keyframes of a mapping run -- the
 * VIO pose and the fisheye images of each keyframe (ros/pc/web/mapjob.py
 * writes them from a bag). Views and ORB are the localizer's own
 * (VirtualView, OrbExtractor): what is matched online is what was mapped.
 *
 *   maploc_build --keyframes DIR --calib kalibr_imucam_chain.yaml --masks DIR --out map.bin
 *                [--report report.json] [--preview preview.json] [--threads N] [options below]
 *
 * preview.json, for a COLMAP-like view of the result: landmarks [x, y, z, grey] (grey = the
 * views' pixel at the landmark, averaged), keyframes [x, y, z, qw, qx, qy, qz] after the
 * bundle adjustment, vio [x, y, z, yaw] before it, start, the rig (each camera's T_imu_cam),
 * each keyframe's image files and the landmarks it sees.
 *
 * DIR/index.txt:   start tx ty tz qx qy qz qw                 first VIO pose of the run
 *                  kf stamp tx ty tz qx qy qz qw img0 .. imgN  one keyframe: VIO pose, one image per camera
 * Poses as OpenVINS publishes /ov_msckf/poseimu (the IMU in its global frame, the quaternion read as Hamilton).
 *
 * 1. per keyframe and camera: virtual pinhole view, ORB
 * 2. image pairs from the VIO poses: camera centres within --pair-radius, optical axes within
 *    --pair-axis-deg, the --pairs-per-image nearest of each image, whatever the time between
 *    them (a revisit is a loop closure)
 * 3. matching: Hamming, ratio test, mutual best; verified by an essential matrix RANSAC on
 *    the pair alone (robust to VIO drift)
 * 4. tracks (union-find), multi-view triangulation from the VIO poses
 * 5. bundle adjustment (Ceres): keyframe poses and landmarks; consecutive keyframes held to
 *    the VIO's relative motion (scale), every keyframe to the VIO's tilt (gravity), the first
 *    one fixed (the map frame is the mapping run's VIO frame); two rounds, outliers dropped
 *    in between
 * 6. landmarks with up to 3 descriptors and the keyframes' visibility -> map.bin (version 2)
 * 7. self test: the Localizer on keyframes of the run against the new map
 *
 * Progress on stdout as "PROGRESS <stage> <done> <total>" lines.
 */
#include "geometry.h"
#include "localizer.h"
#include "map.h"
#include "orb_extractor.h"
#include "rig.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>

using namespace ov_maploc;

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Args {
  std::map<std::string, std::string> v;
  std::string get(const std::string &k, const std::string &d = "") const {
    auto it = v.find(k);
    return it == v.end() ? d : it->second;
  }
  double num(const std::string &k, double d) const {
    auto it = v.find(k);
    return it == v.end() ? d : std::stod(it->second);
  }
};

struct Params {
  int view_size = 512;
  double view_fov = 120.0;
  int orb_n = 2000;          ///< ORB per view in the map
  double pair_radius = 5.0;  ///< m between camera centres
  double pair_axis_deg = 60; ///< between optical axes
  int pairs_per_image = 12;
  double ratio = 0.8;
  int max_hamming = 64;
  double ransac_px = 2.0;     ///< essential RANSAC threshold, virtual view px
  int min_pair_inliers = 20;
  double max_reproj_px = 3.0;
  double min_angle_deg = 1.5; ///< triangulation angle
  double huber_px = 2.0;
  int ba_iters = 60;
  double rel_sigma_t = 0.02;      ///< m, plus rel_sigma_t_frac of the distance
  double rel_sigma_t_frac = 0.02;
  double rel_sigma_r_deg = 0.3;
  double tilt_sigma_deg = 0.5;
  int threads = 0;
  int selftest = 100; ///< keyframes tried by the self test (0: off)
  int thumb_size = 256; ///< keyframe view thumbnails in the map, px (0: none)
};

struct Keyframe {
  double stamp = 0;
  Eigen::Matrix4d T_vio = Eigen::Matrix4d::Identity(); ///< T_odom_imu from the VIO
  std::vector<std::string> files;
};

struct Image {
  int kf = 0, cam = 0;
  std::vector<cv::KeyPoint> kps;
  cv::Mat desc;
  std::vector<uint8_t> grey; ///< the view's pixel at each keypoint (the preview's point colours)
  std::vector<uchar> thumb;  ///< the view as a small JPEG (map.bin kf_thumbs, for the board's match display)
};

struct Match {
  int a, b; ///< images
  std::vector<std::pair<int, int>> kp;
};

struct Track {
  std::vector<std::pair<int, int>> obs; ///< (image, keypoint)
  Eigen::Vector3d X = Eigen::Vector3d::Zero();
};

Eigen::Matrix4d pose_from(const double *v) { // tx ty tz qx qy qz qw
  const Eigen::Quaterniond q(v[6], v[3], v[4], v[5]);
  return T_from(q.normalized().toRotationMatrix(), Eigen::Vector3d(v[0], v[1], v[2]));
}

template <class F> void parallel_for(int n, int threads, F f) {
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

std::mutex out_mtx;
void progress(const char *stage, int done, int total) {
  std::lock_guard<std::mutex> lk(out_mtx);
  std::printf("PROGRESS %s %d %d\n", stage, done, total);
  std::fflush(stdout);
}

/// A counter that prints PROGRESS about every 2 %.
struct Counter {
  const char *stage;
  int total;
  std::atomic<int> n{0};
  Counter(const char *s, int t) : stage(s), total(t) { progress(stage, 0, total); }
  void tick() {
    const int k = ++n;
    if (k == total || k % std::max(1, total / 50) == 0)
      progress(stage, k, total);
  }
};

double median(std::vector<double> v) {
  if (v.empty())
    return 0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

std::string jnum(double x, int prec = 4) {
  if (!std::isfinite(x))
    return "null";
  char b[64];
  std::snprintf(b, sizeof(b), "%.*f", prec, x);
  return b;
}

// ---- union-find over (image, keypoint) ----
struct DSU {
  std::vector<int> p;
  explicit DSU(size_t n) : p(n) { std::iota(p.begin(), p.end(), 0); }
  int find(int x) {
    while (p[x] != x)
      x = p[x] = p[p[x]];
    return x;
  }
  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b)
      p[std::max(a, b)] = std::min(a, b);
  }
};

// ---- bundle adjustment terms ----
struct ReprojCost {
  ReprojCost(const Eigen::Matrix3d &R_cam_imu, const Eigen::Vector3d &t_cam_imu, double f, double c, const Eigen::Vector2d &uv)
      : R(R_cam_imu), t(t_cam_imu), f(f), c(c), uv(uv) {}
  template <class T> bool operator()(const T *aa, const T *p, const T *X, T *r) const {
    const T d[3] = {X[0] - p[0], X[1] - p[1], X[2] - p[2]};
    const T naa[3] = {-aa[0], -aa[1], -aa[2]};
    T xi[3];
    ceres::AngleAxisRotatePoint(naa, d, xi); // R_map_imu^T (X - p)
    T xc[3];
    for (int i = 0; i < 3; i++)
      xc[i] = T(R(i, 0)) * xi[0] + T(R(i, 1)) * xi[1] + T(R(i, 2)) * xi[2] + T(t(i));
    r[0] = T(f) * xc[0] / xc[2] + T(c) - T(uv.x());
    r[1] = T(f) * xc[1] / xc[2] + T(c) - T(uv.y());
    return true;
  }
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  double f, c;
  Eigen::Vector2d uv;
};

/// Keyframe b relative to keyframe a, as the VIO measured it.
struct RelPoseCost {
  RelPoseCost(const Eigen::Matrix4d &T_ab, double sigma_t, double sigma_r) : st(sigma_t), sr(sigma_r) {
    t = T_ab.block<3, 1>(0, 3);
    const Eigen::Quaterniond q(Eigen::Matrix3d(T_ab.block<3, 3>(0, 0)));
    qm[0] = q.w();
    qm[1] = q.x();
    qm[2] = q.y();
    qm[3] = q.z();
  }
  template <class T> bool operator()(const T *aa_a, const T *p_a, const T *aa_b, const T *p_b, T *r) const {
    const T d[3] = {p_b[0] - p_a[0], p_b[1] - p_a[1], p_b[2] - p_a[2]};
    const T naa[3] = {-aa_a[0], -aa_a[1], -aa_a[2]};
    T tab[3];
    ceres::AngleAxisRotatePoint(naa, d, tab);
    for (int i = 0; i < 3; i++)
      r[i] = (tab[i] - T(t(i))) / T(st);
    T qa[4], qb[4], qai[4], qab[4], qmi[4], qe[4], ae[3];
    ceres::AngleAxisToQuaternion(aa_a, qa);
    ceres::AngleAxisToQuaternion(aa_b, qb);
    qai[0] = qa[0];
    qai[1] = -qa[1];
    qai[2] = -qa[2];
    qai[3] = -qa[3];
    ceres::QuaternionProduct(qai, qb, qab);
    qmi[0] = T(qm[0]);
    qmi[1] = T(-qm[1]);
    qmi[2] = T(-qm[2]);
    qmi[3] = T(-qm[3]);
    ceres::QuaternionProduct(qmi, qab, qe);
    ceres::QuaternionToAngleAxis(qe, ae);
    for (int i = 0; i < 3; i++)
      r[3 + i] = ae[i] / T(sr);
    return true;
  }
  Eigen::Vector3d t;
  double qm[4];
  double st, sr;
};

/// Gravity (the map's z axis) in the IMU frame, as the VIO saw it: the VIO's tilt does not drift.
struct TiltCost {
  TiltCost(const Eigen::Vector3d &g, double sigma) : g(g), s(sigma) {}
  template <class T> bool operator()(const T *aa, T *r) const {
    const T ez[3] = {T(0), T(0), T(1)};
    const T naa[3] = {-aa[0], -aa[1], -aa[2]};
    T gi[3];
    ceres::AngleAxisRotatePoint(naa, ez, gi);
    for (int i = 0; i < 3; i++)
      r[i] = (gi[i] - T(g(i))) / T(s);
    return true;
  }
  Eigen::Vector3d g;
  double s;
};

Eigen::Vector3d aa_of(const Eigen::Matrix3d &R) {
  const Eigen::AngleAxisd a(R);
  return a.angle() * a.axis();
}

} // namespace

int main(int argc, char **argv) {
  Args a;
  for (int i = 1; i + 1 < argc; i += 2)
    a.v[argv[i]] = argv[i + 1];
  const std::string dir = a.get("--keyframes"), calib = a.get("--calib"), masks = a.get("--masks"), out = a.get("--out");
  if (dir.empty() || calib.empty() || out.empty()) {
    std::cerr << "usage: maploc_build --keyframes DIR --calib kalibr_imucam_chain.yaml [--masks DIR] --out map.bin\n"
                 "       [--report report.json] [--preview preview.json] [--threads N] [--view-size 512] [--view-fov 120]\n"
                 "       [--orb-n 2000] [--pair-radius 5] [--pair-axis-deg 60] [--pairs-per-image 12] [--selftest 100]\n";
    return 2;
  }
  Params P;
  P.view_size = (int)a.num("--view-size", P.view_size);
  P.view_fov = a.num("--view-fov", P.view_fov);
  P.orb_n = (int)a.num("--orb-n", P.orb_n);
  P.pair_radius = a.num("--pair-radius", P.pair_radius);
  P.pair_axis_deg = a.num("--pair-axis-deg", P.pair_axis_deg);
  P.pairs_per_image = (int)a.num("--pairs-per-image", P.pairs_per_image);
  P.min_pair_inliers = (int)a.num("--min-pair-inliers", P.min_pair_inliers);
  P.max_reproj_px = a.num("--max-reproj", P.max_reproj_px);
  P.min_angle_deg = a.num("--min-angle-deg", P.min_angle_deg);
  P.ba_iters = (int)a.num("--ba-iters", P.ba_iters);
  P.selftest = (int)a.num("--selftest", P.selftest);
  P.thumb_size = (int)a.num("--thumb-size", P.thumb_size);
  P.threads = (int)a.num("--threads", 0);
  if (P.threads <= 0)
    P.threads = (int)std::max(1u, std::thread::hardware_concurrency());
  const double t_start = now_s();
  std::map<std::string, double> timing;

  try {
    // ---------------------------------------------------------------- inputs
    const auto cams = load_ring(calib, masks);
    const int C = (int)cams.size();
    std::vector<VirtualView> views;
    for (const auto &c : cams)
      views.emplace_back(c, P.view_size, P.view_fov);
    const double vf = views[0].f, vc = views[0].cx_cv;
    OrbSpec spec;
    spec.n_features = P.orb_n;

    std::ifstream idx(dir + "/index.txt");
    if (!idx)
      throw std::runtime_error("cannot open " + dir + "/index.txt");
    Eigen::Matrix4d T_start = Eigen::Matrix4d::Identity();
    bool have_start = false;
    std::vector<Keyframe> kfs;
    for (std::string line; std::getline(idx, line);) {
      std::istringstream ss(line);
      std::string tag;
      if (!(ss >> tag) || tag[0] == '#')
        continue;
      if (tag == "start") {
        double v[7];
        for (double &x : v)
          ss >> x;
        T_start = pose_from(v);
        have_start = true;
      } else if (tag == "kf") {
        Keyframe k;
        double v[7];
        ss >> k.stamp;
        for (double &x : v)
          ss >> x;
        k.T_vio = pose_from(v);
        for (std::string f; ss >> f;)
          k.files.push_back(dir + "/" + f);
        if (!ss.eof() && ss.fail())
          throw std::runtime_error("bad line in index.txt: " + line);
        if ((int)k.files.size() != C)
          throw std::runtime_error("keyframe at " + std::to_string(k.stamp) + " has " + std::to_string(k.files.size()) +
                                   " images, the calibration " + std::to_string(C) + " cameras");
        kfs.push_back(k);
      }
    }
    const int K = (int)kfs.size();
    if (K < 3)
      throw std::runtime_error("need at least 3 keyframes, index.txt has " + std::to_string(K));
    if (!have_start)
      T_start = kfs[0].T_vio;
    std::printf("maploc_build: %d keyframes x %d cameras, views %d px / %.0f deg, ORB %d per view, %d threads\n", K, C, P.view_size,
                P.view_fov, P.orb_n, P.threads);

    // ---------------------------------------------------------------- 1. features
    double t0 = now_s();
    const int I = K * C;
    std::vector<Image> imgs(I);
    std::atomic<int> bad_imgs{0};
    {
      Counter cnt("features", I);
      parallel_for(I, P.threads, [&](int i) {
        Image &im = imgs[i];
        im.kf = i / C;
        im.cam = i % C;
        const cv::Mat g = cv::imread(kfs[im.kf].files[im.cam], cv::IMREAD_GRAYSCALE);
        if (g.empty() || g.cols != cams[im.cam].width || g.rows != cams[im.cam].height) {
          bad_imgs++;
        } else {
          OrbExtractor orb(spec);
          const cv::Mat v = views[im.cam].render(g);
          orb.extract(v, views[im.cam].mask, im.kps, im.desc);
          for (const auto &k : im.kps)
            im.grey.push_back(v.at<uint8_t>(std::min(v.rows - 1, std::max(0, (int)std::lround(k.pt.y))),
                                            std::min(v.cols - 1, std::max(0, (int)std::lround(k.pt.x)))));
          if (P.thumb_size > 0) {
            cv::Mat th;
            cv::resize(v, th, cv::Size(P.thumb_size, P.thumb_size), 0, 0, cv::INTER_AREA);
            cv::imencode(".jpg", th, im.thumb, {cv::IMWRITE_JPEG_QUALITY, 75});
          }
        }
        cnt.tick();
      });
    }
    if (bad_imgs > 0)
      throw std::runtime_error(std::to_string(bad_imgs.load()) + " keyframe images missing or not " + std::to_string(cams[0].width) + "x" +
                               std::to_string(cams[0].height));
    std::vector<int> base(I + 1, 0); // global keypoint id = base[image] + kp
    for (int i = 0; i < I; i++)
      base[i + 1] = base[i] + (int)imgs[i].kps.size();
    timing["features"] = now_s() - t0;
    std::printf("features: %d keypoints, %.0f per view, %.1f s\n", base[I], double(base[I]) / I, timing["features"]);

    // camera poses from the VIO
    auto T_map_cam = [&](const std::vector<Eigen::Matrix4d> &T_kf, int i) { return T_kf[imgs[i].kf] * cams[imgs[i].cam].T_imu_cam; };
    std::vector<Eigen::Matrix4d> T_kf(K);
    for (int k = 0; k < K; k++)
      T_kf[k] = kfs[k].T_vio;

    // ---------------------------------------------------------------- 2. pairs
    t0 = now_s();
    std::vector<Eigen::Vector3d> centre(I), axis(I);
    for (int i = 0; i < I; i++) {
      const Eigen::Matrix4d T = T_map_cam(T_kf, i);
      centre[i] = T.block<3, 1>(0, 3);
      axis[i] = T.block<3, 1>(0, 2);
    }
    const double cos_axis = std::cos(P.pair_axis_deg * M_PI / 180.0);
    std::set<std::pair<int, int>> pairset;
    for (int i = 0; i < I; i++) {
      std::vector<std::pair<double, int>> cand;
      for (int j = 0; j < I; j++) {
        if (imgs[j].kf == imgs[i].kf)
          continue;
        const double d = (centre[i] - centre[j]).norm();
        const double ca = axis[i].dot(axis[j]);
        if (d <= P.pair_radius && ca >= cos_axis)
          cand.emplace_back(d + 0.02 * std::acos(std::min(1.0, ca)) * 180.0 / M_PI, j);
      }
      const size_t n = std::min(cand.size(), (size_t)P.pairs_per_image);
      std::partial_sort(cand.begin(), cand.begin() + n, cand.end());
      for (size_t k = 0; k < n; k++)
        pairset.insert({std::min(i, cand[k].second), std::max(i, cand[k].second)});
    }
    const std::vector<std::pair<int, int>> pairs(pairset.begin(), pairset.end());
    timing["pairs"] = now_s() - t0;
    std::printf("pairs: %zu candidate image pairs\n", pairs.size());

    // ---------------------------------------------------------------- 3. matching
    t0 = now_s();
    std::vector<Match> matches(pairs.size());
    {
      Counter cnt("matching", (int)pairs.size());
      parallel_for((int)pairs.size(), P.threads, [&](int pi) {
        Match &m = matches[pi];
        m.a = pairs[pi].first;
        m.b = pairs[pi].second;
        const Image &A = imgs[m.a], &B = imgs[m.b];
        if (A.kps.size() < 20 || B.kps.size() < 20) {
          cnt.tick();
          return;
        }
        cv::BFMatcher bf(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> ab, ba;
        bf.knnMatch(A.desc, B.desc, ab, 2);
        bf.knnMatch(B.desc, A.desc, ba, 1);
        std::vector<std::pair<int, int>> put;
        for (size_t q = 0; q < ab.size(); q++) {
          if (ab[q].size() < 2)
            continue;
          const cv::DMatch &b0 = ab[q][0];
          if (b0.distance > P.max_hamming || b0.distance >= P.ratio * ab[q][1].distance)
            continue;
          if (ba[b0.trainIdx].empty() || ba[b0.trainIdx][0].trainIdx != (int)q)
            continue;
          put.emplace_back((int)q, b0.trainIdx);
        }
        if ((int)put.size() >= P.min_pair_inliers) {
          std::vector<cv::Point2d> p1, p2;
          for (const auto &kv : put) {
            p1.emplace_back((A.kps[kv.first].pt.x - vc) / vf, (A.kps[kv.first].pt.y - vc) / vf);
            p2.emplace_back((B.kps[kv.second].pt.x - vc) / vf, (B.kps[kv.second].pt.y - vc) / vf);
          }
          cv::Mat mask;
          const cv::Mat E = cv::findEssentialMat(p1, p2, 1.0, cv::Point2d(0, 0), cv::RANSAC, 0.999, P.ransac_px / vf, mask);
          if (!E.empty() && cv::countNonZero(mask) >= P.min_pair_inliers)
            for (size_t k = 0; k < put.size(); k++)
              if (mask.at<uint8_t>((int)k))
                m.kp.push_back(put[k]);
        }
        cnt.tick();
      });
    }
    int pairs_ok = 0, loop_pairs = 0;
    size_t n_match = 0;
    for (const auto &m : matches)
      if (!m.kp.empty()) {
        pairs_ok++;
        n_match += m.kp.size();
        if (std::abs(kfs[imgs[m.a].kf].stamp - kfs[imgs[m.b].kf].stamp) > 30.0)
          loop_pairs++;
      }
    timing["matching"] = now_s() - t0;
    std::printf("matching: %d of %zu pairs verified (%d more than 30 s apart: revisits), %zu matches, %.1f s\n", pairs_ok, pairs.size(),
                loop_pairs, n_match, timing["matching"]);

    // ---------------------------------------------------------------- 4. tracks + triangulation
    t0 = now_s();
    DSU dsu(base[I]);
    for (const auto &m : matches)
      for (const auto &kv : m.kp)
        dsu.unite(base[m.a] + kv.first, base[m.b] + kv.second);
    std::unordered_map<int, std::vector<std::pair<int, int>>> groups;
    for (const auto &m : matches)
      for (const auto &kv : m.kp)
        for (const auto &ik : {std::make_pair(m.a, kv.first), std::make_pair(m.b, kv.second)}) {
          auto &g = groups[dsu.find(base[ik.first] + ik.second)];
          g.push_back(ik);
        }
    std::vector<Track> tracks;
    int conflicts = 0;
    for (auto &kv : groups) {
      auto &o = kv.second;
      std::sort(o.begin(), o.end());
      o.erase(std::unique(o.begin(), o.end()), o.end());
      bool bad = false;
      for (size_t k = 1; k < o.size(); k++)
        bad = bad || o[k].first == o[k - 1].first; // two keypoints of one image
      if (bad) {
        conflicts++;
        continue;
      }
      if (o.size() >= 2)
        tracks.push_back({o, Eigen::Vector3d::Zero()});
    }
    groups.clear();

    // ray of an observation in the map frame, and the reprojection error in the view, px
    auto ray = [&](const std::vector<Eigen::Matrix4d> &Tk, int i, int kp, Eigen::Vector3d &c, Eigen::Vector3d &d) {
      const Eigen::Matrix4d T = T_map_cam(Tk, i);
      const cv::Point2f &pt = imgs[i].kps[kp].pt;
      c = T.block<3, 1>(0, 3);
      d = (T.block<3, 3>(0, 0) * Eigen::Vector3d((pt.x - vc) / vf, (pt.y - vc) / vf, 1.0)).normalized();
    };
    auto reproj = [&](const std::vector<Eigen::Matrix4d> &Tk, int i, int kp, const Eigen::Vector3d &X, double &depth) {
      const Eigen::Matrix4d T = T_inv(T_map_cam(Tk, i));
      const Eigen::Vector3d x = T.block<3, 3>(0, 0) * X + T.block<3, 1>(0, 3);
      depth = x.z();
      if (depth < 1e-6)
        return 1e9;
      const cv::Point2f &pt = imgs[i].kps[kp].pt;
      return std::hypot(vf * x.x() / x.z() + vc - pt.x, vf * x.y() / x.z() + vc - pt.y);
    };
    const double cos_min_angle = std::cos(P.min_angle_deg * M_PI / 180.0);
    // (re)triangulate a track, drop observations beyond max_px, keep it if >= 2 views at a useful angle remain
    auto triangulate = [&](const std::vector<Eigen::Matrix4d> &Tk, Track &t, double max_px, bool solve) {
      for (int round = 0; round < 2; round++) {
        if (t.obs.size() < 2)
          return false;
        std::vector<Eigen::Vector3d> cs(t.obs.size()), ds(t.obs.size());
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        for (size_t k = 0; k < t.obs.size(); k++) {
          ray(Tk, t.obs[k].first, t.obs[k].second, cs[k], ds[k]);
          const Eigen::Matrix3d Pk = Eigen::Matrix3d::Identity() - ds[k] * ds[k].transpose();
          A += Pk;
          b += Pk * cs[k];
        }
        double min_cos = 1.0;
        for (size_t p = 0; p < ds.size(); p++)
          for (size_t q = p + 1; q < ds.size(); q++)
            min_cos = std::min(min_cos, ds[p].dot(ds[q]));
        if (min_cos > cos_min_angle)
          return false;
        if (solve || round > 0)
          t.X = A.ldlt().solve(b);
        std::vector<std::pair<int, int>> keep;
        for (const auto &o : t.obs) {
          double depth;
          if (reproj(Tk, o.first, o.second, t.X, depth) <= max_px && depth > 0.1)
            keep.push_back(o);
        }
        if (keep.size() == t.obs.size())
          return t.obs.size() >= 2;
        t.obs.swap(keep);
        solve = true;
      }
      return false;
    };
    std::vector<uint8_t> keep_t(tracks.size(), 0);
    parallel_for((int)tracks.size(), P.threads, [&](int ti) { keep_t[ti] = triangulate(T_kf, tracks[ti], 2 * P.max_reproj_px, true); });
    {
      std::vector<Track> kept;
      for (size_t ti = 0; ti < tracks.size(); ti++)
        if (keep_t[ti])
          kept.push_back(std::move(tracks[ti]));
      std::printf("tracks: %zu from the matches (%d with conflicting keypoints dropped), %zu triangulated\n", tracks.size(), conflicts,
                  kept.size());
      tracks.swap(kept);
    }
    timing["triangulation"] = now_s() - t0;
    if (tracks.size() < 100)
      throw std::runtime_error("only " + std::to_string(tracks.size()) +
                               " landmarks triangulated: too little overlap between keyframes, or the VIO trajectory is wrong");

    auto reproj_stats = [&](const std::vector<Eigen::Matrix4d> &Tk, double &mean, double &med) {
      std::vector<double> e;
      for (const auto &t : tracks)
        for (const auto &o : t.obs) {
          double depth;
          e.push_back(reproj(Tk, o.first, o.second, t.X, depth));
        }
      mean = e.empty() ? 0 : std::accumulate(e.begin(), e.end(), 0.0) / e.size();
      med = median(e);
    };
    double err0_mean, err0_med;
    reproj_stats(T_kf, err0_mean, err0_med);

    // ---------------------------------------------------------------- 5. bundle adjustment
    t0 = now_s();
    std::vector<std::array<double, 3>> aa(K), pp(K);
    for (int k = 0; k < K; k++) {
      const Eigen::Vector3d r = aa_of(T_kf[k].block<3, 3>(0, 0));
      for (int i = 0; i < 3; i++) {
        aa[k][i] = r(i);
        pp[k][i] = T_kf[k](i, 3);
      }
    }
    std::vector<Eigen::Matrix3d> R_cam_imu(C);
    std::vector<Eigen::Vector3d> t_cam_imu(C);
    for (int c = 0; c < C; c++) {
      const Eigen::Matrix4d T = T_inv(cams[c].T_imu_cam);
      R_cam_imu[c] = T.block<3, 3>(0, 0);
      t_cam_imu[c] = T.block<3, 1>(0, 3);
    }
    auto pull = [&] {
      for (int k = 0; k < K; k++)
        T_kf[k] = T_from(exp_so3(Eigen::Vector3d(aa[k][0], aa[k][1], aa[k][2])), Eigen::Vector3d(pp[k][0], pp[k][1], pp[k][2]));
    };
    std::vector<std::string> ba_log;
    for (int round = 0; round < 2; round++) {
      std::vector<std::array<double, 3>> Xs(tracks.size());
      for (size_t ti = 0; ti < tracks.size(); ti++)
        for (int i = 0; i < 3; i++)
          Xs[ti][i] = tracks[ti].X(i);
      ceres::Problem problem;
      for (size_t ti = 0; ti < tracks.size(); ti++)
        for (const auto &o : tracks[ti].obs) {
          const int k = imgs[o.first].kf, c = imgs[o.first].cam;
          const cv::Point2f &pt = imgs[o.first].kps[o.second].pt;
          problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ReprojCost, 2, 3, 3, 3>(
                                       new ReprojCost(R_cam_imu[c], t_cam_imu[c], vf, vc, Eigen::Vector2d(pt.x, pt.y))),
                                   new ceres::HuberLoss(P.huber_px), aa[k].data(), pp[k].data(), Xs[ti].data());
        }
      for (int k = 0; k + 1 < K; k++) {
        const Eigen::Matrix4d T_ab = T_inv(kfs[k].T_vio) * kfs[k + 1].T_vio;
        const double st = P.rel_sigma_t + P.rel_sigma_t_frac * T_ab.block<3, 1>(0, 3).norm();
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RelPoseCost, 6, 3, 3, 3, 3>(
                                     new RelPoseCost(T_ab, st, P.rel_sigma_r_deg * M_PI / 180.0)),
                                 nullptr, aa[k].data(), pp[k].data(), aa[k + 1].data(), pp[k + 1].data());
      }
      for (int k = 0; k < K; k++) {
        const Eigen::Vector3d g = kfs[k].T_vio.block<3, 3>(0, 0).transpose() * Eigen::Vector3d::UnitZ();
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TiltCost, 3, 3>(new TiltCost(g, P.tilt_sigma_deg * M_PI / 180.0)),
                                 nullptr, aa[k].data());
      }
      problem.SetParameterBlockConstant(aa[0].data()); // gauge: the first keyframe stays where the VIO put it
      problem.SetParameterBlockConstant(pp[0].data());
      ceres::Solver::Options opt;
      opt.linear_solver_type = ceres::SPARSE_SCHUR;
      auto *ordering = new ceres::ParameterBlockOrdering;
      for (auto &x : Xs)
        if (problem.HasParameterBlock(x.data()))
          ordering->AddElementToGroup(x.data(), 0);
      for (int k = 0; k < K; k++) {
        if (problem.HasParameterBlock(aa[k].data()))
          ordering->AddElementToGroup(aa[k].data(), 1);
        if (problem.HasParameterBlock(pp[k].data()))
          ordering->AddElementToGroup(pp[k].data(), 1);
      }
      opt.linear_solver_ordering.reset(ordering);
      opt.max_num_iterations = P.ba_iters;
      opt.num_threads = P.threads;
      opt.function_tolerance = 1e-6;
      struct Cb : ceres::IterationCallback {
        int round, total;
        ceres::CallbackReturnType operator()(const ceres::IterationSummary &s) override {
          progress(round == 0 ? "ba1" : "ba2", s.iteration, total);
          return ceres::SOLVER_CONTINUE;
        }
      } cb;
      cb.round = round;
      cb.total = P.ba_iters;
      opt.callbacks.push_back(&cb);
      opt.update_state_every_iteration = false;
      ceres::Solver::Summary sum;
      ceres::Solve(opt, &problem, &sum);
      progress(round == 0 ? "ba1" : "ba2", P.ba_iters, P.ba_iters);
      char line[256];
      std::snprintf(line, sizeof(line), "BA round %d: %zu landmarks, %d residual blocks, cost %.4g -> %.4g, %d iterations, %s, %.1f s",
                    round + 1, tracks.size(), sum.num_residual_blocks, sum.initial_cost, sum.final_cost, (int)sum.iterations.size() - 1,
                    sum.termination_type == ceres::CONVERGENCE ? "converged" : "stopped", sum.total_time_in_seconds);
      std::printf("%s\n", line);
      ba_log.push_back(line);
      pull();
      for (size_t ti = 0; ti < tracks.size(); ti++)
        tracks[ti].X = Eigen::Vector3d(Xs[ti][0], Xs[ti][1], Xs[ti][2]);
      // drop what the adjusted poses disagree with
      std::vector<uint8_t> ok(tracks.size(), 0);
      parallel_for((int)tracks.size(), P.threads, [&](int ti) { ok[ti] = triangulate(T_kf, tracks[ti], P.max_reproj_px, false); });
      std::vector<Track> kept;
      for (size_t ti = 0; ti < tracks.size(); ti++)
        if (ok[ti])
          kept.push_back(std::move(tracks[ti]));
      std::printf("  %zu landmarks within %.1f px of every view\n", kept.size(), P.max_reproj_px);
      tracks.swap(kept);
    }
    timing["ba"] = now_s() - t0;
    double err1_mean, err1_med;
    reproj_stats(T_kf, err1_mean, err1_med);
    std::vector<double> dpos, dyaw;
    for (int k = 0; k < K; k++) {
      dpos.push_back((T_kf[k].block<3, 1>(0, 3) - kfs[k].T_vio.block<3, 1>(0, 3)).norm());
      dyaw.push_back(std::abs(wrap_rad(yaw_of(T_kf[k].block<3, 3>(0, 0)) - yaw_of(kfs[k].T_vio.block<3, 3>(0, 0)))) * 180 / M_PI);
    }
    std::printf("reprojection: mean %.2f px / median %.2f px before BA, %.2f / %.2f after; keyframes moved by BA: median %.3f m, max %.3f m, "
                "yaw max %.2f deg\n",
                err0_mean, err0_med, err1_mean, err1_med, median(dpos), *std::max_element(dpos.begin(), dpos.end()),
                *std::max_element(dyaw.begin(), dyaw.end()));

    // ---------------------------------------------------------------- 6. the map
    t0 = now_s();
    RuntimeMap m;
    m.view_size = P.view_size;
    m.view_fov_deg = P.view_fov;
    m.orb_scale_factor = spec.scale_factor;
    m.orb_n_levels = spec.n_levels;
    m.orb_edge_threshold = spec.edge_threshold;
    m.orb_patch_size = spec.patch_size;
    m.orb_fast_threshold = spec.fast_threshold;
    m.orb_grid = spec.grid;
    m.orb_n_features_map = P.orb_n;
    for (const auto &c : cams) {
      m.rig_fisheye.push_back(c.model());
      m.rig_T_imu_cam.push_back(c.T_imu_cam);
    }
    // The run started at its first VIO pose; BA moved the keyframes. Carry the start along with the first keyframe (fixed).
    m.start_T_map_imu = T_kf[0] * T_inv(kfs[0].T_vio) * T_start;
    std::vector<std::vector<int>> vis(K);
    std::vector<cv::Mat> rows;
    for (size_t ti = 0; ti < tracks.size(); ti++) {
      const Track &t = tracks[ti];
      const int lid = (int)m.points_xyz.size();
      m.points_xyz.push_back(t.X);
      double e = 0;
      for (const auto &o : t.obs) {
        double depth;
        e += reproj(T_kf, o.first, o.second, t.X, depth);
        vis[imgs[o.first].kf].push_back(lid);
      }
      m.points_err.push_back((float)(e / t.obs.size()));
      m.points_track.push_back((uint16_t)std::min<size_t>(t.obs.size(), 65535));
      // up to 3 descriptors: the medoid, then the farthest from those chosen
      const int n = (int)t.obs.size();
      std::vector<std::vector<int>> dist(n, std::vector<int>(n, 0));
      for (int p = 0; p < n; p++)
        for (int q = p + 1; q < n; q++)
          dist[p][q] = dist[q][p] = (int)cv::norm(imgs[t.obs[p].first].desc.row(t.obs[p].second), imgs[t.obs[q].first].desc.row(t.obs[q].second),
                                                  cv::NORM_HAMMING);
      std::vector<int> chosen;
      int best = 0;
      long best_sum = -1;
      for (int p = 0; p < n; p++) {
        const long s = std::accumulate(dist[p].begin(), dist[p].end(), 0L);
        if (best_sum < 0 || s < best_sum) {
          best_sum = s;
          best = p;
        }
      }
      chosen.push_back(best);
      while ((int)chosen.size() < std::min(3, n)) {
        int far = -1, far_d = -1;
        for (int p = 0; p < n; p++) {
          if (std::find(chosen.begin(), chosen.end(), p) != chosen.end())
            continue;
          int dmin = 1 << 30;
          for (int c : chosen)
            dmin = std::min(dmin, dist[p][c]);
          if (dmin > far_d) {
            far_d = dmin;
            far = p;
          }
        }
        if (far < 0 || far_d < 8) // the rest look the same as those chosen
          break;
        chosen.push_back(far);
      }
      for (int c : chosen) {
        rows.push_back(imgs[t.obs[c].first].desc.row(t.obs[c].second));
        m.desc_point.push_back(lid);
      }
    }
    cv::vconcat(rows, m.desc);
    m.kf_vis_ptr.push_back(0);
    for (int k = 0; k < K; k++) {
      auto &v = vis[k];
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
      m.kf_stamp.push_back(kfs[k].stamp);
      m.kf_T_map_imu.push_back(T_kf[k]);
      m.kf_vis_idx.insert(m.kf_vis_idx.end(), v.begin(), v.end());
      m.kf_vis_ptr.push_back((int32_t)m.kf_vis_idx.size());
    }
    size_t n_obs = 0;
    for (const auto &t : tracks)
      n_obs += t.obs.size();
    if (P.thumb_size > 0) {
      m.thumb_size = P.thumb_size;
      m.kf_thumbs_ptr.push_back(0);
      for (const auto &im : imgs) {
        m.kf_thumbs.insert(m.kf_thumbs.end(), im.thumb.begin(), im.thumb.end());
        m.kf_thumbs_ptr.push_back((int32_t)m.kf_thumbs.size());
      }
    }
    std::ostringstream meta;
    meta << "{\"builder\": \"maploc_build\", \"keyframes\": " << K << ", \"cameras\": " << C << ", \"landmarks\": " << tracks.size()
         << ", \"observations\": " << n_obs << ", \"reproj_mean_px\": " << jnum(err1_mean, 3) << ", \"calib\": \"" << calib << "\"}";
    m.meta_json = meta.str();
    m.save(out);
    timing["map"] = now_s() - t0;
    std::printf("map: %zu landmarks, %d descriptor rows, %zu observations, %d keyframes -> %s\n", m.points_xyz.size(), m.desc.rows, n_obs,
                K, out.c_str());

    // ---------------------------------------------------------------- 7. self test
    int st_try = 0, st_ok = 0;
    std::vector<double> st_err, st_yaw, st_inl;
    if (P.selftest > 0) {
      t0 = now_s();
      const RuntimeMap loaded = RuntimeMap::load(out);
      LocParams lp;
      lp.threads = 1;
      Localizer loc(std::make_shared<RuntimeMap>(loaded), cams, lp);
      loc.set_start(loaded.start_T_map_imu * T_inv(T_start));
      const int step = std::max(1, K / P.selftest);
      const int n = (K + step - 1) / step;
      Counter cnt("selftest", n);
      for (int k = 0; k < K; k += step) {
        std::vector<cv::Mat> grays;
        for (int c = 0; c < C; c++)
          grays.push_back(cv::imread(kfs[k].files[c], cv::IMREAD_GRAYSCALE));
        const LocResult r = loc.localize(grays, kfs[k].T_vio, kfs[k].stamp);
        st_try++;
        if (r.ok) {
          st_ok++;
          st_err.push_back((r.T_meas.block<3, 1>(0, 3) - T_kf[k].block<3, 1>(0, 3)).norm());
          st_yaw.push_back(std::abs(wrap_rad(yaw_of(r.T_meas.block<3, 3>(0, 0)) - yaw_of(T_kf[k].block<3, 3>(0, 0)))) * 180 / M_PI);
          st_inl.push_back(r.inliers);
        }
        cnt.tick();
      }
      timing["selftest"] = now_s() - t0;
      std::printf("self test: %d of %d keyframes relocalized, position error median %.3f m, yaw %.2f deg, inliers median %.0f\n", st_ok,
                  st_try, median(st_err), median(st_yaw), median(st_inl));
    }

    // ---------------------------------------------------------------- report + preview
    const std::string rep = a.get("--report");
    if (!rep.empty()) {
      std::ofstream f(rep);
      f << "{\n \"keyframes\": " << K << ", \"cameras\": " << C << ", \"images\": " << I << ",\n"
        << " \"keypoints_per_view\": " << jnum(double(base[I]) / I, 1) << ",\n"
        << " \"pairs\": " << pairs.size() << ", \"pairs_verified\": " << pairs_ok << ", \"pairs_revisit\": " << loop_pairs
        << ", \"matches\": " << n_match << ",\n"
        << " \"landmarks\": " << tracks.size() << ", \"observations\": " << n_obs << ", \"descriptor_rows\": " << m.desc.rows << ",\n"
        << " \"reproj_before\": {\"mean\": " << jnum(err0_mean, 3) << ", \"median\": " << jnum(err0_med, 3) << "},\n"
        << " \"reproj_after\": {\"mean\": " << jnum(err1_mean, 3) << ", \"median\": " << jnum(err1_med, 3) << "},\n"
        << " \"ba_moved\": {\"median_m\": " << jnum(median(dpos)) << ", \"max_m\": " << jnum(*std::max_element(dpos.begin(), dpos.end()))
        << ", \"max_yaw_deg\": " << jnum(*std::max_element(dyaw.begin(), dyaw.end()), 3) << "},\n"
        << " \"ba\": [";
      for (size_t i = 0; i < ba_log.size(); i++)
        f << (i ? ", " : "") << "\"" << ba_log[i] << "\"";
      f << "],\n \"selftest\": {\"tried\": " << st_try << ", \"ok\": " << st_ok << ", \"pos_err_median_m\": " << jnum(median(st_err))
        << ", \"yaw_err_median_deg\": " << jnum(median(st_yaw), 3) << ", \"inliers_median\": " << jnum(median(st_inl), 0) << "},\n"
        << " \"params\": {\"view_size\": " << P.view_size << ", \"view_fov_deg\": " << jnum(P.view_fov, 1) << ", \"orb_n\": " << P.orb_n
        << ", \"pair_radius_m\": " << jnum(P.pair_radius, 2) << ", \"pair_axis_deg\": " << jnum(P.pair_axis_deg, 1)
        << ", \"pairs_per_image\": " << P.pairs_per_image << ", \"max_reproj_px\": " << jnum(P.max_reproj_px, 2)
        << ", \"min_angle_deg\": " << jnum(P.min_angle_deg, 2) << "},\n \"seconds\": {";
      bool first = true;
      for (const auto &kv : timing) {
        f << (first ? "" : ", ") << "\"" << kv.first << "\": " << jnum(kv.second, 1);
        first = false;
      }
      f << ", \"total\": " << jnum(now_s() - t_start, 1) << "}\n}\n";
    }
    const std::string prev = a.get("--preview");
    if (!prev.empty()) {
      std::ofstream f(prev);
      const size_t stride = std::max<size_t>(1, (m.points_xyz.size() + 149999) / 150000); // the page draws ~150k points well
      f << "{\"stride\": " << stride << ", \"points\": [";
      for (size_t i = 0; i < m.points_xyz.size(); i += stride) {
        double g = 0;
        for (const auto &o : tracks[i].obs)
          g += imgs[o.first].grey[o.second];
        f << (i ? "," : "") << "[" << jnum(m.points_xyz[i].x(), 3) << "," << jnum(m.points_xyz[i].y(), 3) << "," << jnum(m.points_xyz[i].z(), 3)
          << "," << (int)std::lround(g / tracks[i].obs.size()) << "]";
      }
      f << "], \"keyframes\": [";
      for (int k = 0; k < K; k++) {
        const Eigen::Quaterniond q(Eigen::Matrix3d(T_kf[k].block<3, 3>(0, 0)));
        f << (k ? "," : "") << "[" << jnum(T_kf[k](0, 3), 4) << "," << jnum(T_kf[k](1, 3), 4) << "," << jnum(T_kf[k](2, 3), 4) << ","
          << jnum(q.w(), 6) << "," << jnum(q.x(), 6) << "," << jnum(q.y(), 6) << "," << jnum(q.z(), 6) << "]";
      }
      f << "], \"vio\": [";
      for (int k = 0; k < K; k++) {
        const Eigen::Matrix4d &T = kfs[k].T_vio;
        f << (k ? "," : "") << "[" << jnum(T(0, 3), 3) << "," << jnum(T(1, 3), 3) << "," << jnum(T(2, 3), 3) << ","
          << jnum(yaw_of(T.block<3, 3>(0, 0)), 4) << "]";
      }
      const Eigen::Matrix4d &S = m.start_T_map_imu;
      f << "], \"start\": [" << jnum(S(0, 3), 3) << "," << jnum(S(1, 3), 3) << "," << jnum(S(2, 3), 3) << ","
        << jnum(yaw_of(S.block<3, 3>(0, 0)), 4) << "], \"cams\": [";
      for (int c = 0; c < C; c++) {
        std::string label = cams[c].topic.substr(std::min(cams[c].topic.size(), cams[c].topic.find_first_not_of('/')));
        label = label.substr(0, label.find('/'));
        f << (c ? "," : "") << "{\"name\": \"" << cams[c].name << "\", \"label\": \"" << label << "\", \"aspect\": "
          << jnum(double(cams[c].height) / cams[c].width, 4) << ", \"T_imu_cam\": [";
        for (int r = 0; r < 3; r++)
          f << (r ? "," : "") << "[" << jnum(cams[c].T_imu_cam(r, 0), 6) << "," << jnum(cams[c].T_imu_cam(r, 1), 6) << ","
            << jnum(cams[c].T_imu_cam(r, 2), 6) << "," << jnum(cams[c].T_imu_cam(r, 3), 5) << "]";
        f << "]}";
      }
      f << "], \"stamps\": [";
      for (int k = 0; k < K; k++)
        f << (k ? "," : "") << jnum(kfs[k].stamp, 3);
      f << "], \"files\": [";
      for (int k = 0; k < K; k++) {
        f << (k ? "," : "") << "[";
        for (int c = 0; c < C; c++) {
          const std::string &p = kfs[k].files[c];
          f << (c ? "," : "") << "\"" << p.substr(p.find_last_of('/') + 1) << "\"";
        }
        f << "]";
      }
      f << "], \"vis\": [";   // landmarks each keyframe sees, as indices into points (those kept by the stride)
      for (int k = 0; k < K; k++) {
        f << (k ? "," : "") << "[";
        bool first = true;
        for (int j = m.kf_vis_ptr[k]; j < m.kf_vis_ptr[k + 1]; j++) {
          const int lid = m.kf_vis_idx[j];
          if (lid % (int)stride)
            continue;
          f << (first ? "" : ",") << lid / (int)stride;
          first = false;
        }
        f << "]";
      }
      f << "]}\n";
    }
    std::printf("done in %.1f s\n", now_s() - t_start);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "maploc_build: %s\n", e.what());
    return 1;
  }
  return 0;
}
