/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 */
#include "map.h"

#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace ov_maploc {

namespace {

struct Raw {
  uint8_t dtype = 0;
  std::vector<uint64_t> shape;
  std::vector<char> data;
  size_t count() const {
    size_t n = 1;
    for (auto s : shape)
      n *= s;
    return n;
  }
};

size_t dtype_size(uint8_t d) {
  switch (d) {
  case 1:
    return 1;
  case 2:
    return 2;
  case 3:
  case 4:
    return 4;
  case 5:
    return 8;
  }
  throw std::runtime_error("map: unknown dtype " + std::to_string(d));
}

template <class T> void read_pod(std::ifstream &f, T &v) {
  f.read(reinterpret_cast<char *>(&v), sizeof(T));
  if (!f)
    throw std::runtime_error("map: truncated file");
}

template <class T> std::vector<T> as_vec(const Raw &r, uint8_t dtype, const std::string &name) {
  if (r.dtype != dtype)
    throw std::runtime_error("map: array " + name + " has dtype " + std::to_string(r.dtype));
  std::vector<T> v(r.count());
  std::memcpy(v.data(), r.data.data(), r.data.size());
  return v;
}

std::vector<Eigen::Matrix4d> as_mat4(const Raw &r, const std::string &name) {
  auto v = as_vec<double>(r, 5, name);
  if (v.size() % 16)
    throw std::runtime_error("map: " + name + " is not a stack of 4x4");
  std::vector<Eigen::Matrix4d> out(v.size() / 16);
  for (size_t i = 0; i < out.size(); i++)
    out[i] = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(v.data() + 16 * i);
  return out;
}

} // namespace

RuntimeMap RuntimeMap::load(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("map: cannot open " + path);
  char magic[8];
  f.read(magic, 8);
  if (!f || std::string(magic, 8) != "OVMAPLOC")
    throw std::runtime_error("map: " + path + " is not an ov_maploc map (convert map.npz with map_npz_to_bin.py)");
  uint32_t version = 0, n = 0;
  read_pod(f, version);
  if (version != 1)
    throw std::runtime_error("map: unsupported version " + std::to_string(version));
  read_pod(f, n);
  std::map<std::string, Raw> arr;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t len = 0;
    read_pod(f, len);
    std::string name(len, '\0');
    f.read(&name[0], len);
    Raw r;
    uint8_t ndim = 0;
    uint16_t pad = 0;
    read_pod(f, r.dtype);
    read_pod(f, ndim);
    read_pod(f, pad);
    r.shape.resize(ndim);
    for (auto &s : r.shape)
      read_pod(f, s);
    r.data.resize(r.count() * dtype_size(r.dtype));
    f.read(r.data.data(), (std::streamsize)r.data.size());
    if (!f)
      throw std::runtime_error("map: truncated array " + name);
    arr[name] = std::move(r);
  }
  auto get = [&](const std::string &name) -> const Raw & {
    auto it = arr.find(name);
    if (it == arr.end())
      throw std::runtime_error("map: missing array " + name);
    return it->second;
  };

  RuntimeMap m;
  auto xyz = as_vec<float>(get("points_xyz"), 4, "points_xyz");
  m.points_xyz.resize(xyz.size() / 3);
  for (size_t i = 0; i < m.points_xyz.size(); i++)
    m.points_xyz[i] = Eigen::Vector3d(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
  m.points_err = as_vec<float>(get("points_err"), 4, "points_err");
  m.points_track = as_vec<uint16_t>(get("points_track"), 2, "points_track");

  const Raw &d = get("desc");
  if (d.dtype != 1 || d.shape.size() != 2 || d.shape[1] != 32)
    throw std::runtime_error("map: desc must be M x 32 uint8 (ORB)");
  m.desc = cv::Mat((int)d.shape[0], 32, CV_8U);
  std::memcpy(m.desc.data, d.data.data(), d.data.size());
  m.desc_point = as_vec<int32_t>(get("desc_point"), 3, "desc_point");
  // descriptor rows per landmark (counting sort, no assumption on order)
  const size_t N = m.points_xyz.size();
  m.point_desc_ptr.assign(N + 1, 0);
  for (int32_t p : m.desc_point) {
    if (p < 0 || (size_t)p >= N)
      throw std::runtime_error("map: desc_point out of range");
    m.point_desc_ptr[p + 1]++;
  }
  for (size_t i = 0; i < N; i++)
    m.point_desc_ptr[i + 1] += m.point_desc_ptr[i];
  m.point_desc_rows.resize(m.desc_point.size());
  std::vector<int32_t> fill(m.point_desc_ptr.begin(), m.point_desc_ptr.end() - 1);
  for (size_t r = 0; r < m.desc_point.size(); r++)
    m.point_desc_rows[fill[m.desc_point[r]]++] = (int32_t)r;

  m.kf_stamp = as_vec<double>(get("kf_stamp"), 5, "kf_stamp");
  m.kf_T_map_imu = as_mat4(get("kf_T_map_imu"), "kf_T_map_imu");
  m.kf_vis_ptr = as_vec<int32_t>(get("kf_vis_ptr"), 3, "kf_vis_ptr");
  m.kf_vis_idx = as_vec<int32_t>(get("kf_vis_idx"), 3, "kf_vis_idx");
  if (m.kf_vis_ptr.size() != m.kf_stamp.size() + 1 || m.kf_T_map_imu.size() != m.kf_stamp.size())
    throw std::runtime_error("map: keyframe arrays disagree");

  m.start_T_map_imu = as_mat4(get("start_T_map_imu"), "start_T_map_imu").at(0);
  auto view = as_vec<double>(get("view"), 5, "view");
  m.view_size = (int)view.at(0);
  m.view_fov_deg = view.at(1);
  auto orb = as_vec<double>(get("orb"), 5, "orb");
  m.orb_scale_factor = orb.at(0);
  m.orb_n_levels = (int)orb.at(1);
  m.orb_edge_threshold = (int)orb.at(2);
  m.orb_patch_size = (int)orb.at(3);
  m.orb_fast_threshold = (int)orb.at(4);
  m.orb_grid = (int)orb.at(5);
  m.orb_n_features_map = as_vec<int32_t>(get("orb_n_features_map"), 3, "orb_n_features_map").at(0);
  auto fe = as_vec<double>(get("rig_fisheye"), 5, "rig_fisheye");
  for (size_t i = 0; i + 5 <= fe.size(); i += 5)
    m.rig_fisheye.push_back({fe[i], fe[i + 1], fe[i + 2], fe[i + 3], fe[i + 4]});
  m.rig_T_imu_cam = as_mat4(get("rig_T_imu_cam"), "rig_T_imu_cam");
  if (arr.count("T_world_map")) {
    m.has_T_world_map = true;
    m.T_world_map = as_mat4(get("T_world_map"), "T_world_map").at(0);
  }
  if (arr.count("meta_json")) {
    const Raw &j = get("meta_json");
    m.meta_json = std::string(j.data.data(), j.data.size());
  }
  return m;
}

} // namespace ov_maploc
