/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 *
 * ROS 1 node, the port of ros_node.cpp (ROS 2): relocalization against a
 * prebuilt map, known start, loosely coupled to a running OpenVINS (which is
 * not modified).
 *
 *   roslaunch ov_maploc maploc.launch map:=/path/map.bin \
 *       calib:=/path/kalibr_imucam_chain.yaml masks:=/path/masks
 *
 * Subscribes  <rostopic of each camera in calib>   sensor_msgs/Image (rgb8/bgr8/mono8), or with
 *                                                   ~transport:=compressed <rostopic>/compressed
 *                                                   (decoded only for the frames relocalized)
 *             /ov_msckf/poseimu                     pose after every OpenVINS update (frame "global")
 * Publishes   TF map -> global                      T_map_odom (4 DoF), with every VIO pose, same stamp
 *             /maploc/pose                          PoseWithCovarianceStamped in "map" at the VIO rate
 *             /maploc/status                        std_msgs/String, JSON per relocalization attempt
 *             /maploc/map_points                    PointCloud2 (latched), landmarks in "map"
 *             /maploc/debug, /maploc/debug_image    with ~debug: per attempt the 2D-3D matches (JSON: per
 *                                                   camera [u, v, landmark, inlier] in its view) and the
 *                                                   views side by side (JPEG, one view_size square each)
 */
#include "geometry.h"
#include "localizer.h"

#include <algorithm>
#include <atomic>
#include <boost/bind/bind.hpp>
#include <condition_variable>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include <iomanip>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sstream>
#include <std_msgs/String.h>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>

using namespace ov_maploc;

namespace {

double stamp_s(const ros::Time &t) { return std::round((t.sec + t.nsec * 1e-9) * 1e6) / 1e6; }

Eigen::Matrix4d pose_to_T(const geometry_msgs::Pose &p) {
  // OpenVINS writes its JPL quaternion of R_GtoI into these fields; read as a
  // Hamilton quaternion that is the IMU orientation in its global frame.
  const Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
  return T_from(q.normalized().toRotationMatrix(), Eigen::Vector3d(p.position.x, p.position.y, p.position.z));
}

void T_to_pose(const Eigen::Matrix4d &T, geometry_msgs::Pose &p) {
  const Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
  p.position.x = T(0, 3);
  p.position.y = T(1, 3);
  p.position.z = T(2, 3);
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
}

/// One camera's image of a frame as received: raw, or compressed and decoded
/// only when the frame is relocalized.
struct Img {
  sensor_msgs::ImageConstPtr raw;
  sensor_msgs::CompressedImageConstPtr jpg;
  explicit operator bool() const { return raw || jpg; }
};

cv::Mat to_gray(const Img &m) {
  if (m.jpg) {
    cv::Mat g = cv::imdecode(m.jpg->data, cv::IMREAD_GRAYSCALE);
    if (g.empty())
      throw std::runtime_error("maploc: cannot decode a '" + m.jpg->format + "' image");
    return g;
  }
  cv_bridge::CvImageConstPtr cv = cv_bridge::toCvShare(m.raw);
  cv::Mat g;
  if (m.raw->encoding == "rgb8")
    cv::cvtColor(cv->image, g, cv::COLOR_RGB2GRAY);
  else if (m.raw->encoding == "bgr8")
    cv::cvtColor(cv->image, g, cv::COLOR_BGR2GRAY);
  else if (m.raw->encoding == "mono8")
    g = cv->image.clone();
  else
    throw std::runtime_error("maploc: unsupported image encoding " + m.raw->encoding);
  return g;
}

} // namespace

class MaplocNode {
public:
  MaplocNode(ros::NodeHandle &nh, ros::NodeHandle &pnh) {
    std::string map_path, calib, masks, transport;
    double reloc_hz;
    pnh.param<std::string>("map", map_path, "");
    pnh.param<std::string>("calib", calib, "");
    pnh.param<std::string>("masks", masks, "");
    pnh.param<std::string>("transport", transport, "raw"); // raw | compressed
    pnh.param<std::string>("pose_topic", pose_topic_, "/ov_msckf/poseimu");
    pnh.param("reloc_hz", reloc_hz, 1.0);
    pnh.param<std::string>("start", start_, ""); // "x,y,yaw_deg" or "x,y,z,yaw_deg" of the first VIO pose in the map
    pnh.param<std::string>("map_frame", map_frame_, "map");
    pnh.param<std::string>("odom_frame", odom_frame_, "global");
    LocParams p;
    pnh.param("orb_n", p.orb_n, p.orb_n);
    pnh.param("kf_radius", p.kf_radius, p.kf_radius);
    pnh.param("kf_radius_max", p.kf_radius_max, p.kf_radius_max);
    pnh.param("kf_max", p.kf_max, p.kf_max);
    pnh.param("gate_margin_deg", p.gate_margin_deg, p.gate_margin_deg);
    pnh.param("ratio", p.ratio, p.ratio);
    pnh.param("max_hamming", p.max_hamming, p.max_hamming);
    pnh.param("ransac_px", p.ransac_px, p.ransac_px);
    pnh.param("ransac_trials", p.ransac_trials, p.ransac_trials);
    pnh.param("min_inliers", p.min_inliers, p.min_inliers);
    pnh.param("max_tilt_deg", p.max_tilt_deg, p.max_tilt_deg);
    pnh.param("max_jump_m", p.max_jump_m, p.max_jump_m);
    pnh.param("max_jump_deg", p.max_jump_deg, p.max_jump_deg);
    pnh.param("alpha", p.alpha, p.alpha);
    pnh.param("good_inliers", p.good_inliers, p.good_inliers);
    pnh.param("jump_rate", p.jump_rate, p.jump_rate);
    pnh.param("reanchor_n", p.reanchor_n, p.reanchor_n);
    pnh.param("reanchor_inliers", p.reanchor_inliers, p.reanchor_inliers);
    pnh.param("reanchor_tol_m", p.reanchor_tol_m, p.reanchor_tol_m);
    pnh.param("reanchor_tol_deg", p.reanchor_tol_deg, p.reanchor_tol_deg);
    pnh.param("threads", p.threads, p.threads);
    pnh.param("search_global", p.search_global, p.search_global);
    pnh.param("debug", p.debug, p.debug);
    pnh.param("search_min_inliers", p.search_min_inliers, p.search_min_inliers);
    if (map_path.empty() || calib.empty())
      throw std::runtime_error("maploc: set the '~map' and '~calib' parameters");
    if (transport != "raw" && transport != "compressed")
      throw std::runtime_error("maploc: ~transport must be 'raw' or 'compressed', not '" + transport + "'");
    if (reloc_hz <= 0)
      throw std::runtime_error("maploc: ~reloc_hz must be > 0");

    auto map = std::make_shared<RuntimeMap>(RuntimeMap::load(map_path));
    const auto cams = load_ring(calib, masks);
    loc_ = std::make_unique<Localizer>(map, cams, p);
    n_cams_ = (int)cams.size();

    for (int i = 0; i < n_cams_; i++) {
      if (transport == "compressed")
        subs_.push_back(nh.subscribe<sensor_msgs::CompressedImage>(cams[i].topic + "/compressed", 5,
                                                                    boost::bind(&MaplocNode::on_jpg, this, boost::placeholders::_1, i)));
      else
        subs_.push_back(
            nh.subscribe<sensor_msgs::Image>(cams[i].topic, 5, boost::bind(&MaplocNode::on_raw, this, boost::placeholders::_1, i)));
    }
    sub_pose_ = nh.subscribe(pose_topic_, 100, &MaplocNode::on_pose, this);
    pub_pose_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/maploc/pose", 10);
    pub_status_ = nh.advertise<std_msgs::String>("/maploc/status", 10);
    pub_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("/maploc/map_points", 1, /*latch=*/true);
    if (p.debug) {
      pub_dbg_ = nh.advertise<std_msgs::String>("/maploc/debug", 2);
      pub_dbg_img_ = nh.advertise<sensor_msgs::CompressedImage>("/maploc/debug_image", 2);
    }
    publish_map_points(*map);
    timer_reloc_ = nh.createWallTimer(ros::WallDuration(1.0 / reloc_hz), &MaplocNode::tick_reloc, this);
    worker_ = std::thread([this] { worker_loop(); });
    ROS_INFO("map %s: %zu landmarks, %zu keyframes, %d cameras (%s); relocalizing at %.1f Hz", map_path.c_str(), map->num_points(),
             map->num_keyframes(), n_cams_, transport.c_str(), reloc_hz);
  }

  ~MaplocNode() {
    {
      std::lock_guard<std::mutex> lk(job_mtx_);
      stop_ = true;
    }
    job_cv_.notify_all();
    if (worker_.joinable())
      worker_.join();
  }

private:
  struct Job {
    double stamp = 0;
    std::vector<Img> imgs;
    Eigen::Matrix4d T_odom_imu;
  };

  void on_raw(const sensor_msgs::ImageConstPtr &m, int i) {
    Img img;
    img.raw = m;
    on_image(i, m->header.stamp, img);
  }

  void on_jpg(const sensor_msgs::CompressedImageConstPtr &m, int i) {
    Img img;
    img.jpg = m;
    on_image(i, m->header.stamp, img);
  }

  void on_image(int i, const ros::Time &stamp, const Img &img) {
    double s = stamp_s(stamp);
    // The cameras of one frame share a trigger edge, but each driver converts it
    // with its own clock read: on the RK3588_SLAM board they differ by up to
    // ~1 us, and an exact key splits about a quarter of the frames. Join within 1 ms.
    auto near = pending_.lower_bound(s - 1e-3);
    if (near != pending_.end() && near->first <= s + 1e-3)
      s = near->first;
    auto &slot = pending_[s];
    slot.resize(n_cams_);
    slot[i] = img;
    if (std::all_of(slot.begin(), slot.end(), [](const Img &x) { return (bool)x; })) {
      frames_.push_back({s, slot});
      if (frames_.size() > 8)
        frames_.pop_front();
      for (auto it = pending_.begin(); it != pending_.end() && it->first <= s;)
        it = pending_.erase(it);
    }
    while (pending_.size() > 64) // a camera stopped: do not grow without bound
      pending_.erase(pending_.begin());
  }

  void on_pose(const geometry_msgs::PoseWithCovarianceStampedConstPtr &m) {
    const Eigen::Matrix4d T = pose_to_T(m->pose.pose);
    poses_.push_back({stamp_s(m->header.stamp), T});
    if (poses_.size() > 400)
      poses_.pop_front();
    if (!loc_->started()) {
      Eigen::Matrix4d T_map_imu0 = loc_->map().start_T_map_imu;
      if (!start_.empty()) {
        std::vector<double> v;
        std::string tok;
        std::istringstream ss(start_);
        while (std::getline(ss, tok, ','))
          v.push_back(std::stod(tok));
        if (v.size() != 3 && v.size() != 4)
          throw std::runtime_error("maploc: start must be 'x,y,yaw_deg' or 'x,y,z,yaw_deg'");
        const double z = v.size() == 4 ? v[2] : T_map_imu0(2, 3);
        T_map_imu0 = T_from(Rz(v.back() * M_PI / 180.0), Eigen::Vector3d(v[0], v[1], z));
      }
      loc_->set_start(T_map_imu0 * T_inv(T));
      ROS_INFO("known start set from the first VIO pose");
    }
    geometry_msgs::PoseWithCovarianceStamped out;
    out.header.stamp = m->header.stamp;
    out.header.frame_id = map_frame_;
    T_to_pose(loc_->predict(T), out.pose.pose);
    pub_pose_.publish(out);
    publish_tf(m->header.stamp);
  }

  /// VIO pose of the update run on the image stamped s: OpenVINS stamps poses
  /// at t_img + td with td estimated online, so track td instead of assuming 0.
  bool pose_for(double s, Eigen::Matrix4d &T) {
    double best = 1e9, bt = 0;
    for (const auto &kv : poses_) {
      const double d = std::abs(kv.first - (s + td_));
      if (d <= 0.04 && d < best) {
        best = d;
        bt = kv.first;
        T = kv.second;
      }
    }
    if (best > 1e8)
      return false;
    td_ = bt - s;
    return true;
  }

  void tick_reloc(const ros::WallTimerEvent &) {
    if (busy_ || frames_.empty() || !loc_->started())
      return;
    // newest frame OpenVINS already produced a pose for (it runs behind the images)
    for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
      Eigen::Matrix4d T;
      if (pose_for(it->first, T)) {
        {
          std::lock_guard<std::mutex> lk(job_mtx_);
          job_ = Job{it->first, it->second, T};
          has_job_ = true;
          busy_ = true;
        }
        frames_.clear();
        job_cv_.notify_one();
        return;
      }
    }
  }

  void worker_loop() {
    while (true) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(job_mtx_);
        job_cv_.wait(lk, [this] { return stop_ || has_job_; });
        if (stop_)
          return;
        job = std::move(job_);
        has_job_ = false;
      }
      try {
        std::vector<cv::Mat> grays;
        for (const auto &m : job.imgs)
          grays.push_back(to_gray(m));
        const LocResult r = loc_->localize(grays, job.T_odom_imu, job.stamp);
        n_try_++;
        if (r.ok) {
          n_ok_++;
          last_fix_ = job.stamp;
        }
        char buf[768];
        std::snprintf(buf, sizeof(buf),
                      "{\"t\": %.3f, \"ok\": %s, \"search\": %s, \"why\": \"%s\", \"inliers\": %d, \"matches\": %d, \"landmarks\": %d, "
                      "\"since_fix_s\": %.2f, \"jump_m\": %.2f, \"jump_deg\": %.2f, \"reanchor\": %s, "
                      "\"ms\": %.1f, \"ms_orb\": %.1f, \"ms_match\": %.1f, \"ms_pnp\": %.1f, "
                      "\"tilt_deg\": %.2f, \"fixes\": \"%d/%d\"}",
                      job.stamp, r.ok ? "true" : "false", r.search ? "true" : "false", r.why.c_str(), r.inliers, r.n_corr, r.n_landmarks,
                      last_fix_ < 0 ? -1.0 : job.stamp - last_fix_, r.jump_m, r.jump_deg, r.reanchor ? "true" : "false",
                      r.t_total * 1e3, r.t_orb * 1e3, r.t_match * 1e3, r.t_pnp * 1e3, r.tilt_deg, n_ok_, n_try_);
        std_msgs::String st;
        st.data = buf;
        pub_status_.publish(st);
        if (r.reanchor)
          ROS_WARN("re-anchored: %d consistent fixes beyond the jump gate (jump %.2f m, %.1f deg)", loc_->params().reanchor_n, r.jump_m,
                   r.jump_deg);
        if (!r.ok || n_try_ % 10 == 0)
          ROS_INFO("%s", buf);
        if (loc_->params().debug)
          publish_debug(job.stamp, r);
      } catch (const std::exception &e) {
        ROS_ERROR("relocalization failed: %s", e.what());
      }
      busy_ = false;
    }
  }

  /// The attempt for the page's match display: the views it matched in, and per camera every match.
  void publish_debug(double stamp, const LocResult &r) {
    std::ostringstream js;
    js << "{\"t\": " << std::fixed << std::setprecision(3) << stamp << ", \"ok\": " << (r.ok ? "true" : "false")
       << ", \"search\": " << (r.search ? "true" : "false") << ", \"why\": \"" << r.why << "\", \"inliers\": " << r.inliers
       << ", \"view\": " << (r.views.empty() || r.views[0].empty() ? 0 : r.views[0].cols) << ", \"cams\": [";
    for (int c = 0; c < n_cams_; c++) {
      js << (c ? "," : "") << "[";
      bool first = true;
      for (size_t i = 0; i < r.obs.size(); i++) {
        if (r.obs[i].cam != c)
          continue;
        js << (first ? "" : ",") << "[" << std::setprecision(1) << r.obs[i].uv.x() << "," << r.obs[i].uv.y() << "," << r.obs[i].lid << ","
           << (i < r.inlier.size() && r.inlier[i] ? 1 : 0) << "]";
        first = false;
      }
      js << "]";
    }
    js << "]}";
    std_msgs::String m;
    m.data = js.str();
    pub_dbg_.publish(m);
    std::vector<cv::Mat> vs;
    for (const auto &v : r.views)
      if (!v.empty())
        vs.push_back(v);
    if ((int)vs.size() == n_cams_) {
      cv::Mat strip;
      cv::hconcat(vs, strip);
      sensor_msgs::CompressedImage img;
      img.header.stamp = ros::Time(stamp);
      img.format = "jpeg";
      cv::imencode(".jpg", strip, img.data, {cv::IMWRITE_JPEG_QUALITY, 75});
      pub_dbg_img_.publish(img);
    }
  }

  void publish_tf(const ros::Time &stamp) {
    const Eigen::Matrix4d T = loc_->T_map_odom();
    geometry_msgs::TransformStamped tr;
    tr.header.stamp = stamp;
    tr.header.frame_id = map_frame_;
    tr.child_frame_id = odom_frame_;
    geometry_msgs::Pose p;
    T_to_pose(T, p);
    tr.transform.translation.x = p.position.x;
    tr.transform.translation.y = p.position.y;
    tr.transform.translation.z = p.position.z;
    tr.transform.rotation = p.orientation;
    tfb_.sendTransform(tr);
  }

  void publish_map_points(const RuntimeMap &m) {
    sensor_msgs::PointCloud2 pc;
    pc.header.frame_id = map_frame_;
    pc.header.stamp = ros::Time::now();
    pc.height = 1;
    pc.width = (uint32_t)m.num_points();
    const char *names[4] = {"x", "y", "z", "intensity"};
    for (int i = 0; i < 4; i++) {
      sensor_msgs::PointField f;
      f.name = names[i];
      f.offset = 4 * i;
      f.datatype = sensor_msgs::PointField::FLOAT32;
      f.count = 1;
      pc.fields.push_back(f);
    }
    pc.is_bigendian = false;
    pc.point_step = 16;
    pc.row_step = 16 * pc.width;
    pc.is_dense = true;
    pc.data.resize(pc.row_step);
    float *d = reinterpret_cast<float *>(pc.data.data());
    for (size_t i = 0; i < m.num_points(); i++) {
      d[4 * i] = (float)m.points_xyz[i].x();
      d[4 * i + 1] = (float)m.points_xyz[i].y();
      d[4 * i + 2] = (float)m.points_xyz[i].z();
      d[4 * i + 3] = (float)m.points_xyz[i].z();
    }
    pub_cloud_.publish(pc);
  }

  std::unique_ptr<Localizer> loc_;
  int n_cams_ = 0;
  std::string pose_topic_, start_, map_frame_, odom_frame_;
  std::vector<ros::Subscriber> subs_;
  ros::Subscriber sub_pose_;
  ros::Publisher pub_pose_, pub_status_, pub_cloud_, pub_dbg_, pub_dbg_img_;
  tf2_ros::TransformBroadcaster tfb_;
  ros::WallTimer timer_reloc_;

  // spinner thread only (ros::spin is single-threaded)
  std::map<double, std::vector<Img>> pending_;
  std::deque<std::pair<double, std::vector<Img>>> frames_;
  std::deque<std::pair<double, Eigen::Matrix4d>> poses_;
  double td_ = 0;

  // worker hand-off
  std::thread worker_;
  std::mutex job_mtx_;
  std::condition_variable job_cv_;
  Job job_;
  bool has_job_ = false, stop_ = false;
  std::atomic<bool> busy_{false};
  int n_try_ = 0, n_ok_ = 0;
  double last_fix_ = -1;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "maploc");
  ros::NodeHandle nh, pnh("~");
  try {
    MaplocNode node(nh, pnh);
    ros::spin();
  } catch (const std::exception &e) {
    ROS_FATAL("%s", e.what());
    return 1;
  }
  return 0;
}
