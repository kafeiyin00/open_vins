/*
 * ov_maploc: map reuse for a multi-fisheye rig on top of OpenVINS.
 * Copyright (C) 2026 kafeiyin00. GNU General Public License v3.0 (as OpenVINS).
 *
 * ROS 2 node: relocalization against a prebuilt map, known start, loosely
 * coupled to a running OpenVINS (which is not modified).
 *
 *   ros2 run ov_maploc maploc_node --ros-args -p map:=/path/map.bin \
 *       -p calib:=/path/kalibr_imucam_chain.yaml -p masks:=/path/masks
 *
 * Subscribes  <rostopic of each camera in calib>   sensor_msgs/Image (rgb8/bgr8/mono8)
 *             /ov_msckf/poseimu                     pose after every OpenVINS update (frame "global")
 * Publishes   TF map -> global                      T_map_odom (4 DoF), with every VIO pose, same stamp
 *             /maploc/pose                          PoseWithCovarianceStamped in "map" at the VIO rate
 *             /maploc/status                        std_msgs/String, JSON per relocalization attempt
 *             /maploc/map_points                    PointCloud2 (transient local), landmarks in "map"
 */
#include "geometry.h"
#include "localizer.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <map>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sstream>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>

using namespace ov_maploc;

namespace {

double stamp_s(const builtin_interfaces::msg::Time &t) { return std::round((t.sec + t.nanosec * 1e-9) * 1e6) / 1e6; }

Eigen::Matrix4d pose_to_T(const geometry_msgs::msg::Pose &p) {
  // OpenVINS writes its JPL quaternion of R_GtoI into these fields; read as a
  // Hamilton quaternion that is the IMU orientation in its global frame.
  const Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
  return T_from(q.normalized().toRotationMatrix(), Eigen::Vector3d(p.position.x, p.position.y, p.position.z));
}

void T_to_pose(const Eigen::Matrix4d &T, geometry_msgs::msg::Pose &p) {
  const Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
  p.position.x = T(0, 3);
  p.position.y = T(1, 3);
  p.position.z = T(2, 3);
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
}

cv::Mat to_gray(const sensor_msgs::msg::Image::ConstSharedPtr &m) {
  cv_bridge::CvImageConstPtr cv = cv_bridge::toCvShare(m);
  cv::Mat g;
  if (m->encoding == "rgb8")
    cv::cvtColor(cv->image, g, cv::COLOR_RGB2GRAY);
  else if (m->encoding == "bgr8")
    cv::cvtColor(cv->image, g, cv::COLOR_BGR2GRAY);
  else if (m->encoding == "mono8")
    g = cv->image.clone();
  else
    throw std::runtime_error("maploc: unsupported image encoding " + m->encoding);
  return g;
}

} // namespace

class MaplocNode : public rclcpp::Node {
public:
  MaplocNode() : Node("maploc") {
    const std::string map_path = declare_parameter<std::string>("map", "");
    const std::string calib = declare_parameter<std::string>("calib", "");
    const std::string masks = declare_parameter<std::string>("masks", "");
    pose_topic_ = declare_parameter<std::string>("pose_topic", "/ov_msckf/poseimu");
    const double reloc_hz = declare_parameter<double>("reloc_hz", 1.0);
    start_ = declare_parameter<std::string>("start", ""); // "x,y,yaw_deg" of the first VIO pose in the map
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "global");
    LocParams p;
    p.orb_n = declare_parameter<int>("orb_n", p.orb_n);
    p.kf_radius = declare_parameter<double>("kf_radius", p.kf_radius);
    p.kf_radius_max = declare_parameter<double>("kf_radius_max", p.kf_radius_max);
    p.kf_max = declare_parameter<int>("kf_max", p.kf_max);
    p.gate_margin_deg = declare_parameter<double>("gate_margin_deg", p.gate_margin_deg);
    p.ratio = declare_parameter<double>("ratio", p.ratio);
    p.max_hamming = declare_parameter<int>("max_hamming", p.max_hamming);
    p.ransac_px = declare_parameter<double>("ransac_px", p.ransac_px);
    p.ransac_trials = declare_parameter<int>("ransac_trials", p.ransac_trials);
    p.min_inliers = declare_parameter<int>("min_inliers", p.min_inliers);
    p.max_tilt_deg = declare_parameter<double>("max_tilt_deg", p.max_tilt_deg);
    p.max_jump_m = declare_parameter<double>("max_jump_m", p.max_jump_m);
    p.max_jump_deg = declare_parameter<double>("max_jump_deg", p.max_jump_deg);
    p.alpha = declare_parameter<double>("alpha", p.alpha);
    p.threads = declare_parameter<int>("threads", p.threads);
    if (map_path.empty() || calib.empty())
      throw std::runtime_error("maploc: set the 'map' and 'calib' parameters");

    auto map = std::make_shared<RuntimeMap>(RuntimeMap::load(map_path));
    const auto cams = load_ring(calib, masks);
    loc_ = std::make_unique<Localizer>(map, cams, p);
    n_cams_ = (int)cams.size();

    for (int i = 0; i < n_cams_; i++)
      subs_.push_back(create_subscription<sensor_msgs::msg::Image>(
          cams[i].topic, rclcpp::SensorDataQoS(), [this, i](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(i, m); }));
    sub_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        pose_topic_, 100, [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr m) { on_pose(m); });
    pub_pose_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/maploc/pose", 10);
    pub_status_ = create_publisher<std_msgs::msg::String>("/maploc/status", 10);
    pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("/maploc/map_points", rclcpp::QoS(1).transient_local());
    tfb_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    publish_map_points(*map);
    timer_reloc_ = create_wall_timer(std::chrono::duration<double>(1.0 / reloc_hz), [this] { tick_reloc(); });
    worker_ = std::thread([this] { worker_loop(); });
    RCLCPP_INFO(get_logger(), "map %s: %zu landmarks, %zu keyframes, %d cameras; relocalizing at %.1f Hz", map_path.c_str(),
                map->num_points(), map->num_keyframes(), n_cams_, reloc_hz);
  }

  ~MaplocNode() override {
    {
      std::lock_guard<std::mutex> lk(job_mtx_);
      stop_ = true;
    }
    job_cv_.notify_all();
    if (worker_.joinable())
      worker_.join();
  }

private:
  using ImageMsg = sensor_msgs::msg::Image::ConstSharedPtr;
  struct Job {
    double stamp = 0;
    std::vector<ImageMsg> msgs;
    Eigen::Matrix4d T_odom_imu;
  };

  void on_image(int i, const ImageMsg &m) {
    const double s = stamp_s(m->header.stamp);
    auto &slot = pending_[s];
    slot.resize(n_cams_);
    slot[i] = m;
    if (std::all_of(slot.begin(), slot.end(), [](const ImageMsg &x) { return (bool)x; })) {
      frames_.push_back({s, slot});
      if (frames_.size() > 8)
        frames_.pop_front();
      for (auto it = pending_.begin(); it != pending_.end() && it->first <= s;)
        it = pending_.erase(it);
    }
    while (pending_.size() > 64) // a camera stopped: do not grow without bound
      pending_.erase(pending_.begin());
  }

  void on_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr &m) {
    const Eigen::Matrix4d T = pose_to_T(m->pose.pose);
    poses_.push_back({stamp_s(m->header.stamp), T});
    if (poses_.size() > 400)
      poses_.pop_front();
    if (!loc_->started()) {
      Eigen::Matrix4d T_map_imu0 = loc_->map().start_T_map_imu;
      if (!start_.empty()) {
        double x = 0, y = 0, yaw = 0;
        char c1, c2;
        std::istringstream ss(start_);
        if (!(ss >> x >> c1 >> y >> c2 >> yaw))
          throw std::runtime_error("maploc: start must be 'x,y,yaw_deg'");
        T_map_imu0 = T_from(Rz(yaw * M_PI / 180.0), Eigen::Vector3d(x, y, T_map_imu0(2, 3)));
      }
      loc_->set_start(T_map_imu0 * T_inv(T));
      RCLCPP_INFO(get_logger(), "known start set from the first VIO pose");
    }
    geometry_msgs::msg::PoseWithCovarianceStamped out;
    out.header.stamp = m->header.stamp;
    out.header.frame_id = map_frame_;
    T_to_pose(loc_->predict(T), out.pose.pose);
    pub_pose_->publish(out);
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

  void tick_reloc() {
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
        for (const auto &m : job.msgs)
          grays.push_back(to_gray(m));
        const LocResult r = loc_->localize(grays, job.T_odom_imu);
        n_try_++;
        if (r.ok) {
          n_ok_++;
          last_fix_ = job.stamp;
        }
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "{\"t\": %.3f, \"ok\": %s, \"why\": \"%s\", \"inliers\": %d, \"matches\": %d, \"landmarks\": %d, "
                      "\"since_fix_s\": %.2f, \"ms\": %.1f, \"ms_orb\": %.1f, \"ms_match\": %.1f, \"ms_pnp\": %.1f, "
                      "\"tilt_deg\": %.2f, \"fixes\": \"%d/%d\"}",
                      job.stamp, r.ok ? "true" : "false", r.why.c_str(), r.inliers, r.n_corr, r.n_landmarks,
                      last_fix_ < 0 ? -1.0 : job.stamp - last_fix_, r.t_total * 1e3, r.t_orb * 1e3, r.t_match * 1e3,
                      r.t_pnp * 1e3, r.tilt_deg, n_ok_, n_try_);
        std_msgs::msg::String st;
        st.data = buf;
        pub_status_->publish(st);
        if (!r.ok || n_try_ % 10 == 0)
          RCLCPP_INFO(get_logger(), "%s", buf);
      } catch (const std::exception &e) {
        RCLCPP_ERROR(get_logger(), "relocalization failed: %s", e.what());
      }
      busy_ = false;
    }
  }

  void publish_tf(const builtin_interfaces::msg::Time &stamp) {
    const Eigen::Matrix4d T = loc_->T_map_odom();
    geometry_msgs::msg::TransformStamped tr;
    tr.header.stamp = stamp;
    tr.header.frame_id = map_frame_;
    tr.child_frame_id = odom_frame_;
    geometry_msgs::msg::Pose p;
    T_to_pose(T, p);
    tr.transform.translation.x = p.position.x;
    tr.transform.translation.y = p.position.y;
    tr.transform.translation.z = p.position.z;
    tr.transform.rotation = p.orientation;
    tfb_->sendTransform(tr);
  }

  void publish_map_points(const RuntimeMap &m) {
    sensor_msgs::msg::PointCloud2 pc;
    pc.header.frame_id = map_frame_;
    pc.header.stamp = now();
    pc.height = 1;
    pc.width = (uint32_t)m.num_points();
    const char *names[4] = {"x", "y", "z", "intensity"};
    for (int i = 0; i < 4; i++) {
      sensor_msgs::msg::PointField f;
      f.name = names[i];
      f.offset = 4 * i;
      f.datatype = sensor_msgs::msg::PointField::FLOAT32;
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
    pub_cloud_->publish(pc);
  }

  std::unique_ptr<Localizer> loc_;
  int n_cams_ = 0;
  std::string pose_topic_, start_, map_frame_, odom_frame_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subs_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tfb_;
  rclcpp::TimerBase::SharedPtr timer_reloc_;

  // executor thread only
  std::map<double, std::vector<ImageMsg>> pending_;
  std::deque<std::pair<double, std::vector<ImageMsg>>> frames_;
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
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<MaplocNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("maploc"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
