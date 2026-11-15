/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_TRACK_KLT_H
#define OV_CORE_TRACK_KLT_H

#include "TrackBase.h"

#include <unordered_map>

namespace ov_core {

/**
 * @brief KLT tracking of features.
 *
 * This is the implementation of a KLT visual frontend for tracking sparse features.
 * We can track either monocular cameras across time (temporally) along with
 * stereo cameras which we also track across time (temporally) but track from left to right
 * to find the stereo correspondence information also.
 * This uses the [calcOpticalFlowPyrLK](https://github.com/opencv/opencv/blob/master/modules/video/src/lkpyramid.cpp)
 * OpenCV function to do the KLT tracking.
 */
class TrackKLT : public TrackBase {

public:
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   * @param fast_threshold FAST detection threshold
   * @param gridx size of grid in the x-direction / u-direction
   * @param gridy size of grid in the y-direction / v-direction
   * @param minpxdist features need to be at least this number pixels away from each other
   */
  explicit TrackKLT(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                    HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist) {}

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief Process new stereo pair of images
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

  /**
   * @brief Process N images as a ring of stereo pairs (0-1, 1-2, ..., N-1 - 0)
   * @param message Contains our timestamp, images, and camera ids (all N cameras)
   *
   * Every camera is tracked temporally exactly once; new features are detected
   * per adjacent pair and matched into the neighbour (seeded from the
   * extrinsics), so one feature id can carry observations in two or three
   * cameras of the same frame.
   */
  void feed_ring(const CameraData &message);

  /**
   * @brief Detects new features in the current image
   * @param img0pyr image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param pts0 vector of currently extracted keypoints in this image
   * @param ids0 vector of feature ids for each currently extracted keypoint
   *
   * Given an image and its currently extracted features, this will try to add new features if needed.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   * Passed images should already be grayscaled.
   */
  void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                   std::vector<size_t> &ids0);

  /**
   * @brief Detects new features in the current stereo pair
   * @param img0pyr left image we will detect features on (first level of pyramid)
   * @param img1pyr right image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param mask1 mask which has what ROI we do not want features in
   * @param cam_id_left first camera sensor id
   * @param cam_id_right second camera sensor id
   * @param pts0 left vector of currently extracted keypoints
   * @param pts1 right vector of currently extracted keypoints
   * @param ids0 left vector of feature ids for each currently extracted keypoint
   * @param ids1 right vector of feature ids for each currently extracted keypoint
   *
   * This does the same logic as the perform_detection_monocular() function, but we also enforce stereo contraints.
   * So we detect features in the left image, and then KLT track them onto the right image.
   * If we have valid tracks, then we have both the keypoint on the left and its matching point in the right image.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   */
  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

  /**
   * @brief Pair detection for feed_ring(): like perform_detection_stereo() but the
   * search in the second image starts from the extrinsics-predicted location
   * (far-point assumption) and a match must survive a forward-backward check
   * and lie within ring_max_angle_deg of the predicted bearing.
   */
  void perform_detection_pair(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                              const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                              std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1,
                              const std::vector<cv::Mat> *warped_pyr = nullptr);

  /**
   * @brief KLT track between two images, and do RANSAC afterwards
   * @param img0pyr starting image pyramid
   * @param img1pyr image pyramid we want to track too
   * @param pts0 starting points
   * @param pts1 points we have tracked
   * @param id0 id of the first camera
   * @param id1 id of the second camera
   * @param mask_out what points had valid tracks
   *
   * This will track features from the first image into the second image.
   * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
   * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
   */
  void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);

  // Ring stereo: accept a cross-camera match only within this many degrees of the
  // extrinsics-predicted bearing (parallax of a point >= 1 m away on a 0.23 m
  // baseline is < 13 deg) and if the backward track returns within this many px.
  double ring_max_angle_deg = 15.0;
  double ring_fb_px = 2.0;

  // Ring stereo cost (2026-10-16, the board: 35 of 44 ms tracking went to the pair detection, which added
  // ~0 features a frame and kept ~3 seen by two cameras). Read from the environment on the first ring frame:
  //   OV_RING_PARALLEL_WARP (1)  render every pair's left image in its right camera and build that pyramid for
  //                              all pairs at once, before the (sequential) pair loop -- same result, less wall time
  //   OV_RING_RETRY_FRAMES (0)   an existing track that failed to match into the right camera is tried again only
  //                              after this many frames (0 = every frame, as before)
  //   OV_RING_RIGHT_FAST (1)     top up the right camera with mono features in each pair; 0 leaves that to the pair
  //                              where the camera is the left one, which extracts in it anyway
  //   OV_RING_PYR_LEVELS (3)     highest pyramid level of the cross-camera KLT: 3 = four levels. The left image
  //                              rendered in the right camera is black outside the overlap (57 % of the frame on
  //                              the 4-camera ring), so with the temporal tracker's pyr_levels (5: 1/32 scale, a
  //                              15 px window spans the image) the coarse levels pull the flow away and the
  //                              forward-backward check rejects nearly every candidate, even one seeded at its
  //                              true location. 2026-11-14, two bags of the board, 4 serial runs each: features
  //                              seen by two cameras 0.6-1.2 -> 14-15 a frame; start-end distance on the harder
  //                              bag 0.28 / 0.28 / 0.28 / 1.02 m against 0.30 / 3.53 / 3.53 m and one divergence;
  //                              no more time. OV_RING_PYR_LEVELS=5 is the behaviour before that day.
  int ring_parallel_warp = -1, ring_retry_frames = 0, ring_right_fast = 1, ring_pyr_levels = -1;
  long ring_frame = 0;
  std::map<std::pair<size_t, size_t>, std::unordered_map<size_t, long>> ring_retry;
  // per-frame stage times of the pair detection (s) and the running summary logged every 5 s
  double ring_t_fast0 = 0, ring_t_warp = 0, ring_t_klt = 0, ring_t_fast1 = 0;
  struct RingStats {
    int frames = 0;
    double features = 0, shared = 0, detected = 0, t_det = 0, t_fast0 = 0, t_warp = 0, t_klt = 0, t_fast1 = 0, t0 = -1;
  } ring_stats;

  // Active perception, SLAM side (NV_SIM paper, 2026-11-14): more triangulated landmarks towards the goal. Read from the
  // environment on the first ring frame:
  //   OV_GOAL_FILE    rows "t gx gy gz": the goal direction in the IMU frame against the camera clock (nearest row within
  //                   0.2 s is used); unset = goal-directed density off
  //   OV_GOAL_DEG (30)       half-angle of the goal sector
  //   OV_GOAL_FEATS (40)     per left camera of a pair, top the goal sector up to this many features with a lower FAST
  //                          threshold and OV_GOAL_MIN_PX (5) spacing, and match them (and the sector's existing tracks)
  //                          across even when the camera needs no new features
  //   OV_GOAL_DEPTHS ("")    comma-separated depth hypotheses (m), e.g. "2,1,0.6,0.4": a candidate the far-point seed did
  //                          not match is seeded again at the projection of the point at each depth; an epipolar gate
  //                          (OV_GOAL_EPI_DEG, 1.5) replaces the 15 deg bearing gate for these. Applied in the goal sector,
  //                          or to every candidate with OV_GOAL_HYP_ALL=1 (the compute-matched uniform control)
  //   OV_GOAL_STATS          append one line per frame: t, sector features, new sector features, stereo matches from
  //                          the far-point seed, from depth hypotheses, goal-stage time (ms), ring detection time (ms)
  int goal_on = -1, goal_feats = 40, goal_min_px = 5, goal_hyp_all = 0;
  double goal_cos = 0.866, goal_epi_sin = 0.026;
  std::vector<double> goal_depths;
  std::vector<std::pair<double, Eigen::Vector3d>> goal_track;
  Eigen::Vector3d goal_I = Eigen::Vector3d::Zero();
  bool goal_valid = false;
  double ring_t_goal = 0;
  int goal_n_roi = 0, goal_n_new = 0, goal_n_far = 0, goal_n_hyp = 0;
  FILE *goal_stats = nullptr;
  bool in_goal(size_t cam, const cv::Point2f &pt) const;
  // per camera, the unit bearing of every 4th pixel (lazily built), so that in_goal needs no undistortion
  mutable std::map<size_t, cv::Mat> goal_bearings;
  mutable std::mutex goal_bearings_mtx;
  void goal_setup();
  // Allocation map (NV_SIM, 2026-11-15; generalises the goal sector): a per-frame weight over the directions OV_ALLOC_DIRS (rows
  // "x y z", IMU frame; HEALPix nside 4 centres) from OV_ALLOC_FILE (rows "t w_0 .. w_{n-1}", camera clock, nearest row within
  // 0.2 s). Direction d is topped up to floor(target_d), target_d = OV_ALLOC_FEATS * w_d / sum(w), tracked features (counted per camera; FAST at half
  // the threshold, OV_GOAL_MIN_PX spacing); directions with target_d >= 1 are "active": their candidates are matched across
  // cameras also when the camera needs no new features, and get the depth hypotheses (OV_GOAL_DEPTHS). Same budget, different
  // allocation: the policies differ only in w.
  int alloc_on = 0, alloc_feats = 60;
  std::vector<Eigen::Vector3d> alloc_dirs;
  std::vector<std::pair<double, std::vector<float>>> alloc_track;
  std::vector<float> alloc_target;
  bool alloc_valid = false;
  mutable std::map<size_t, cv::Mat> alloc_idx;
  int alloc_dir(size_t cam, const cv::Point2f &pt) const;
  const cv::Mat *alloc_table(size_t cam) const;   // the per-camera direction index of every 4th pixel (built once, locked once)

  /// Ring stereo: per pair, remap tables that render the LEFT image in the RIGHT camera's projection (far-point assumption)
  std::map<std::pair<size_t, size_t>, std::pair<cv::Mat, cv::Mat>> ring_maps;
  std::mutex ring_maps_mtx;
  const std::pair<cv::Mat, cv::Mat> &get_ring_map(size_t cam_id_left, size_t cam_id_right, int rows, int cols);

  // Parameters for our FAST grid detector
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist;

  // How many pyramid levels to track
  int pyr_levels = 5;
  cv::Size win_size = cv::Size(15, 15);

  // Last set of image pyramids
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
