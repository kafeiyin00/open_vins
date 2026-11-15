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

#include <cstdio>
#include <fstream>
#include <sstream>
#include "TrackKLT.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <typeinfo>

#include <cmath>
#include <set>

#include "Grider_FAST.h"
#include "Grider_GRID.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"

using namespace ov_core;

// Per-cell feature budget of the grid extraction, and the downsampled mask that
// says which cells to skip. A cell is skipped only when the mask covers all of it
// (its INTER_AREA average is 255): Grider_GRID checks the mask per pixel, so cells
// the mask edge crosses can still give features. The budget is spread over the
// open cells only, else a mask that hides most of the image (fisheye, keeping a
// central disc) leaves a handful of features per camera -- below the 10 that
// RANSAC needs, so tracking never starts. Without a mask this is the stock
// num_features / (grid_x * grid_y) + 1.
static int grid_budget(const cv::Mat &mask, const cv::Size &size_grid, int num_features, cv::Mat &mask_grid) {
  cv::resize(mask, mask_grid, size_grid, 0.0, 0.0, cv::INTER_AREA);
  int open = std::max(1, cv::countNonZero(mask_grid != 255));
  return (int)((double)num_features / (double)open) + 1;
}

void TrackKLT::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // Preprocessing steps that we do not parallelize
  // NOTE: DO NOT PARALLELIZE THESE!
  // NOTE: These seem to be much slower if you parallelize them...
  rT1 = boost::posix_time::microsec_clock::local_time();
  size_t num_images = message.images.size();
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat img;
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(msg_id), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
      clahe->apply(message.images.at(msg_id), img);
    } else {
      img = message.images.at(msg_id);
    }

    // Extract image pyramid
    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);

    // Save!
    img_curr[cam_id] = img;
    img_pyramid_curr[cam_id] = imgpyr;
  }

  // Either call our stereo or monocular version
  // If we are doing binocular tracking, then we should parallize our tracking
  if (num_images == 1) {
    feed_monocular(message, 0);
  } else if (num_images == 2 && use_stereo) {
    feed_stereo(message, 0, 1);
  } else if (num_images > 2 && use_stereo) {
    feed_ring(message);
  } else if (!use_stereo) {
    parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
                    for (int i = range.start; i < range.end; i++) {
                      feed_monocular(message, i);
                    }
                  }));
  } else {
    PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
    std::exit(EXIT_FAILURE);
  }
}

void TrackKLT::feed_monocular(const CameraData &message, size_t msg_id) {

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Get our image objects for this image
  cv::Mat img = img_curr.at(cam_id);
  std::vector<cv::Mat> imgpyr = img_pyramid_curr.at(cam_id);
  cv::Mat mask = message.masks.at(msg_id);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id].empty()) {
    // Detect new features
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    perform_detection_monocular(imgpyr, mask, good_left, good_ids_left);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  int pts_before_detect = (int)pts_last[cam_id].size();
  auto pts_left_old = pts_last[cam_id];
  auto ids_left_old = ids_last[cam_id];
  perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

  // Lets track temporally
  perform_matching(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll);
  assert(pts_left_new.size() == ids_left_old.size());
  rT4 = boost::posix_time::microsec_clock::local_time();

  // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
        (int)pts_left_new.at(i).pt.y >= img.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
      continue;
    // If it is a good track, and also tracked from left to right
    if (mask_ll[i]) {
      good_left.push_back(pts_left_new[i]);
      good_ids_left.push_back(ids_left_old[i]);
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
  }
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%zu detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)pts_last[cam_id].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void TrackKLT::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Get our image objects for this image
  cv::Mat img_left = img_curr.at(cam_id_left);
  cv::Mat img_right = img_curr.at(cam_id_right);
  std::vector<cv::Mat> imgpyr_left = img_pyramid_curr.at(cam_id_left);
  std::vector<cv::Mat> imgpyr_right = img_pyramid_curr.at(cam_id_right);
  cv::Mat mask_left = message.masks.at(msg_id_left);
  cv::Mat mask_right = message.masks.at(msg_id_right);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
    // Track into the new image
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;
    perform_detection_stereo(imgpyr_left, imgpyr_right, mask_left, mask_right, cam_id_left, cam_id_right, good_left, good_right,
                             good_ids_left, good_ids_right);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  int pts_before_detect = (int)pts_last[cam_id_left].size();
  auto pts_left_old = pts_last[cam_id_left];
  auto pts_right_old = pts_last[cam_id_right];
  auto ids_left_old = ids_last[cam_id_left];
  auto ids_right_old = ids_last[cam_id_right];
  perform_detection_stereo(img_pyramid_last[cam_id_left], img_pyramid_last[cam_id_right], img_mask_last[cam_id_left],
                           img_mask_last[cam_id_right], cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old,
                           ids_right_old);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll, mask_rr;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
  std::vector<cv::KeyPoint> pts_right_new = pts_right_old;

  // Lets track temporally
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    perform_matching(img_pyramid_last[is_left ? cam_id_left : cam_id_right], is_left ? imgpyr_left : imgpyr_right,
                                     is_left ? pts_left_old : pts_right_old, is_left ? pts_left_new : pts_right_new,
                                     is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                                     is_left ? mask_ll : mask_rr);
                  }
                }));
  rT4 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // left to right matching
  // TODO: we should probably still do this to reject outliers
  // TODO: maybe we should collect all tracks that are in both frames and make they pass this?
  // std::vector<uchar> mask_lr;
  // perform_matching(imgpyr_left, imgpyr_right, pts_left_new, pts_right_new, cam_id_left, cam_id_right, mask_lr);
  rT5 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // If any of our masks are empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty() && mask_rr.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left].clear();
    pts_last[cam_id_right].clear();
    ids_last[cam_id_left].clear();
    ids_last[cam_id_right].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > img_left.cols ||
        (int)pts_left_new.at(i).pt.y > img_left.rows)
      continue;
    // See if we have the same feature in the right
    bool found_right = false;
    size_t index_right = 0;
    for (size_t n = 0; n < ids_right_old.size(); n++) {
      if (ids_left_old.at(i) == ids_right_old.at(n)) {
        found_right = true;
        index_right = n;
        break;
      }
    }
    // If it is a good track, and also tracked from left to right
    // Else track it as a mono feature in just the left image
    if (mask_ll[i] && found_right && mask_rr[index_right]) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
          (int)pts_right_new.at(index_right).pt.x >= img_right.cols || (int)pts_right_new.at(index_right).pt.y >= img_right.rows)
        continue;
      good_left.push_back(pts_left_new.at(i));
      good_right.push_back(pts_right_new.at(index_right));
      good_ids_left.push_back(ids_left_old.at(i));
      good_ids_right.push_back(ids_right_old.at(index_right));
      // PRINT_DEBUG("adding to stereo - %u , %u\n", ids_left_old.at(i), ids_right_old.at(index_right));
    } else if (mask_ll[i]) {
      good_left.push_back(pts_left_new.at(i));
      good_ids_left.push_back(ids_left_old.at(i));
      // PRINT_DEBUG("adding to left - %u \n",ids_left_old.at(i));
    }
  }

  // Loop through all right points
  for (size_t i = 0; i < pts_right_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= img_right.cols ||
        (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;
    // See if we have the same feature in the right
    bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_right_old.at(i)) != good_ids_right.end());
    // If it has not already been added as a good feature, add it as a mono track
    if (mask_rr[i] && !added_already) {
      good_right.push_back(pts_right_new.at(i));
      good_ids_right.push_back(ids_right_old.at(i));
      // PRINT_DEBUG("adding to right - %u \n", ids_right_old.at(i));
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
  }
  for (size_t i = 0; i < good_right.size(); i++) {
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
  }
  rT6 = boost::posix_time::microsec_clock::local_time();

  //  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)pts_last[cam_id_left].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", (rT5 - rT4).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT6 - rT5).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT6 - rT1).total_microseconds() * 1e-6);
}

void TrackKLT::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less than grid_px_size points away then existing features
  cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                      (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x);
    int y_grid = std::floor(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  // If we don't need any features, just return
  double min_feat_percent = 0.50;
  int num_featsneeded = num_features - (int)pts0.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
    return;

  // This is old extraction code that would extract from the whole image
  // This can be slow as this will recompute extractions for grid areas that we have max features already
  // std::vector<cv::KeyPoint> pts0_ext;
  // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // We also check a downsampled mask such that we don't extract in areas where it is all masked!
  cv::Mat mask0_grid;
  int num_features_grid = grid_budget(mask0, size_grid, num_features, mask0_grid);

  // Create grids we need to extract from and then extract our features (use fast with griding)
  int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
  std::vector<std::pair<int, int>> valid_locs;
  for (int x = 0; x < grid_2d_grid.cols; x++) {
    for (int y = 0; y < grid_2d_grid.rows; y++) {
      if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
        valid_locs.emplace_back(x, y);
      }
    }
  }
  std::vector<cv::KeyPoint> pts0_ext;
  Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    // Check that it is in bounds
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
      continue;
    // See if there is a point at this location
    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else lets add it!
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Loop through and record only ones that are valid
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
  // NOTE: this is due to the fact that we select update features based on feat id
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
  // NOTE: not sure how to remove... maybe a better way?
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    pts0.push_back(kpts0_new.at(i));
    // move id foward and append this new point
    size_t temp = ++currid;
    ids0.push_back(temp);
  }
}

void TrackKLT::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size_close0((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                       (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
  float size_x0 = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y0 = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid0(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x0);
    int y_grid = std::floor(kpt.pt.y / size_y0);
    if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  double min_feat_percent = 0.50;
  int num_featsneeded_0 = num_features - (int)pts0.size();

  // LEFT: if we need features we should extract them in the current frame
  // LEFT: we will also try to track them from this frame over to the right frame
  // LEFT: in the case that we have two features that are the same, then we should merge them
  if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts0_ext;
    // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    int num_features_grid = grid_budget(mask0, size_grid0, num_features, mask0_grid);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid0.cols; x++) {
      for (int y = 0; y < grid_2d_grid0.rows; y++) {
        if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
      kpts0_new.push_back(kpt);
      pts0_new.push_back(kpt.pt);
    }

    // TODO: Project points from the left frame into the right frame
    // TODO: This will not work for large baseline systems.....
    // TODO: If we had some depth estimates we could do a better projection
    // TODO: Or project and search along the epipolar line??
    std::vector<cv::KeyPoint> kpts1_new;
    std::vector<cv::Point2f> pts1_new;
    kpts1_new = kpts0_new;
    pts1_new = pts0_new;

    // If we have points, do KLT tracking to get the valid projections into the right image
    if (!pts0_new.empty()) {

      // Do our KLT tracking from the left to the right frame of reference
      // NOTE: we have a pretty big window size here since our projection might be bad
      // NOTE: but this might cause failure in cases of repeated textures (eg. checkerboard)
      std::vector<uchar> mask;
      // perform_matching(img0pyr, img1pyr, kpts0_new, kpts1_new, cam_id_left, cam_id_right, mask);
      std::vector<float> error;
      cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
      cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask, error, win_size, pyr_levels, term_crit,
                               cv::OPTFLOW_USE_INITIAL_FLOW);

      // Loop through and record only ones that are valid
      for (size_t i = 0; i < pts0_new.size(); i++) {

        // Check to see if the feature is out of bounds (oob) in either image
        bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img0pyr.at(0).cols || (int)pts0_new.at(i).y < 0 ||
                         (int)pts0_new.at(i).y >= img0pyr.at(0).rows);
        bool oob_right = ((int)pts1_new.at(i).x < 0 || (int)pts1_new.at(i).x >= img1pyr.at(0).cols || (int)pts1_new.at(i).y < 0 ||
                          (int)pts1_new.at(i).y >= img1pyr.at(0).rows);

        // Check to see if it there is already a feature in the right image at this location
        //  1) If this is not already in the right image, then we should treat it as a stereo
        //  2) Otherwise we will treat this as just a monocular track of the feature
        // TODO: we should check to see if we can combine this new feature and the one in the right
        // TODO: seems if reject features which overlay with right features already we have very poor tracking perf
        if (!oob_left && !oob_right && mask[i] == 1) {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          kpts1_new.at(i).pt = pts1_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          pts1.push_back(kpts1_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
          ids1.push_back(temp);
        } else if (!oob_left) {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
        }
      }
    }
  }

  // RIGHT: Now summarise the number of tracks in the right image
  // RIGHT: We will try to extract some monocular features if we have the room
  // RIGHT: This will also remove features if there are multiple in the same location
  cv::Size size_close1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
  float size_x1 = (float)img1pyr.at(0).cols / (float)grid_x;
  float size_y1 = (float)img1pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid1(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
  cv::Mat mask1_updated = mask0.clone();
  it0 = pts1.begin();
  it1 = ids1.begin();
  while (it0 != pts1.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img1pyr.at(0).cols - edge || y < edge || y >= img1pyr.at(0).rows - edge) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x1);
    int y_grid = std::floor(kpt.pt.y / size_y1);
    if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    // NOTE: if it is *not* a stereo point, then we will not delete the feature
    // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
    bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
    if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask1.at<uint8_t>(y, x) > 127) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img1pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img1pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // RIGHT: if we need features we should extract them in the current frame
  // RIGHT: note that we don't track them to the left as we already did left->right tracking above
  int num_featsneeded_1 = num_features - (int)pts1.size();
  if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts1_ext;
    // Grider_FAST::perform_griding(img1pyr.at(0), mask1_updated, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask1_grid;
    int num_features_grid = grid_budget(mask1, size_grid1, num_features, mask1_grid);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid1.cols; x++) {
      for (int y = 0; y < grid_2d_grid1.rows; y++) {
        if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts1_ext;
    Grider_GRID::perform_griding(img1pyr.at(0), mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    for (auto &kpt : pts1_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      pts1.push_back(kpt);
      size_t temp = ++currid;
      ids1.push_back(temp);
      grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
    }
  }
}

void TrackKLT::perform_detection_pair(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1,
                                        const std::vector<cv::Mat> *warped_pyr) {
  auto now = [] { return boost::posix_time::microsec_clock::local_time(); };
  auto secs = [](const boost::posix_time::ptime &a, const boost::posix_time::ptime &b) { return (b - a).total_microseconds() * 1e-6; };
  boost::posix_time::ptime tp0 = now();

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size_close0((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                       (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
  float size_x0 = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y0 = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid0(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x0);
    int y_grid = std::floor(kpt.pt.y / size_y0);
    if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  double min_feat_percent = 0.50;
  int num_featsneeded_0 = num_features - (int)pts0.size();

  // LEFT: if we need features we should extract them in the current frame
  // LEFT: we will also try to track them from this frame over to the right frame
  // LEFT: in the case that we have two features that are the same, then we should merge them
  bool need_std0 = num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features));
  bool goal_here = (goal_valid && goal_on > 0) || (alloc_valid && alloc_on > 0);
  if (need_std0 || goal_here) {
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    if (need_std0) {

      // This is old extraction code that would extract from the whole image
      // This can be slow as this will recompute extractions for grid areas that we have max features already
      // std::vector<cv::KeyPoint> pts0_ext;
      // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

      // We also check a downsampled mask such that we don't extract in areas where it is all masked!
      cv::Mat mask0_grid;
      int num_features_grid = grid_budget(mask0, size_grid0, num_features, mask0_grid);

      // Create grids we need to extract from and then extract our features (use fast with griding)
      int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
      std::vector<std::pair<int, int>> valid_locs;
      for (int x = 0; x < grid_2d_grid0.cols; x++) {
        for (int y = 0; y < grid_2d_grid0.rows; y++) {
          if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
            valid_locs.emplace_back(x, y);
          }
        }
      }
      std::vector<cv::KeyPoint> pts0_ext;
      Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

      // Now, reject features that are close a current feature
      for (auto &kpt : pts0_ext) {
        // Check that it is in bounds
        int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
        int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
          continue;
        // See if there is a point at this location
        if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
          continue;
        // Else lets add it!
        grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
        kpts0_new.push_back(kpt);
        pts0_new.push_back(kpt.pt);
      }
    }

    // Active perception: top the goal sector up with features (lower FAST threshold, finer spacing), also when the camera
    // needs no new features, so that the sector is matched across and triangulated (see TrackKLT.h, OV_GOAL_*)
    if (goal_here && alloc_on > 0) {
      // allocation map: top every direction up to its target (see TrackKLT.h)
      boost::posix_time::ptime tg0 = now();
      const cv::Mat &im = img0pyr.at(0);
      const cv::Mat *tab = alloc_table(cam_id_left);
      auto alloc_dir = [tab](size_t, const cv::Point2f &pt) -> int {
        if (!tab)
          return -1;
        int x = std::min(std::max((int)(pt.x / 4.0f), 0), tab->cols - 1), y = std::min(std::max((int)(pt.y / 4.0f), 0), tab->rows - 1);
        return tab->at<int16_t>(y, x);
      };
      std::vector<int> cnt(alloc_dirs.size(), 0);
      for (const auto &k : pts0) {
        int d = alloc_dir(cam_id_left, k.pt);
        if (d >= 0) cnt[d]++;
      }
      for (const auto &k : kpts0_new) {
        int d = alloc_dir(cam_id_left, k.pt);
        if (d >= 0) cnt[d]++;
      }
      int x0 = im.cols, y0 = im.rows, x1 = -1, y1 = -1;              // bounding box of the grid cells whose direction needs features
      for (int y = 8; y < im.rows; y += 16)
        for (int x = 8; x < im.cols; x += 16) {
          int d = alloc_dir(cam_id_left, cv::Point2f((float)x, (float)y));
          if (d >= 0 && alloc_target[d] >= 1.0f && cnt[d] < (int)alloc_target[d])
            x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
        }
      int n_add = 0;
      if (x1 >= x0) {
        cv::Rect roi(std::max(0, x0 - 16), std::max(0, y0 - 16), 0, 0);
        roi.width = std::min(im.cols, x1 + 16) - roi.x;
        roi.height = std::min(im.rows, y1 + 16) - roi.y;
        std::vector<cv::KeyPoint> fast;
        cv::FAST(im(roi), fast, std::max(5, threshold / 2), true);
        for (auto &kp : fast)
          kp.pt += cv::Point2f((float)roi.x, (float)roi.y);
        std::sort(fast.begin(), fast.end(), [](const cv::KeyPoint &p, const cv::KeyPoint &q) { return p.response > q.response; });
        cv::Mat occ = cv::Mat::zeros(im.rows / goal_min_px + 1, im.cols / goal_min_px + 1, CV_8UC1);
        auto mark = [&](const cv::Point2f &q) {
          int gx = (int)(q.x / goal_min_px), gy = (int)(q.y / goal_min_px);
          if (gx >= 0 && gy >= 0 && gx < occ.cols && gy < occ.rows)
            occ.at<uint8_t>(gy, gx) = 255;
        };
        for (const auto &k : pts0)
          mark(k.pt);
        for (const auto &k : kpts0_new)
          mark(k.pt);
        for (const auto &kp : fast) {
          if (n_add >= alloc_feats)
            break;
          int x = (int)kp.pt.x, y = (int)kp.pt.y;
          if (x < 10 || y < 10 || x >= im.cols - 10 || y >= im.rows - 10)
            continue;
          if (mask0.at<uint8_t>(y, x) > 127)
            continue;
          int gx = x / goal_min_px, gy = y / goal_min_px;
          if (occ.at<uint8_t>(gy, gx) > 127)
            continue;
          int d = alloc_dir(cam_id_left, kp.pt);
          if (d < 0 || alloc_target[d] < 1.0f || cnt[d] >= (int)alloc_target[d])
            continue;
          occ.at<uint8_t>(gy, gx) = 255;
          kpts0_new.push_back(kp);
          pts0_new.push_back(kp.pt);
          cnt[d]++;
          n_add++;
        }
      }
      int n_act = 0;
      for (size_t d = 0; d < cnt.size(); d++)
        if (alloc_target[d] >= 1.0f)
          n_act += cnt[d];
      goal_n_roi += n_act;
      goal_n_new += n_add;
      ring_t_goal += secs(tg0, now());
    } else if (goal_here) {
      boost::posix_time::ptime tg0 = now();
      const cv::Mat &im = img0pyr.at(0);
      int n_roi = 0, n_add = 0;
      for (const auto &k : pts0)
        if (in_goal(cam_id_left, k.pt))
          n_roi++;
      for (const auto &k : kpts0_new)
        if (in_goal(cam_id_left, k.pt))
          n_roi++;
      if (n_roi < goal_feats) {
        int x0 = im.cols, y0 = im.rows, x1 = -1, y1 = -1;
        for (int y = 8; y < im.rows; y += 16)
          for (int x = 8; x < im.cols; x += 16)
            if (in_goal(cam_id_left, cv::Point2f((float)x, (float)y))) {
              x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
            }
        if (x1 >= x0) {
          cv::Rect roi(std::max(0, x0 - 16), std::max(0, y0 - 16), 0, 0);
          roi.width = std::min(im.cols, x1 + 16) - roi.x;
          roi.height = std::min(im.rows, y1 + 16) - roi.y;
          std::vector<cv::KeyPoint> fast;
          cv::FAST(im(roi), fast, std::max(5, threshold / 2), true);
          for (auto &kp : fast)
            kp.pt += cv::Point2f((float)roi.x, (float)roi.y);
          std::sort(fast.begin(), fast.end(), [](const cv::KeyPoint &p, const cv::KeyPoint &q) { return p.response > q.response; });
          cv::Mat occ = cv::Mat::zeros(im.rows / goal_min_px + 1, im.cols / goal_min_px + 1, CV_8UC1);
          auto mark = [&](const cv::Point2f &q) {
            int gx = (int)(q.x / goal_min_px), gy = (int)(q.y / goal_min_px);
            if (gx >= 0 && gy >= 0 && gx < occ.cols && gy < occ.rows)
              occ.at<uint8_t>(gy, gx) = 255;
          };
          for (const auto &k : pts0)
            mark(k.pt);
          for (const auto &k : kpts0_new)
            mark(k.pt);
          for (const auto &kp : fast) {
            if (n_roi + n_add >= goal_feats)
              break;
            int x = (int)kp.pt.x, y = (int)kp.pt.y;
            if (x < 10 || y < 10 || x >= im.cols - 10 || y >= im.rows - 10)
              continue;
            if (mask0.at<uint8_t>(y, x) > 127)
              continue;
            int gx = x / goal_min_px, gy = y / goal_min_px;
            if (occ.at<uint8_t>(gy, gx) > 127 || !in_goal(cam_id_left, kp.pt))
              continue;
            occ.at<uint8_t>(gy, gx) = 255;
            kpts0_new.push_back(kp);
            pts0_new.push_back(kp.pt);
            n_add++;
          }
        }
      }
      goal_n_roi += n_roi + n_add;
      goal_n_new += n_add;
      ring_t_goal += secs(tg0, now());
    }

    ring_t_fast0 += secs(tp0, now());
    // Cross-camera matching. Candidates are the NEW left detections plus every
    // existing left track that has no observation in the right camera yet.
    // The right image is 90 deg away, so instead of tracking between the two
    // raw images the left image is first re-rendered in the right camera's
    // projection (rotation only, far-point assumption): the same world patch
    // then looks alike in both, and pyramidal KLT from the predicted location
    // only has to absorb the parallax. A forward-backward check and a bearing
    // gate against the extrinsics keep the false matches out.
    auto rot_it = pair_rotations.find({cam_id_left, cam_id_right});
    bool have_rot = (rot_it != pair_rotations.end());
    std::vector<cv::Point2f> cand_pts0;   // location in the left image
    std::vector<int> cand_new;            // index into kpts0_new, or -1 for an existing track
    std::vector<size_t> cand_ids;         // id for existing tracks
    for (size_t i = 0; i < kpts0_new.size(); i++) {
      cand_pts0.push_back(kpts0_new.at(i).pt);
      cand_new.push_back((int)i);
      cand_ids.push_back(0);
    }
    if (have_rot) {
      std::set<size_t> ids_right(ids1.begin(), ids1.end());
      auto &retry = ring_retry[{cam_id_left, cam_id_right}];
      for (size_t i = 0; i < pts0.size(); i++) {
        if (ids_right.count(ids0.at(i)))
          continue;
        if (!need_std0 && !in_goal(cam_id_left, pts0.at(i).pt))
          continue;
        if (ring_retry_frames > 0) {
          auto r = retry.find(ids0.at(i));
          if (r != retry.end() && r->second > ring_frame)
            continue;
        }
        cand_pts0.push_back(pts0.at(i).pt);
        cand_new.push_back(-1);
        cand_ids.push_back(ids0.at(i));
      }
    }
    std::vector<cv::Point2f> cand_pts1(cand_pts0.size());
    std::vector<uchar> seed_ok(cand_pts0.size(), 0);
    std::vector<Eigen::Vector3d> bearing_pred(cand_pts0.size(), Eigen::Vector3d(0, 0, 1));
    int edge_r = 10;
    for (size_t i = 0; i < cand_pts0.size(); i++) {
      if (!have_rot) {
        cand_pts1.at(i) = cand_pts0.at(i);
        seed_ok[i] = 1;
        continue;
      }
      Eigen::Vector2f n0 = camera_calib.at(cam_id_left)->undistort_f(Eigen::Vector2f(cand_pts0.at(i).x, cand_pts0.at(i).y));
      Eigen::Vector3d b0(n0(0), n0(1), 1.0);
      b0.normalize();
      Eigen::Vector3d b1 = rot_it->second * b0;
      if (b1(2) <= 0.05)
        continue;
      Eigen::Vector2f n1((float)(b1(0) / b1(2)), (float)(b1(1) / b1(2)));
      Eigen::Vector2f p1 = camera_calib.at(cam_id_right)->distort_f(n1);
      if (p1(0) < edge_r || p1(0) >= img1pyr.at(0).cols - edge_r || p1(1) < edge_r || p1(1) >= img1pyr.at(0).rows - edge_r)
        continue;
      if ((int)mask1.at<uint8_t>((int)p1(1), (int)p1(0)) > 127)
        continue;
      cand_pts1.at(i) = cv::Point2f(p1(0), p1(1));
      bearing_pred.at(i) = b1;
      seed_ok[i] = 1;
    }

    std::vector<uchar> match_ok(cand_pts0.size(), 0);
    int n_seeded_total = 0, n_klt = 0, n_fb = 0, n_ang = 0;
    if (!cand_pts0.empty()) {
      std::vector<cv::Point2f> src_pts, dst_pts;
      std::vector<size_t> src_idx;
      for (size_t i = 0; i < cand_pts0.size(); i++) {
        if (!seed_ok[i])
          continue;
        n_seeded_total++;
        src_idx.push_back(i);
        // in the warped-left image the feature sits at its predicted right-image location
        src_pts.push_back(have_rot ? cand_pts1.at(i) : cand_pts0.at(i));
        dst_pts.push_back(cand_pts1.at(i));
      }
      if (!src_pts.empty()) {
        boost::posix_time::ptime tw0 = now();
        std::vector<cv::Mat> srcpyr;
        if (have_rot && warped_pyr && !warped_pyr->empty()) {
          srcpyr = *warped_pyr;
        } else if (have_rot) {
          const auto &maps = get_ring_map(cam_id_left, cam_id_right, img1pyr.at(0).rows, img1pyr.at(0).cols);
          cv::Mat warped;
          cv::remap(img0pyr.at(0), warped, maps.first, maps.second, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
          cv::buildOpticalFlowPyramid(warped, srcpyr, win_size, ring_pyr_levels >= 0 ? ring_pyr_levels : pyr_levels);
        } else {
          srcpyr = img0pyr;
        }
        // the rendered left image has content in the overlap only: see ring_pyr_levels
        int klt_levels = (have_rot && ring_pyr_levels >= 0) ? ring_pyr_levels : pyr_levels;
        boost::posix_time::ptime tw1 = now();
        ring_t_warp += secs(tw0, tw1);
        std::vector<uchar> st_f, st_b;
        std::vector<float> err_f, err_b;
        cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(srcpyr, img1pyr, src_pts, dst_pts, st_f, err_f, win_size, klt_levels, term_crit,
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        std::vector<cv::Point2f> back_pts = src_pts;
        cv::calcOpticalFlowPyrLK(img1pyr, srcpyr, dst_pts, back_pts, st_b, err_b, win_size, klt_levels, term_crit,
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        ring_t_klt += secs(tw1, now());
        double cos_max = std::cos(ring_max_angle_deg * M_PI / 180.0);
        for (size_t k = 0; k < src_pts.size(); k++) {
          size_t i = src_idx[k];
          if (!st_f[k])
            continue;
          n_klt++;
          if (!st_b[k] || cv::norm(back_pts.at(k) - src_pts.at(k)) > ring_fb_px)
            continue;
          n_fb++;
          cv::Point2f p1 = dst_pts.at(k);
          if (p1.x < 0 || p1.y < 0 || (int)p1.x >= img1pyr.at(0).cols || (int)p1.y >= img1pyr.at(0).rows)
            continue;
          if (have_rot) {
            Eigen::Vector2f n1 = camera_calib.at(cam_id_right)->undistort_f(Eigen::Vector2f(p1.x, p1.y));
            Eigen::Vector3d bm(n1(0), n1(1), 1.0);
            bm.normalize();
            if (bm.dot(bearing_pred.at(i)) < cos_max)
              continue;
          }
          n_ang++;
          match_ok[i] = 1;
          cand_pts1.at(i) = p1;
        }
        goal_n_far += n_ang;
        // Active perception: depth hypotheses. A candidate the far-point seed missed is seeded again where the point
        // would appear at each hypothesised depth (near points are tens of px off the far-point prediction); an epipolar
        // gate replaces the bearing gate for these matches.
        if (have_rot && !goal_depths.empty() && (goal_here || goal_hyp_all) && ring_R_ItoC.count(cam_id_left) &&
            ring_R_ItoC.count(cam_id_right)) {
          boost::posix_time::ptime th0 = now();
          const Eigen::Matrix3d &Rlr = rot_it->second;
          Eigen::Vector3d t_lr = ring_p_IinC.at(cam_id_right) - Rlr * ring_p_IinC.at(cam_id_left);   // left origin in the right frame
          double cos30 = std::cos(30.0 * M_PI / 180.0);
          for (double dh : goal_depths) {
            std::vector<cv::Point2f> s2, d2;
            std::vector<size_t> i2;
            std::vector<Eigen::Vector3d> binf2;
            for (size_t i = 0; i < cand_pts0.size(); i++) {
              if (!seed_ok[i] || match_ok[i])
                continue;
              if (!goal_hyp_all && !in_goal(cam_id_left, cand_pts0.at(i)))
                continue;
              Eigen::Vector2f n0 = camera_calib.at(cam_id_left)->undistort_f(Eigen::Vector2f(cand_pts0.at(i).x, cand_pts0.at(i).y));
              Eigen::Vector3d b0(n0(0), n0(1), 1.0);
              b0.normalize();
              Eigen::Vector3d q = Rlr * b0 + t_lr / dh;
              if (q(2) <= 0.05)
                continue;
              Eigen::Vector2f ph = camera_calib.at(cam_id_right)->distort_f(Eigen::Vector2f((float)(q(0) / q(2)), (float)(q(1) / q(2))));
              if (ph(0) < edge_r || ph(0) >= img1pyr.at(0).cols - edge_r || ph(1) < edge_r || ph(1) >= img1pyr.at(0).rows - edge_r)
                continue;
              if ((int)mask1.at<uint8_t>((int)ph(1), (int)ph(0)) > 127)
                continue;
              s2.push_back(cand_pts1.at(i));
              d2.push_back(cv::Point2f(ph(0), ph(1)));
              i2.push_back(i);
              binf2.push_back(Rlr * b0);
            }
            if (s2.empty())
              continue;
            std::vector<uchar> sf, sb;
            std::vector<float> ef, eb;
            cv::calcOpticalFlowPyrLK(srcpyr, img1pyr, s2, d2, sf, ef, win_size, klt_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
            std::vector<cv::Point2f> bk = s2;
            cv::calcOpticalFlowPyrLK(img1pyr, srcpyr, d2, bk, sb, eb, win_size, klt_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
            for (size_t k = 0; k < s2.size(); k++) {
              size_t i = i2[k];
              if (match_ok[i] || !sf[k] || !sb[k] || cv::norm(bk.at(k) - s2.at(k)) > ring_fb_px)
                continue;
              cv::Point2f p1 = d2.at(k);
              if (p1.x < 0 || p1.y < 0 || (int)p1.x >= img1pyr.at(0).cols || (int)p1.y >= img1pyr.at(0).rows)
                continue;
              Eigen::Vector2f n1 = camera_calib.at(cam_id_right)->undistort_f(Eigen::Vector2f(p1.x, p1.y));
              Eigen::Vector3d bm(n1(0), n1(1), 1.0);
              bm.normalize();
              Eigen::Vector3d ne = binf2[k].cross(t_lr);
              if (ne.norm() < 1e-12)
                continue;
              ne.normalize();
              if (std::abs(bm.dot(ne)) > goal_epi_sin || bm.dot(binf2[k].normalized()) < cos30)
                continue;
              match_ok[i] = 1;
              cand_pts1.at(i) = p1;
              goal_n_hyp++;
            }
          }
          ring_t_goal += secs(th0, now());
        }
      }
      PRINT_ALL("[RING-PAIR] cam%zu->cam%zu rot=%d: %d new + %d existing candidates, %d seeded, %d klt, %d fb, %d angle -> stereo\n",
                cam_id_left, cam_id_right, (int)have_rot, (int)kpts0_new.size(), (int)(cand_pts0.size() - kpts0_new.size()),
                n_seeded_total, n_klt, n_fb, n_ang);

      // An existing track that did not make it into the right camera waits ring_retry_frames before the next try
      if (have_rot && ring_retry_frames > 0) {
        auto &retry = ring_retry[{cam_id_left, cam_id_right}];
        for (size_t i = 0; i < cand_pts0.size(); i++)
          if (cand_new[i] < 0 && !match_ok[i])
            retry[cand_ids[i]] = ring_frame + ring_retry_frames;
        if (ring_frame % 64 == 0) // forget the ids that are long gone
          for (auto r = retry.begin(); r != retry.end();)
            r = (r->second < ring_frame) ? retry.erase(r) : std::next(r);
      }

      // Record: new features go to the left (and to the right if matched), existing tracks gain a right observation
      for (size_t i = 0; i < cand_pts0.size(); i++) {
        if (cand_new[i] >= 0) {
          cv::KeyPoint kp0 = kpts0_new.at(cand_new[i]);
          bool oob_left = ((int)kp0.pt.x < 0 || (int)kp0.pt.x >= img0pyr.at(0).cols || (int)kp0.pt.y < 0 || (int)kp0.pt.y >= img0pyr.at(0).rows);
          if (oob_left)
            continue;
          size_t temp = ++currid;
          pts0.push_back(kp0);
          ids0.push_back(temp);
          if (match_ok[i]) {
            cv::KeyPoint kp1 = kp0;
            kp1.pt = cand_pts1.at(i);
            pts1.push_back(kp1);
            ids1.push_back(temp);
          }
        } else if (match_ok[i]) {
          cv::KeyPoint kp1;
          kp1.pt = cand_pts1.at(i);
          kp1.size = 7;
          pts1.push_back(kp1);
          ids1.push_back(cand_ids[i]);
        }
      }
    }
  }

  // RIGHT: Now summarise the number of tracks in the right image
  // RIGHT: We will try to extract some monocular features if we have the room
  // RIGHT: This will also remove features if there are multiple in the same location
  cv::Size size_close1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
  float size_x1 = (float)img1pyr.at(0).cols / (float)grid_x;
  float size_y1 = (float)img1pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid1(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
  cv::Mat mask1_updated = mask1.clone();
  it0 = pts1.begin();
  it1 = ids1.begin();
  while (it0 != pts1.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img1pyr.at(0).cols - edge || y < edge || y >= img1pyr.at(0).rows - edge) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x1);
    int y_grid = std::floor(kpt.pt.y / size_y1);
    if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    // NOTE: if it is *not* a stereo point, then we will not delete the feature
    // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
    bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
    if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask1.at<uint8_t>(y, x) > 127) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img1pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img1pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // RIGHT: if we need features we should extract them in the current frame
  // RIGHT: note that we don't track them to the left as we already did left->right tracking above
  boost::posix_time::ptime tr0 = now();
  int num_featsneeded_1 = num_features - (int)pts1.size();
  if (ring_right_fast && num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts1_ext;
    // Grider_FAST::perform_griding(img1pyr.at(0), mask1_updated, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask1_grid;
    int num_features_grid = grid_budget(mask1, size_grid1, num_features, mask1_grid);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid1.cols; x++) {
      for (int y = 0; y < grid_2d_grid1.rows; y++) {
        if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts1_ext;
    Grider_GRID::perform_griding(img1pyr.at(0), mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    for (auto &kpt : pts1_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      pts1.push_back(kpt);
      size_t temp = ++currid;
      ids1.push_back(temp);
      grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
    }
  }
  ring_t_fast1 += secs(tr0, now());
}

const std::pair<cv::Mat, cv::Mat> &TrackKLT::get_ring_map(size_t cam_id_left, size_t cam_id_right, int rows, int cols) {
  std::lock_guard<std::mutex> lck(ring_maps_mtx);
  auto key = std::make_pair(cam_id_left, cam_id_right);
  auto it = ring_maps.find(key);
  if (it != ring_maps.end())
    return it->second;
  // Disk cache (OV_RING_MAP_CACHE=<dir>): the tables depend only on the calibration and the image size, and the
  // per-pixel undistortion below takes ~0.2 s per pair on a Xeon but ~20 s on an RK3588 (70 s before the first frame,
  // paid again at every in-flight restart). The file name carries a hash of everything the tables are built from
  // (both cameras' model and intrinsics, the pair rotation), so a new calibration never loads an old table.
  std::string cache_path;
  if (const char *dir = std::getenv("OV_RING_MAP_CACHE")) {
    uint64_t hash = 1469598103934665603ULL; // FNV-1a
    auto mix = [&hash](const void *p, size_t n) {
      for (size_t i = 0; i < n; i++)
        hash = (hash ^ ((const unsigned char *)p)[i]) * 1099511628211ULL;
    };
    for (size_t id : {cam_id_left, cam_id_right}) {
      const CamBase &cam = *camera_calib.at(id);
      std::string model = typeid(cam).name();
      Eigen::MatrixXd values = camera_calib.at(id)->get_value();
      mix(model.data(), model.size());
      mix(values.data(), sizeof(double) * values.size());
    }
    const Eigen::Matrix3d &R = pair_rotations.at(key);
    mix(R.data(), sizeof(double) * 9);
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)hash);
    cache_path = std::string(dir) + "/ring_" + std::to_string(cam_id_left) + "_" + std::to_string(cam_id_right) + "_" +
                 std::to_string(rows) + "x" + std::to_string(cols) + "_" + hex + ".bin";
    std::ifstream f(cache_path, std::ios::binary);
    int32_t hdr[2] = {0, 0};
    if (f && f.read((char *)hdr, sizeof(hdr)) && hdr[0] == rows && hdr[1] == cols) {
      cv::Mat mx(rows, cols, CV_32FC1), my(rows, cols, CV_32FC1);
      if (f.read((char *)mx.data, sizeof(float) * rows * cols) && f.read((char *)my.data, sizeof(float) * rows * cols)) {
        PRINT_WARNING("[RING]: loaded remap cam%zu->cam%zu from %s\n", cam_id_left, cam_id_right, cache_path.c_str());   // visible at verbosity WARNING
        ring_maps[key] = std::make_pair(mx, my);
        return ring_maps.at(key);
      }
    }
  }
  // For every RIGHT pixel: bearing -> rotate into the LEFT camera -> project. A
  // remap with these tables renders the left image as the right camera would
  // see it if everything were at infinity.
  Eigen::Matrix3d R_LfromR = pair_rotations.at(key).transpose();
  cv::Mat mapx(rows, cols, CV_32FC1, cv::Scalar(-1));
  cv::Mat mapy(rows, cols, CV_32FC1, cv::Scalar(-1));
  auto t0 = boost::posix_time::microsec_clock::local_time();
  for (int v = 0; v < rows; v++) {
    for (int u = 0; u < cols; u++) {
      Eigen::Vector2f n1 = camera_calib.at(cam_id_right)->undistort_f(Eigen::Vector2f((float)u, (float)v));
      Eigen::Vector3d b1(n1(0), n1(1), 1.0);
      b1.normalize();
      Eigen::Vector3d b0 = R_LfromR * b1;
      if (b0(2) <= 0.05)
        continue;
      Eigen::Vector2f p0 = camera_calib.at(cam_id_left)->distort_f(Eigen::Vector2f((float)(b0(0) / b0(2)), (float)(b0(1) / b0(2))));
      if (p0(0) < 0 || p0(0) >= cols - 1 || p0(1) < 0 || p0(1) >= rows - 1)
        continue;
      mapx.at<float>(v, u) = p0(0);
      mapy.at<float>(v, u) = p0(1);
    }
  }
  auto t1 = boost::posix_time::microsec_clock::local_time();
  PRINT_WARNING("[RING]: built remap cam%zu->cam%zu view in %.2f s (no cache hit)\n", cam_id_left, cam_id_right, (t1 - t0).total_microseconds() * 1e-6);
  if (!cache_path.empty()) {
    std::ofstream f(cache_path, std::ios::binary);
    const int32_t hdr[2] = {rows, cols};
    if (f && f.write((const char *)hdr, sizeof(hdr)) && f.write((const char *)mapx.data, sizeof(float) * rows * cols) &&
        f.write((const char *)mapy.data, sizeof(float) * rows * cols)) {
      PRINT_WARNING("[RING]: cached remap cam%zu->cam%zu to %s\n", cam_id_left, cam_id_right, cache_path.c_str());
    } else {
      PRINT_WARNING("[RING]: could not write %s\n", cache_path.c_str());
    }
  }
  ring_maps[key] = std::make_pair(mapx, mapy);
  return ring_maps.at(key);
}

void TrackKLT::feed_ring(const CameraData &message) {

  // N cameras, adjacent pairs (k, k+1 mod N) are stereo pairs
  size_t N = message.images.size();
  std::vector<size_t> cams(N);
  for (size_t k = 0; k < N; k++)
    cams[k] = message.sensor_ids.at(k);
  std::vector<std::unique_lock<std::mutex>> locks;
  for (size_t k = 0; k < N; k++)
    locks.emplace_back(mtx_feeds.at(cams[k]));
  if (ring_parallel_warp < 0) {
    auto env = [](const char *name, int dflt) {
      const char *v = std::getenv(name);
      return v && *v ? std::atoi(v) : dflt;
    };
    ring_parallel_warp = env("OV_RING_PARALLEL_WARP", 1);
    ring_retry_frames = std::max(0, env("OV_RING_RETRY_FRAMES", 0));
    ring_right_fast = env("OV_RING_RIGHT_FAST", 1);
    ring_pyr_levels = std::min(pyr_levels, std::max(0, env("OV_RING_PYR_LEVELS", 3)));
    PRINT_INFO("[RING]: parallel warp %d, retry a failed cross-camera match after %d frames, right-image top-up %d, "
               "cross-camera KLT up to pyramid level %d (temporal %d)\n",
               ring_parallel_warp, ring_retry_frames, ring_right_fast, ring_pyr_levels, pyr_levels);
  }
  ring_frame++;
  ring_t_fast0 = ring_t_warp = ring_t_klt = ring_t_fast1 = 0;
  if (goal_on < 0)
    goal_setup();
  goal_valid = false;
  ring_t_goal = 0;
  goal_n_roi = goal_n_new = goal_n_far = goal_n_hyp = 0;
  alloc_valid = false;
  if (alloc_on > 0) {
    auto it = std::lower_bound(alloc_track.begin(), alloc_track.end(), message.timestamp,
                               [](const std::pair<double, std::vector<float>> &a, double t) { return a.first < t; });
    double best = 1e9;
    const std::vector<float> *w = nullptr;
    for (auto c : {it, it == alloc_track.begin() ? it : std::prev(it)})
      if (c != alloc_track.end() && std::abs(c->first - message.timestamp) < best) {
        best = std::abs(c->first - message.timestamp);
        w = &c->second;
      }
    if (w && best < 0.2) {
      double sw = 0;
      for (float v : *w)
        sw += std::max(0.0f, v);
      if (sw > 0) {
        alloc_target.assign(w->size(), 0.0f);
        for (size_t d = 0; d < w->size(); d++)
          alloc_target[d] = (float)(alloc_feats * std::max(0.0f, (*w)[d]) / sw);
        alloc_valid = true;
      }
    }
  }
  if (goal_on == 2) {
    std::ifstream in(goal_live);
    double t, x, y, z;
    if (in >> t >> x >> y >> z && std::abs(t - message.timestamp) < 0.5) {
      goal_I = Eigen::Vector3d(x, y, z);
      goal_valid = goal_I.norm() > 1e-6;
      if (goal_valid)
        goal_I.normalize();
    }
  } else if (goal_on > 0) {
    auto it = std::lower_bound(goal_track.begin(), goal_track.end(), message.timestamp,
                               [](const std::pair<double, Eigen::Vector3d> &a, double t) { return a.first < t; });
    double best = 1e9;
    for (auto c : {it, it == goal_track.begin() ? it : std::prev(it)})
      if (c != goal_track.end() && std::abs(c->first - message.timestamp) < best) {
        best = std::abs(c->first - message.timestamp);
        goal_I = c->second;
      }
    goal_valid = best < 0.2 && goal_I.norm() > 1e-6;
    if (goal_valid)
      goal_I.normalize();
  }

  std::vector<cv::Mat> imgs(N), masks(N);
  std::vector<std::vector<cv::Mat>> pyrs(N);
  for (size_t k = 0; k < N; k++) {
    imgs[k] = img_curr.at(cams[k]);
    pyrs[k] = img_pyramid_curr.at(cams[k]);
    masks[k] = message.masks.at(k);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  auto save_last = [&](const std::vector<std::vector<cv::KeyPoint>> &pts, const std::vector<std::vector<size_t>> &ids) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    for (size_t k = 0; k < N; k++) {
      img_last[cams[k]] = imgs[k];
      img_pyramid_last[cams[k]] = pyrs[k];
      img_mask_last[cams[k]] = masks[k];
      pts_last[cams[k]] = pts[k];
      ids_last[cams[k]] = ids[k];
    }
  };

  // First frame (or after a reset): detect per pair on the current images
  bool all_empty = true;
  for (size_t k = 0; k < N; k++)
    if (!pts_last[cams[k]].empty())
      all_empty = false;
  if (all_empty) {
    std::vector<std::vector<cv::KeyPoint>> pts(N);
    std::vector<std::vector<size_t>> ids(N);
    for (size_t k = 0; k < N; k++) {
      size_t L = k, R = (k + 1) % N;
      perform_detection_pair(pyrs[L], pyrs[R], masks[L], masks[R], cams[L], cams[R], pts[L], pts[R], ids[L], ids[R]);
    }
    save_last(pts, ids);
    return;
  }

  // Top-off detection on the LAST images, pair by pair (a camera's new points from
  // pair k are visible to pair k+1, where it is the left camera)
  std::vector<std::vector<cv::KeyPoint>> pts_old(N);
  std::vector<std::vector<size_t>> ids_old(N);
  int pts_before = 0;
  for (size_t k = 0; k < N; k++) {
    pts_old[k] = pts_last[cams[k]];
    ids_old[k] = ids_last[cams[k]];
    pts_before += (int)pts_old[k].size();
  }
  // Every pair's left image rendered in its right camera, all at once: the pair loop below has to run in
  // order, the warps do not depend on it
  std::vector<std::vector<cv::Mat>> warped(N);
  if (ring_parallel_warp) {
    boost::posix_time::ptime tw0 = boost::posix_time::microsec_clock::local_time();
    parallel_for_(cv::Range(0, (int)N), LambdaBody([&](const cv::Range &range) {
                    for (int k = range.start; k < range.end; k++) {
                      size_t L = cams[k], R = cams[(k + 1) % N];
                      if (!pair_rotations.count({L, R}))
                        continue;
                      const cv::Mat &right = img_pyramid_last[R].at(0);
                      const auto &maps = get_ring_map(L, R, right.rows, right.cols);
                      cv::Mat w;
                      cv::remap(img_pyramid_last[L].at(0), w, maps.first, maps.second, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                cv::Scalar(0));
                      cv::buildOpticalFlowPyramid(w, warped[k], win_size, ring_pyr_levels);
                    }
                  }));
    ring_t_warp += (boost::posix_time::microsec_clock::local_time() - tw0).total_microseconds() * 1e-6;
  }
  for (size_t k = 0; k < N; k++) {
    size_t L = k, R = (k + 1) % N;
    perform_detection_pair(img_pyramid_last[cams[L]], img_pyramid_last[cams[R]], img_mask_last[cams[L]], img_mask_last[cams[R]], cams[L],
                           cams[R], pts_old[L], pts_old[R], ids_old[L], ids_old[R], ring_parallel_warp ? &warped[k] : nullptr);
  }
  int pts_after = 0;
  for (size_t k = 0; k < N; k++)
    pts_after += (int)pts_old[k].size();
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Temporal tracking, every camera exactly once, in parallel
  std::vector<std::vector<cv::KeyPoint>> pts_new = pts_old;
  std::vector<std::vector<uchar>> mask_klt(N);
  parallel_for_(cv::Range(0, (int)N), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    perform_matching(img_pyramid_last[cams[i]], pyrs[i], pts_old[i], pts_new[i], cams[i], cams[i], mask_klt[i]);
                  }
                }));
  rT4 = boost::posix_time::microsec_clock::local_time();
  rT5 = rT4;

  bool any = false;
  for (size_t k = 0; k < N; k++)
    if (!mask_klt[k].empty())
      any = true;
  if (!any) {
    std::vector<std::vector<cv::KeyPoint>> empty_pts(N);
    std::vector<std::vector<size_t>> empty_ids(N);
    save_last(empty_pts, empty_ids);
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Keep the good tracks of every camera; ids are shared across neighbours
  std::vector<std::vector<cv::KeyPoint>> good(N);
  std::vector<std::vector<size_t>> good_ids(N);
  std::map<size_t, int> obs_count;
  for (size_t k = 0; k < N; k++) {
    if (mask_klt[k].empty())
      continue;
    for (size_t i = 0; i < pts_new[k].size(); i++) {
      const cv::Point2f &pt = pts_new[k].at(i).pt;
      if (pt.x < 0 || pt.y < 0 || (int)pt.x >= imgs[k].cols || (int)pt.y >= imgs[k].rows)
        continue;
      if (!mask_klt[k][i])
        continue;
      if ((int)masks[k].at<uint8_t>((int)pt.y, (int)pt.x) > 127)
        continue;
      good[k].push_back(pts_new[k].at(i));
      good_ids[k].push_back(ids_old[k].at(i));
      obs_count[ids_old[k].at(i)]++;
    }
  }

  // Feature database update
  int total = 0, shared = 0;
  for (size_t k = 0; k < N; k++) {
    for (size_t i = 0; i < good[k].size(); i++) {
      cv::Point2f npt = camera_calib.at(cams[k])->undistort_cv(good[k].at(i).pt);
      database->update_feature(good_ids[k].at(i), message.timestamp, cams[k], good[k].at(i).pt.x, good[k].at(i).pt.y, npt.x, npt.y);
    }
    total += (int)good[k].size();
  }
  for (auto const &oc : obs_count)
    if (oc.second > 1)
      shared++;
  save_last(good, good_ids);
  rT6 = boost::posix_time::microsec_clock::local_time();

  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for ring detection (%d detected)\n", (rT3 - rT2).total_microseconds() * 1e-6, pts_after - pts_before);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features, %d seen by 2+ cameras)\n",
            (rT6 - rT5).total_microseconds() * 1e-6, total, shared);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT6 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: ring detection stages %.4f left FAST, %.4f warp, %.4f cross KLT, %.4f right FAST\n", ring_t_fast0, ring_t_warp,
            ring_t_klt, ring_t_fast1);

  if (goal_stats) {
    std::fprintf(goal_stats, "%.4f %d %d %d %d %.3f %.3f\n", message.timestamp, goal_n_roi, goal_n_new, goal_n_far, goal_n_hyp,
                 1000.0 * ring_t_goal, (rT3 - rT2).total_microseconds() * 1e-3);
    std::fflush(goal_stats);
  }

  // One line every 5 s at INFO, so the board's vio.log shows what the ring stereo is doing
  RingStats &rs = ring_stats;
  if (rs.t0 < 0)
    rs.t0 = message.timestamp;
  rs.frames++;
  rs.features += total;
  rs.shared += shared;
  rs.detected += pts_after - pts_before;
  rs.t_det += (rT3 - rT2).total_microseconds() * 1e-6;
  rs.t_fast0 += ring_t_fast0;
  rs.t_warp += ring_t_warp;
  rs.t_klt += ring_t_klt;
  rs.t_fast1 += ring_t_fast1;
  if (message.timestamp - rs.t0 >= 5.0) {
    double n = rs.frames;
    PRINT_INFO("[RING]: %d frames in %.1f s, per frame %.0f features (%.1f seen by 2+ cameras, %.1f new), ring detection %.1f ms "
               "(left FAST %.1f, warp %.1f, cross KLT %.1f, right FAST %.1f)\n",
               rs.frames, message.timestamp - rs.t0, rs.features / n, rs.shared / n, rs.detected / n, 1000 * rs.t_det / n,
               1000 * rs.t_fast0 / n, 1000 * rs.t_warp / n, 1000 * rs.t_klt / n, 1000 * rs.t_fast1 / n);
    rs = RingStats();
    rs.t0 = message.timestamp;
  }
}

int TrackKLT::alloc_dir(size_t cam, const cv::Point2f &pt) const {
  const cv::Mat *Ix = alloc_table(cam);
  if (!Ix)
    return -1;
  int x = std::min(std::max((int)(pt.x / 4.0f), 0), Ix->cols - 1), y = std::min(std::max((int)(pt.y / 4.0f), 0), Ix->rows - 1);
  return Ix->at<int16_t>(y, x);
}

const cv::Mat *TrackKLT::alloc_table(size_t cam) const {
  if (alloc_dirs.empty() || !ring_R_ItoC.count(cam))
    return nullptr;
  const cv::Mat *Ix;
  {
    std::lock_guard<std::mutex> lck(goal_bearings_mtx);
    auto it = alloc_idx.find(cam);
    if (it == alloc_idx.end()) {
      int W = camera_calib.at(cam)->w(), H = camera_calib.at(cam)->h();
      cv::Mat m((H + 3) / 4, (W + 3) / 4, CV_16SC1);
      Eigen::Matrix3d R_CtoI = ring_R_ItoC.at(cam).transpose();
      for (int y = 0; y < m.rows; y++)
        for (int x = 0; x < m.cols; x++) {
          Eigen::Vector2f nrm = camera_calib.at(cam)->undistort_f(Eigen::Vector2f(4.0f * x + 1.5f, 4.0f * y + 1.5f));
          Eigen::Vector3d b = (R_CtoI * Eigen::Vector3d(nrm(0), nrm(1), 1.0)).normalized();
          int best = 0;
          double bd = -2.0;
          for (size_t d = 0; d < alloc_dirs.size(); d++) {
            double c = b.dot(alloc_dirs[d]);
            if (c > bd) bd = c, best = (int)d;
          }
          m.at<int16_t>(y, x) = (int16_t)best;
        }
      it = alloc_idx.emplace(cam, m).first;
    }
    Ix = &it->second;
  }
  return Ix;
}

bool TrackKLT::in_goal(size_t cam, const cv::Point2f &pt) const {
  if (alloc_on > 0) {
    if (!alloc_valid)
      return false;
    int d = alloc_dir(cam, pt);
    return d >= 0 && alloc_target[d] >= 1.0f;
  }
  if (!goal_valid || !ring_R_ItoC.count(cam))
    return false;
  const cv::Mat *B;
  {
    std::lock_guard<std::mutex> lck(goal_bearings_mtx);
    auto it = goal_bearings.find(cam);
    if (it == goal_bearings.end()) {
      int W = camera_calib.at(cam)->w(), H = camera_calib.at(cam)->h();
      cv::Mat m((H + 3) / 4, (W + 3) / 4, CV_32FC3);
      for (int y = 0; y < m.rows; y++)
        for (int x = 0; x < m.cols; x++) {
          Eigen::Vector2f n = camera_calib.at(cam)->undistort_f(Eigen::Vector2f(4.0f * x + 1.5f, 4.0f * y + 1.5f));
          Eigen::Vector3f b(n(0), n(1), 1.0f);
          b.normalize();
          m.at<cv::Vec3f>(y, x) = cv::Vec3f(b(0), b(1), b(2));
        }
      it = goal_bearings.emplace(cam, m).first;
    }
    B = &it->second;
  }
  int x = std::min(std::max((int)(pt.x / 4.0f), 0), B->cols - 1), y = std::min(std::max((int)(pt.y / 4.0f), 0), B->rows - 1);
  const cv::Vec3f &b = B->at<cv::Vec3f>(y, x);
  Eigen::Vector3d g = (ring_R_ItoC.at(cam) * goal_I).normalized();
  return b[0] * g(0) + b[1] * g(1) + b[2] * g(2) > goal_cos;
}

void TrackKLT::goal_setup() {
  auto env = [](const char *name) {
    const char *v = std::getenv(name);
    return std::string(v ? v : "");
  };
  goal_on = 0;
  if (!env("OV_GOAL_DEG").empty())
    goal_cos = std::cos(std::atof(env("OV_GOAL_DEG").c_str()) * M_PI / 180.0);
  if (!env("OV_GOAL_FEATS").empty())
    goal_feats = std::atoi(env("OV_GOAL_FEATS").c_str());
  if (!env("OV_GOAL_MIN_PX").empty())
    goal_min_px = std::max(2, std::atoi(env("OV_GOAL_MIN_PX").c_str()));
  if (!env("OV_GOAL_EPI_DEG").empty())
    goal_epi_sin = std::sin(std::atof(env("OV_GOAL_EPI_DEG").c_str()) * M_PI / 180.0);
  if (!env("OV_GOAL_HYP_ALL").empty())
    goal_hyp_all = std::atoi(env("OV_GOAL_HYP_ALL").c_str());
  std::stringstream ds(env("OV_GOAL_DEPTHS"));
  std::string tok;
  while (std::getline(ds, tok, ','))
    if (!tok.empty() && std::atof(tok.c_str()) > 0)
      goal_depths.push_back(std::atof(tok.c_str()));
  std::string f = env("OV_GOAL_FILE");
  if (!f.empty()) {
    std::ifstream in(f);
    double t, x, y, z;
    while (in >> t >> x >> y >> z)
      goal_track.emplace_back(t, Eigen::Vector3d(x, y, z));
    std::sort(goal_track.begin(), goal_track.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    goal_on = goal_track.empty() ? 0 : 1;
  }
  // closed loop (2026-11-15): OV_GOAL_LIVE names a one-line file "t x y z" (goal direction in the IMU frame) that a ROS node
  // rewrites (atomically) from the planner's intention; it is read in every frame and used while it is at most 0.5 s old
  goal_live = env("OV_GOAL_LIVE");
  if (!goal_live.empty())
    goal_on = 2;
  if (!env("OV_ALLOC_FEATS").empty())
    alloc_feats = std::atoi(env("OV_ALLOC_FEATS").c_str());
  if (!env("OV_EXTRA_FEATS").empty()) {          // the uniform control of the allocation maps: the same budget spread over the
    num_features += std::atoi(env("OV_EXTRA_FEATS").c_str());   // whole image of every camera by the standard grid detector
    PRINT_WARNING("[ALLOC]: uniform control, %d features per camera\n", num_features);
  }
  if (!env("OV_ALLOC_DIRS").empty() && !env("OV_ALLOC_FILE").empty()) {
    std::ifstream dn(env("OV_ALLOC_DIRS"));
    double x, y, z;
    while (dn >> x >> y >> z)
      alloc_dirs.emplace_back(Eigen::Vector3d(x, y, z).normalized());
    std::ifstream in(env("OV_ALLOC_FILE"));
    std::string line;
    while (std::getline(in, line)) {
      std::stringstream ls(line);
      double t;
      if (!(ls >> t))
        continue;
      std::vector<float> w;
      float v;
      while (ls >> v)
        w.push_back(v);
      if (w.size() == alloc_dirs.size())
        alloc_track.emplace_back(t, w);
    }
    std::sort(alloc_track.begin(), alloc_track.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    alloc_on = alloc_track.empty() ? 0 : 1;
    PRINT_WARNING("[ALLOC]: %zu directions, %zu rows (on %d), budget %d features\n", alloc_dirs.size(), alloc_track.size(), alloc_on, alloc_feats);
  }
  if (!env("OV_GOAL_STATS").empty())
    goal_stats = std::fopen(env("OV_GOAL_STATS").c_str(), "w");
  PRINT_WARNING("[GOAL]: goal track %zu rows (on %d), sector %.0f deg, %d features, min %d px, %zu depth hypotheses (all candidates %d)\n",
                goal_track.size(), goal_on, std::acos(goal_cos) * 180.0 / M_PI, goal_feats, goal_min_px, goal_depths.size(), goal_hyp_all);
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out) {

  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++)
      mask_out.push_back((uchar)0);
    return;
  }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
  }

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

  // Loop through and record only ones that are valid
  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}
