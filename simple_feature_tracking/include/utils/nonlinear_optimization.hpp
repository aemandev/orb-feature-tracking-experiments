#ifndef NONLINEAR_OPTIMIZATION_H
#define NONLINEAR_OPTIMIZATION_H

#include <opencv2/opencv.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>


typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundle_adjustment_gauss_newton(const VecVector3d &points_3d, const VecVector2d &points_2d, const cv::Mat &K, Sophus::SE3d &pose);
void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const cv::Mat &K, Sophus::SE3d &pose);
void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R, cv::Mat &t);
void bundle_adjustment_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R, cv::Mat &t);



#endif // FEATURE_EXTRACTION_H
