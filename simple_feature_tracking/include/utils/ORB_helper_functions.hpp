#ifndef ORB_HELPER_FUNCTIONS_HPP
#define ORB_HELPER_FUNCTIONS_HPP

#include <opencv2/opencv.hpp>

void find_ORB_features_grid(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, int n_features, int grid_size);
void match_orb_features(const cv::Mat&desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches);
void calculate_fundamental_matrix(const cv::Mat& K, const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches, cv::Mat& R, cv::Mat &t);
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);
void triangulate_points(const std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2,
const std::vector<cv::DMatch> &matches, const cv::Mat &R, const cv::Mat &t, const cv::Mat &K, std::vector<cv::Point3d> &pnts3D);

#endif // FEATURE_EXTRACTION_H
