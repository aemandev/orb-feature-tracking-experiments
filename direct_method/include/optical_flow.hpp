#ifndef OPTICAL_FLOW_HPP
#define OPTICAL_FLOW_HPP

#include <opencv2/opencv.hpp>

void OpticalFlowSingleLevel(const cv::Mat &img1,
                            const cv::Mat &img2,
                            const std::vector<cv::KeyPoint> &kp1,
                            std::vector<cv::KeyPoint> &kp2,
                            std::vector<bool> &success,
                            bool inverse = false, bool has_initial = false);

void OpticalFlowMultiLevel(const cv::Mat &img1,
                            const cv::Mat &img2,
                            const std::vector<cv::KeyPoint> &kp1,
                            std::vector<cv::KeyPoint> &kp2,
                            std::vector<bool> &success,
                            bool inverse= false);
#endif // OPTICAL_FLOW_HPP
