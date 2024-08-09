#ifndef OPTICALFLOWTRACKER_HPP
#define OPTICALFLOWTRACKER_HPP
#include <opencv2/opencv.hpp>


inline float GetPixelValue(const cv::Mat &img, float x, float y);

class OpticalFlowTracker{
    public: 
        OpticalFlowTracker(const cv::Mat &img1_,
                                const cv::Mat &img2_,
                                const std::vector<cv::KeyPoint> &kp1,
                                std::vector<cv::KeyPoint> &kp2,
                                std::vector<bool> &success_,
                                bool inverse_ = true,
                                bool has_initial_ = false);
        void calculateOpticalFlow(const cv::Range &range);
    private:
        const cv::Mat &img1;
        const cv::Mat &img2;
        const std::vector<cv::KeyPoint> &kp1;
        std::vector<cv::KeyPoint> &kp2;
        std::vector<bool> &success;
        bool inverse = true;
        bool has_initial = false;
};
#endif // OPTICALFLOWTRACKER_HPP