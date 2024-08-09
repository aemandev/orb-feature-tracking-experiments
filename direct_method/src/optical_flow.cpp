#include "optical_flow.hpp"
#include "OpticalFlowTracker.hpp"

std::string file_1 = "./LK1.png";  // first image
std::string file_2 = "./LK2.png";  // second image


void OpticalFlowSingleLevel(const cv::Mat &img1,
                            const cv::Mat &img2,
                            const std::vector<cv::KeyPoint> &kp1,
                            std::vector<cv::KeyPoint> &kp2,
                            std::vector<bool> &success,
                            bool inverse, bool has_initial){
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(cv::Range(0, kp1.size()), std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));

}

void OpticalFlowMultiLevel(const cv::Mat &img1,
                            const cv::Mat &img2,
                            const std::vector<cv::KeyPoint> &kp1,
                            std::vector<cv::KeyPoint> &kp2,
                            std::vector<bool> &success,
                            bool inverse){
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    std::vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++){
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i-1], img1_pyr,
            cv::Size(pyr1[i-1].cols * pyramid_scale, pyr1[i-1].rows * pyramid_scale));
            cv::resize(pyr2[i-1], img2_pyr,
            cv::Size(pyr2[i-1].cols * pyramid_scale, pyr2[i-1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    // coarse-to-fine LK tracking in pyramids
    std::vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1){
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level --){
        // from coarse to fine
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);  

        if (level>0) {
            for (auto &kp: kp1_pyr)
            kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr) 
            kp.pt /= pyramid_scale;
        }
    }
    for (auto &kp: kp2_pyr)
    kp2.push_back(kp);
}

int main(int argc, char **argv) {
    cv::Mat img1 = cv::imread(file_1, 0);
    cv::Mat img2 = cv::imread(file_2, 0);

    std::vector<cv::KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(1000, 0.01, 20);  // Max 500 KP
    detector->detect(img1, kp1);

    // Track kp in second image. First use sinlgle level lk in the valiadtion picture
    std::vector<cv::KeyPoint> kp2_single;
    std::vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // Thne test multi level lk
    std::vector<cv::KeyPoint> kp2_multi;
    std::vector<bool> success_multi;
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);

    // Use opencv's flow for validation
    std::vector<cv::Point2f> pt1, pt2;
    for (auto &kp : kp1) pt1.push_back(kp.pt);
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);

    cv::Mat img2_single, img2_multi, img2_CV;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked opencv", img2_CV);
    cv::waitKey(0);


    return 0;
}

