#include "OpticalFlowTracker.hpp"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>


inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}


// Constructor
OpticalFlowTracker::OpticalFlowTracker(const cv::Mat &img1_,
                                        const cv::Mat &img2_,
                                        const std::vector<cv::KeyPoint> &kp1_,
                                        std::vector<cv::KeyPoint> &kp2_,
                                        std::vector<bool> &success_,
                                        bool inverse_,
                                        bool has_initial_):
                                        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_),
                                        success(success_), inverse(inverse_), has_initial(has_initial_){}
// Calculate optical flow
void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range){
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++){
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx, dy need to be estimated
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x; // Different on the x axis for each kp
            dy = kp2[i].pt.y - kp.pt.y; // Different on the y axis for each kp
        }
        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded
        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero(); // Hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero(); // bias
        Eigen::Vector2d J; // jacobian
        for (int iter = 0; iter < iterations; iter++){
            if (inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }
            cost = 0;

            // Compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    // For each kp, compute the cost and jacobian
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) - 
                        GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y);; // Jacobian
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) - 
                                GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) - 
                                GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1)));
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it
                        // and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) - 
                                GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1)));
                    }
                    // compute H, b and set cost
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J*J.transpose();
                    }
                }
            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // Sometimes occurred when we have a black or white patch and H is
                // not invertible. We can simply ignore this point.
                std::cout << "Update is nan" << std::endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }
            
            // update dx, dx
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // converge
                break;
            }
        }
        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
};



