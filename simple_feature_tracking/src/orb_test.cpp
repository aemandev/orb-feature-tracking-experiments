// Opencv imports
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>

// C++ imports
#include <ORB_helper_functions.hpp>
#include <iostream>
#include <nonlinear_optimization.hpp>


inline cv::Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

int main(int argc, char** argv) {
    if (argc < 7 ) {
        std::cout << "Usage: ./orb_test image_path num_features num_grids use_grid num_features" << std::endl;
        return 1;
    }
    int num_grids = std::stoi(argv[3]);
    bool use_grid = std::stoi(argv[4]);
    int n_features = std::stoi(argv[5]);
    bool resize_image = std::stoi(argv[6]);
    int im_width = 540;
    int im_height = 960;
    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    cv::Mat image1_resize, image2_resize;

    if (resize_image){
        cv::resize(image1, image1_resize, cv::Size(im_width, im_height), 0, 0, cv::INTER_LINEAR);
        cv::resize(image2, image2_resize, cv::Size(im_width, im_height), 0, 0, cv::INTER_LINEAR);
    } else {
        image1_resize = image1.clone();
        image2_resize = image2.clone();
    }

    // Initialize orb
    // cv::Ptr<cv::ORB> detector = cv::ORB::create(n_features);
    cv::Ptr<cv::ORB> detector = cv::ORB::create(n_features);
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    // cv::Ptr<cv::
    std::vector<cv::KeyPoint> kp1, kp2;

    // Detect features
    if (use_grid){
        find_ORB_features_grid(image1_resize, kp1, n_features, num_grids);
        find_ORB_features_grid(image2_resize, kp2, n_features, num_grids);
    } else {
        detector->detect(image1_resize, kp1);  
        detector->detect(image2_resize, kp2);
    }

    // Compute the descriptors
    cv::Mat desc1, desc2;
    detector->compute(image1_resize, kp1, desc1);
    detector->compute(image2_resize, kp2, desc2);

    // Matcher
    std::vector<cv::DMatch> matches;
    cv::Mat img_matches;
    match_orb_features(desc1, desc2, matches);
    cv::drawMatches(image1_resize, kp1, image2_resize, kp2, matches, img_matches);
    // cv::imshow("Matches", img_matches);
    std::cout << "Number of matches: " << matches.size() << std::endl;
    cv::waitKey(0);

    // Find fundamental matrix and Rotation and Translation
    cv::Mat K, R, t;
    K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    calculate_fundamental_matrix(K, kp1, kp2, matches, R, t);
    std::cout << "R FUndamental: " << R << std::endl;
    std::cout << "t Fundamental: " << t << std::endl;

    // Triangulate points
    std::vector<cv::Point3d> pnts3D;
    triangulate_points(kp1, kp2, matches, R, t, K, pnts3D);
    // Draw the pnts
    for (cv::Point3d pnt:pnts3D){
        // std::cout << "3D point: " << pnt << std::endl;
        cv::circle(image2_resize, cv::Point(pnt.x, pnt.y), 2, cv::Scalar(0, 255, 0), 2);
    }

    cv::Mat img1_plot = image1_resize.clone();
    cv::Mat img2_plot = image2_resize.clone();
    for (int i = 0; i < matches.size(); i++) {

        float depth1 = pnts3D[i].z;
        // std::cout << "depth: " << depth1 << std::endl;
        cv::Point2d pt1_cam = pixel2cam(kp1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, kp1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        cv::Mat pt2_trans = R * (cv::Mat_<double>(3, 1) << pnts3D[i].x, pnts3D[i].y, pnts3D[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, kp2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

    if ( argc == 8 ) {
        cv::Mat depth_map = cv::imread(argv[7], CV_LOAD_IMAGE_UNCHANGED);
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;

        for (cv::DMatch m:matches){
            ushort d = depth_map.ptr<unsigned short>(int(kp1[m.queryIdx].pt.y))[int(kp1[m.queryIdx].pt.x)];
            if (d==0){
                continue;
            }
            float dd = d / 5000.0;
            cv::Point2d pt1 = pixel2cam(kp1[m.queryIdx].pt, K);
            pts3d.push_back(cv::Point3f(pt1.x*dd, pt1.y*dd, dd));
            pts2d.push_back(cv::Point2f(kp2[m.trainIdx].pt));
        }
        // print
        // std::cout << "3d-2d pairs: " << pts3d.size() << std::endl;
        // std::cout << "3d points: " << pts3d << std::endl;

        cv::Mat r,t;
        cv::solvePnP(pts3d, pts2d, K, cv::Mat(), r, t);
        cv::Mat R_pnp;
        cv::Rodrigues(r, R_pnp);
        std::cout << "R_pnp = " << R_pnp << std::endl;
        std::cout << "t_pnp: " << t << std::endl;

        // Perform bundle adjustment
        VecVector3d points_3d;
        VecVector2d points_2d;
        for (int i = 0; i < pts3d.size(); i++){
            points_3d.push_back(Eigen::Vector3d(pts3d[i].x, pts3d[i].y, pts3d[i].z));
            points_2d.push_back(Eigen::Vector2d(pts2d[i].x, pts2d[i].y));
        }
        Sophus::SE3d pose_gn;
        // Print SE3    
        // std::cout << "SE3: " << SE3_Rt;
        bundle_adjustment_gauss_newton(points_3d,
        points_2d, K, pose_gn);
        std::cout << "Pose by bundle adjustment: \n" << pose_gn.matrix() << std::endl;

        Sophus::SE3d pose_g2o;
        bundleAdjustmentG2O(points_3d, points_2d, K, pose_g2o);
        std::cout << "Pose by g2o: \n" << pose_g2o.matrix() << std::endl;

    }

    return 0;
}