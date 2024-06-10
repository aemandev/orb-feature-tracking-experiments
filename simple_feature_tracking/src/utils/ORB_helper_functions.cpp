#include <ORB_helper_functions.hpp>
#include <nonlinear_optimization.hpp>
#include <opencv2/features2d.hpp>

void find_ORB_features_grid(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, int n_features, int grid_size){
    // Number of rows and columns in the image
    int img_rows = image.rows;
    int img_cols = image.cols;

    // Size of the grid
    int move_row_size = img_rows / grid_size;
    int move_col_size = img_cols / grid_size; 

    cv::Ptr<cv::ORB> detector = cv::ORB::create(n_features);
    for (int i = 0; i < grid_size; i++){
        for (int j = 0; j < grid_size; j++){
            // std::cout << "Processing cell " << i << " " << j << std::endl;
            cv::Rect cell(j * move_col_size, i * move_row_size, move_col_size, move_row_size);
            cv::Mat cell_image = image(cell);
            // cv::imshow("Cell", cell_image);
            // cv::waitKey(0);
            std::vector<cv::KeyPoint> cell_kp;
            detector->detect(cell_image, cell_kp);
            // std::cout << "Number of keypoints in cell " << i << " " << j << ": " << cell_kp.size() << std::endl;
            for (auto& kp : cell_kp){
                kp.pt.x += j * move_col_size;
                kp.pt.y += i * move_row_size;
                keypoints.push_back(kp);
            }
            // Draw gridlines for each cell
            cv::line(image, cv::Point(j * move_col_size, 0), cv::Point(j * move_col_size, img_rows), cv::Scalar(255, 0, 0), 2);
            cv::line(image, cv::Point(0, i * move_row_size), cv::Point(img_cols, i * move_row_size), cv::Scalar(255, 0, 0), 2);
        }

    }
}

void match_orb_features(const cv::Mat&desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches){
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            matches.push_back(knn_matches[i][0]);
        }
    }
    std::sort(matches.begin(), matches.end());
}

void calculate_fundamental_matrix(const cv::Mat& K, const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches, cv::Mat& R, cv::Mat &t){
        std::vector<cv::Point2f> pts1, pts2;
        for (auto& match: matches){
            pts1.push_back(kp1[match.queryIdx].pt);
            pts2.push_back(kp2[match.trainIdx].pt);
        }

        cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT);
        // std::cout << "Fundamental matrix: " << F << std::endl;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K);
        // std::cout << "Essential matrix: " << E << std::endl;

        cv::recoverPose(E, pts1, pts2, K, R, t);
        // std::cout << "Rotation: " << R << std::endl;
        // std::cout << "Translation: " << t << std::endl;

        cv::Mat t_x =
            (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
            t.at<double>(2, 0), 0, -t.at<double>(0, 0),
            -t.at<double>(1, 0), t.at<double>(0, 0), 0);

        for (cv::DMatch m:matches){
            cv::Point2d pt1 = pixel2cam(kp1[m.queryIdx].pt, K);
            cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
            cv::Point2d pt2 = pixel2cam(kp2[m.trainIdx].pt, K);
            cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
            cv::Mat d = y2.t() * t_x * R * y1;
            // std::cout << "epipolar constraint = " << d << std::endl;

        }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void triangulate_points(const std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2,
const std::vector<cv::DMatch> &matches, const cv::Mat &R, const cv::Mat &t, const cv::Mat &K, std::vector<cv::Point3d> &pnts3D){
    cv::Mat T1 = (cv::Mat_<float>(3,4) << 1,0,0,0,
                                      0,1,0,0,
                                      0,0,1,0);
    cv::Mat T2 = (cv::Mat_<float>(3,4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
                                      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
                                      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0));

    std::vector<cv::Point2f> pts1, pts2;
    for (cv::DMatch match:matches){
        pts1.push_back(pixel2cam(kp1[match.queryIdx].pt, K));
        pts2.push_back(pixel2cam(kp2[match.trainIdx].pt, K));
    }
    cv::Mat pnts4D;
    cv::triangulatePoints(T1, T2, pts1, pts2, pnts4D);
    // convert to non-comogenous points
    for (int i = 0; i < pnts4D.cols; i++){
        cv::Mat x = pnts4D.col(i);
        x/= x.at<float>(3,0);
        cv::Point3d p(
            x.at<float>(0,0),
            x.at<float>(1,0),
            x.at<float>(2,0));
        pnts3D.push_back(p);
    }
}
