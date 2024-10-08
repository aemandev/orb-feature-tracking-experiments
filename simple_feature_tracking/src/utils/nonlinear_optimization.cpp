#include <nonlinear_optimization.hpp>
#include <chrono>

void bundle_adjustment_gauss_newton(const VecVector3d &points_3d, const VecVector2d &points_2d, const cv::Mat &K, Sophus::SE3d &pose){
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0,0), fy = K.at<double>(1,1), cx = K.at<double>(0,2), cy = K.at<double>(1,2);

    for (int iter = 0; iter < iterations; iter++){
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // Compute cost. See page 106
        for (int i = 0; i < points_3d.size(); i++){
            // Look at BA notes 
            Eigen::Vector3d pc = pose*points_3d[i];
            double inv_z = 1.0/pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0]/pc[2] + cx, fy*pc[1]/pc[2] + cy);
            // Error in the projection
            Eigen::Vector2d e = points_2d[i] - proj;
            cost+=e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            // pg 159
            J << -fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;
            // H is the approximation of the second-order Hessian matrix in Newton's method
            // They attribute the error each point contributes to the error of the whole system in terms of the
            // reprojection of the points
            H+=J.transpose()*J;
            b+= -J.transpose()*e;
        }

        Vector6d dx;
        // ldlt finds the inverose of H. Factorizes H into lower trianglular, D, and Lower.transpose()
        dx = H.ldlt().solve(b);
        if (isnan(dx[0])){
            std::cout << "Result is nan!" << std::endl;
            break;
        }
        if (iter>0 && cost >= lastCost){
            // Cost increase, update is not good
            std::cout << "Cost: " << cost << ", last cost: " << lastCost << std::endl;
            break;
        }

        // Update estimation. This is equivalent to doing x(k+1) = x(k) + dx
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        std::cout << "Iteration " << iter << " cost=" << std::setprecision(12) << cost << std::endl;
        if (dx.norm() < 1e-6){
            // Converged
            break;
        }

        std::cout << "Estimated pose by gn: " << pose.matrix() << std::endl;
    }
};

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override {
            _estimate = Sophus::SE3d();
        }
        // Left multiplication on SE3
        virtual void oplusImpl(const double *update) override {
            Eigen::Matrix<double, 6, 1> update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
            _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
        }

        virtual bool read(std::istream &in) override {}
        virtual bool write(std::ostream &out) const override{}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d & pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T*_pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();

    }

    virtual void linearizeOplus() override{
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T* _pos3d;
        double fx = _K(0,0);
        double fy = _K(1,1);
        double cx = _K(0,2);
        double cy = _K(1,2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z*Z;
        _jacobianOplusXi << -fx/Z, 0, fx*X/Z2, fx*X*Y/Z2, -fx-fx*X*X/Z2, fx*Y/Z,
                            0, -fy/Z, fy*Y/Z2, fy+fy*Y*Y/Z2, -fy*X*Y/Z2, -fy*X/Z;
    }

    virtual bool read(std::istream &in) override {}
    virtual bool write(std::ostream &out) const override {}
    private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

    virtual void computeError() override {
        const VertexPose *pose = static_cast<const VertexPose * > ( _vertices[0] );
        _error = _measurement - pose->estimate() * _point;
    }
    
    virtual void linearizeOplus() override {
        VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = pose->estimate();
        Eigen::Vector3d xyz_trans = T * _point;
        _jacobianOplusXi.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3,3>(0,3) = Sophus::SO3d::hat(xyz_trans);
    }
    virtual bool read(std::istream &in) override {}
    virtual bool write(std::ostream &out) const override {}
    protected:
    Eigen::Vector3d _point;
};



void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d,
const cv::Mat &K, Sophus::SE3d &pose) {
    //Define g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // Pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    // Gradient descent method, you canm choose from GN, LM, DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // Graph model
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // vertex
    VertexPose *vertex_pose = new VertexPose(); // Camera vertex_pose
    vertex_pose->setId(0);
    // Initial pose estimate
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);
    
    // K 
    Eigen::Matrix3d K_eigen;
    K_eigen <<
    K.at<double>(0, 0), K.at<double>(0,1), K.at<double>(0,2),
    K.at<double>(1, 0), K.at<double>(1,1), K.at<double>(1,2),
    K.at<double>(2, 0), K.at<double>(2,1), K.at<double>(2,2);

    // edges. These should be the reporjection error
    int index = 1;
     for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        // How much to trust this point
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
     }

    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();  
    // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // std::cout << "optimization costs time: " << time_used.count() << " seconds." << std::endl;
    std::cout << "Pose estimated by g2o = \n" << vertex_pose->estimate().matrix() << std::endl;
    pose = vertex_pose->estimate();
}


void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R, cv::Mat &t){
    cv::Point3f p1, p2; // Center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++){
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);

    std::vector<cv::Point3f> q1(N), q2(N); // Remove the center
    for (int i = 0; i < N; i++){
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;   
    }
    // Compute qq1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i= 0; i < N; i++){
        W+= Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    std::cout << "W = " << W << std::endl;

    // Then do SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    std::cout << "U = " << U << std::endl;
    std::cout << "V = " << V << std::endl;

    Eigen::Matrix3d R_ = U*V.transpose();
    if (R_.determinant() < 0){
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
    // Convert to cv::Mat
    R = (cv::Mat_<double>(3, 3) <<
        R_(0,0), R_(0,1), R_(0,2),
        R_(1,0), R_(1,1), R_(1,2),
        R_(2,0), R_(2,1), R_(2,2)
    );
    cv::Mat R_inv;
    cv::Mat t_inv;
    R_inv = R.t();
    std::cout << "R = " << R << std::endl;
    std::cout << "R_inv = " << R_inv << std::endl;
    t = (cv::Mat_<double>(3, 1) << t_(0,0), t_(1,0), t_(2,0));   
    t_inv = -R_inv * t;
    std::cout << "t = " << t << std::endl;
    std::cout << "t_inv = " << t_inv << std::endl;
}

void bundle_adjustment_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, 
cv::Mat &R, cv::Mat &t){
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // Vertex
    VertexPose *pose = new VertexPose();
    pose->setId(0);
    pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(pose);

    //Edges
    for (size_t i=0; i< pts1.size(); i++){
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
            Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector3d(
            pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::cout << "Pose estimated by g2o = \n" << pose->estimate().matrix() << std::endl;
    Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = pose->estimate().translation();
    R = (cv::Mat_<double>(3, 3) <<
        R_(0,0), R_(0,1), R_(0,2),
        R_(1,0), R_(1,1), R_(1,2),
        R_(2,0), R_(2,1), R_(2,2)
    );
    t = (cv::Mat_<double>(3, 1) << t_(0,0), t_(1,0), t_(2,0));
}
