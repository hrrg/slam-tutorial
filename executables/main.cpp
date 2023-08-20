#define FMT_HEADER_ONLY
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <fmt/format.h>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <memory>
#include <regex>
#include <tuple>
#include <unistd.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tsl/robin_map.h>

#include "stereo_calibration.h"
#include <sophus/se3.hpp>

// TODO: Manage parameters
// Constant Parameters
constexpr int ESTIMATION_THRESHOLD_ = 0.0001;
constexpr int MAX_NUM_ITERATIONS_ = 100;
// Configurable Parameters
double initial_threshold_ = 2.0;   // adptive
const double min_motion_th_ = 0.1; // constant

double model_error_sse2_ = 0;
int num_samples_ = 0;
const double min_range_ = 5.0;   // constant
const double max_range_ = 100.0; // constant
const double voxel_size_ = 20;   // constant
double max_distance_ = 100.0;
int max_points_per_voxel_ = 20;
Sophus::SE3d model_deviation_ = Sophus::SE3d();

using Voxel = Eigen::Vector3i;

struct VoxelBlock {
    // buffer of points with a max limit of n_points
    std::vector<Eigen::Vector3d> points;
    int num_points_;
    inline void AddPoint(const Eigen::Vector3d &point) {
        if (points.size() < static_cast<size_t>(num_points_))
            points.push_back(point);
    }
};
struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};

class SLAM_Context {
  public:
    SLAM_Context(){};
    ~SLAM_Context(){};

    Sophus::SE3d motion_model() const {
        Sophus::SE3d pred = Sophus::SE3d(); // 아무 history 없을때 identity
        const size_t N = poses_.size();
        if (N < 2)
            return pred;
        return poses_[N - 2].inverse() * poses_[N - 1];
    }
    std::vector<Eigen::Vector3d> get_registered_map() const {
        std::vector<Eigen::Vector3d> points;
        points.reserve(max_points_per_voxel_ * map_.size());
        for (const auto &[voxel, voxel_block] : map_) {
            (void)voxel;
            for (const auto &point : voxel_block.points) {
                points.push_back(point);
            }
        }
        return points;
    }

  public:
    std::vector<Sophus::SE3d> poses_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
};
SLAM_Context slam_ctx;
// ICP
namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
} // namespace Eigen

inline double square(double x) { return x * x; }

struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};

struct Cloud_Tuple {
    Cloud_Tuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
};

std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
get_correspondence(const std::vector<Eigen::Vector3d> &points,
                   const tsl::robin_map<Voxel, VoxelBlock, VoxelHash> &map,
                   double max_correspondance_distance) {
    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighboor = [&](const Eigen::Vector3d &point) {
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<Voxel> voxels;
        voxels.reserve(27);
        for (int i = kx - 1; i < kx + 1 + 1; ++i) {
            for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        std::vector<Eigen::Vector3d> neighboors;
        neighboors.reserve(27 * max_points_per_voxel_);
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
            auto search = map.find(voxel);
            if (search != map.end()) {
                const auto &points = search->second.points;
                if (!points.empty()) {
                    for (const auto &point : points) {
                        neighboors.emplace_back(point);
                    }
                }
            }
        });

        Eigen::Vector3d closest_neighbor;
        double closest_distance2 = std::numeric_limits<double>::max();
        std::for_each(neighboors.cbegin(), neighboors.cend(), [&](const auto &neighbor) {
            double distance = (neighbor - point).squaredNorm();
            if (distance < closest_distance2) {
                closest_neighbor = neighbor;
                closest_distance2 = distance;
            }
        });

        return closest_neighbor;
    };
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        // Identity
        Cloud_Tuple(points.size()),
        // 1st lambda: Parallel computation
        [max_correspondance_distance, &GetClosestNeighboor](
            const tbb::blocked_range<points_iterator> &r, Cloud_Tuple res) -> Cloud_Tuple {
            auto &[src, tgt] = res;
            src.reserve(r.size());
            tgt.reserve(r.size());
            for (const auto &point : r) {
                Eigen::Vector3d closest_neighboors = GetClosestNeighboor(point);
                if ((closest_neighboors - point).norm() < max_correspondance_distance) {
                    src.emplace_back(point);
                    tgt.emplace_back(closest_neighboors);
                }
            }
            return res;
        },
        // 2nd lambda: Parallel reduction
        [](Cloud_Tuple a, const Cloud_Tuple &b) -> Cloud_Tuple {
            auto &[src, tgt] = a;
            const auto &[srcp, tgtp] = b;
            src.insert(src.end(), //
                       std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(), //
                       std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(source, target);
}

void transform_cloud(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &cloud) {
    std::transform(cloud.cbegin(), cloud.cend(), cloud.begin(),
                   [&](const auto &point) { return T * point; });
}

Sophus::SE3d AlignClouds(const std::vector<Eigen::Vector3d> &source,
                         const std::vector<Eigen::Vector3d> &target, double th) {
    auto compute_jacobian_and_residual = [&](auto i) {
        const Eigen::Vector3d residual = source[i] - target[i];
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(J_r, residual);
    };

    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>{0, source.size()},
        // Identity
        ResultTuple(),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            auto Weight = [&](double residual2) { return square(th) / square(th + residual2); };
            auto &[JTJ_private, JTr_private] = J;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                const double w = Weight(residual.squaredNorm());
                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;
            }
            return J;
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });

    const Eigen::Vector6d x = JTJ.ldlt().solve(-JTr);
    return Sophus::SE3d::exp(x);
}

Sophus::SE3d register_frame_icp(const std::vector<Eigen::Vector3d> &cloud,
                                const tsl::robin_map<Voxel, VoxelBlock, VoxelHash> &map,
                                const Sophus::SE3d &pose_initial_guess,
                                double max_correspondence_distance, double kernel) {
    if (map.empty())
        return pose_initial_guess;

    // Equation (9)
    std::vector<Eigen::Vector3d> source = cloud;
    transform_cloud(pose_initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        // Equation (10)
        const auto &[src, tgt] = get_correspondence(source, map, max_correspondence_distance);
        // Equation (11)
        auto estimation = AlignClouds(src, tgt, kernel);
        // Equation (12)
        transform_cloud(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (estimation.log().norm() < ESTIMATION_THRESHOLD_)
            break;
    }
    // Spit the final transformation
    return T_icp * pose_initial_guess;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double compute_model_error(const Sophus::SE3d &model_deviation, double max_range) {
    const double theta = Eigen::AngleAxisd(model_deviation.rotationMatrix()).angle();
    const double delta_rot = 2.0 * max_range * std::sin(theta / 2.0);
    const double delta_trans = model_deviation.translation().norm();
    return delta_trans + delta_rot;
}

double get_adaptive_threshold() {
    double model_error = compute_model_error(model_deviation_, max_range_);
    if (model_error > min_motion_th_) {
        model_error_sse2_ += model_error * model_error;
        num_samples_++;
    }

    if (num_samples_ < 1) {
        return initial_threshold_;
    }
    return std::sqrt(model_error_sse2_ / num_samples_);
}

std::vector<Eigen::Vector3d> voxelize_downsample(const std::vector<Eigen::Vector3d> &cloud,
                                                 double voxel_size) {
    tsl::robin_map<Voxel, Eigen::Vector3d, VoxelHash> grid;
    grid.reserve(cloud.size());
    for (const auto &point : cloud) {
        const auto voxel = Voxel((point / voxel_size).cast<int>());
        if (grid.contains(voxel))
            continue;
        grid.insert({voxel, point});
    }
    std::vector<Eigen::Vector3d> frame_dowsampled;
    frame_dowsampled.reserve(grid.size());
    for (const auto &[voxel, point] : grid) {
        (void)voxel;
        frame_dowsampled.emplace_back(point);
    }
    return frame_dowsampled;
}

std::vector<Eigen::Vector3d> preprocess(const std::vector<Eigen::Vector3d> &cloud, double max_range,
                                        double min_range) {
    std::vector<Eigen::Vector3d> inliers;
    std::copy_if(cloud.cbegin(), cloud.cend(), std::back_inserter(inliers), [&](const auto &pt) {
        const double norm = pt.norm();
        return norm < max_range && norm > min_range;
    });
    return inliers;
}

void add_cloud_to_map(const std::vector<Eigen::Vector3d> &cloud) {
    auto map = &slam_ctx.map_;
    std::for_each(cloud.cbegin(), cloud.cend(), [&](const auto &point) {
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        auto search = map->find(voxel);
        if (search != map->end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map->insert({voxel, VoxelBlock{{point}, max_points_per_voxel_}});
        }
    });
}

void RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    for (const auto &[voxel, voxel_block] : slam_ctx.map_) {
        const auto &pt = voxel_block.points.front();
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt - origin).squaredNorm() > (max_distance2)) {
            slam_ctx.map_.erase(voxel);
        }
    }
}

void update_map(const std::vector<Eigen::Vector3d> cloud, const Sophus::SE3d &pose) {
    std::vector<Eigen::Vector3d> cloud_transformed(cloud.size());
    std::transform(cloud.cbegin(), cloud.cend(), cloud_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    add_cloud_to_map(cloud_transformed);
    // RemovePointsFarFromLocation(origin);
}

// This is SLAM
std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, Sophus::SE3d>
register_frame(const std::vector<Eigen::Vector3d> &cloud) {
    // Preprocess the input cloud
    // const auto &cloud_cropped = preprocess(cloud, max_range_, min_range_);    // lidar의 range
    // (min, max)
    const auto &cloud_cropped = cloud;
    // Voxelize
    // std::cout<<cloud.size()<<"\n";
    // const auto cloud_downsampled = voxelize_downsample(cloud_cropped, voxel_size_*0.5);  //
    // map으로 저장할것 const auto cloud_source = voxelize_downsample(cloud_cropped,
    // voxel_size_*1.5);       // 연산에 쓸거 std::cout<<cloud_downsampled.size()<<"\n";
    // std::cout<<cloud_source.size()<<"\n";
    const auto cloud_downsampled = cloud_cropped;
    const auto cloud_source = cloud_cropped;
    const double sigma = get_adaptive_threshold(); // Need to see

    const auto T_initial_guess = slam_ctx.motion_model(); // identity 아니면 직전 transformation
    const auto last_pose = !slam_ctx.poses_.empty() ? slam_ctx.poses_.back() : Sophus::SE3d();
    const auto pose_initial_guess = last_pose * T_initial_guess;

    const Sophus::SE3d new_pose = register_frame_icp(cloud_source, slam_ctx.map_,
                                                     pose_initial_guess, 3.0 * sigma, sigma / 3.0);
    const auto model_deviation = pose_initial_guess.inverse() * new_pose;
    model_deviation_ = model_deviation;

    // /poses_.push_back(new_pose);
    return std::make_tuple(cloud_downsampled, cloud_source, new_pose);
}

inline std::string FixFrameId(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}

inline std::unique_ptr<sensor_msgs::msg::PointCloud2>
CreatePointCloud2Msg(const size_t n_points, const std_msgs::msg::Header &header,
                     bool timestamp = false) {

    auto cloud_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    sensor_msgs::PointCloud2Modifier modifier(*cloud_msg);
    cloud_msg->header = header;
    cloud_msg->header.frame_id = FixFrameId(cloud_msg->header.frame_id);
    cloud_msg->fields.clear();
    int offset = 0;
    offset = addPointField(*cloud_msg, "x", 1, sensor_msgs::msg::PointField::FLOAT32, offset);
    offset = addPointField(*cloud_msg, "y", 1, sensor_msgs::msg::PointField::FLOAT32, offset);
    offset = addPointField(*cloud_msg, "z", 1, sensor_msgs::msg::PointField::FLOAT32, offset);
    offset += sizeOfPointField(sensor_msgs::msg::PointField::FLOAT32);
    if (timestamp) {
        // asuming timestamp on a velodyne fashion for now (between 0.0 and 1.0)
        offset =
            addPointField(*cloud_msg, "time", 1, sensor_msgs::msg::PointField::FLOAT64, offset);
        offset += sizeOfPointField(sensor_msgs::msg::PointField::FLOAT64);
    }

    // Resize the point cloud accordingly
    cloud_msg->point_step = offset;
    cloud_msg->row_step = cloud_msg->width * cloud_msg->point_step;
    cloud_msg->data.resize(cloud_msg->height * cloud_msg->row_step);
    modifier.resize(n_points);
    return cloud_msg;
}

inline void FillPointCloud2XYZ(const std::vector<Eigen::Vector3d> &cloud,
                               sensor_msgs::msg::PointCloud2 &msg) {
    sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
    for (size_t i = 0; i < cloud.size(); i++, ++msg_x, ++msg_y, ++msg_z) {
        const Eigen::Vector3d &point = cloud[i];
        *msg_x = point.x();
        *msg_y = point.y();
        *msg_z = point.z();
    }
}

inline std::unique_ptr<sensor_msgs::msg::PointCloud2>
EigenToPointCloud2(const std::vector<Eigen::Vector3d> &cloud, const std_msgs::msg::Header &header) {
    auto msg = CreatePointCloud2Msg(cloud.size(), header);
    FillPointCloud2XYZ(cloud, *msg);
    return msg;
}

// cv::Mat intrinsic_left =
//     (cv::Mat_<double>(3, 3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);
// cv::Mat intrinsic_right =
//     (cv::Mat_<double>(3, 3) << 457.587, 0, 379.999, 0, 456.134, 255.238, 0, 0, 1);
// cv::Mat calib_left =
//     (cv::Mat_<double>(3, 4) << 0.0148655429818, -0.999880929698, 0.00414029679422,
//     -0.0216401454975,
//      0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768, -0.0257744366974,
//      0.00375618835797, 0.999660727178, 0.00981073058949);
// cv::Mat calib_right =
//     (cv::Mat_<double>(3, 4) << 0.0125552670891, -0.999755099723, 0.0182237714554,
//     -0.0198435579556,
//      0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024, -0.0253898008918,
//      0.0179005838253, 0.999517347078, 0.00786212447038);

// MH_01_easy.bag
cv::Mat K_left = (cv::Mat_<double>(3, 3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);

cv::Mat K_right = (cv::Mat_<double>(3, 3) << 457.587, 0, 379.999, 0, 456.134, 255.238, 0, 0, 1);
cv::Mat C_left =
    (cv::Mat_<double>(3, 4) << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
     0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768, -0.0257744366974,
     0.00375618835797, 0.999660727178, 0.00981073058949);
cv::Mat C_right =
    (cv::Mat_<double>(3, 4) << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
     0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024, -0.0253898008918,
     0.0179005838253, 0.999517347078, 0.00786212447038);
cv::Mat P_left = K_left * C_left;
cv::Mat P_right = K_right * C_left;
cv::Mat D_left = (cv::Mat_<double>(5, 1) << -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05);
cv::Mat D_right = (cv::Mat_<double>(5, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

// cv::Mat projection_left = intrinsic_left*calib_left;
// cv::Mat projection_right = intrinsic_right*calib_right;

std::vector<Eigen::Vector3d>
triangulate_points(const std::vector<cv::Point2d> matched_keypoints_left,
                   const std::vector<cv::Point2d> matched_keypoints_right) {
    std::vector<Eigen::Vector3d> frame_points;
    cv::Mat triangulated_points;
    cv::triangulatePoints(P_left, P_right, matched_keypoints_left, matched_keypoints_right,
                          triangulated_points);
    frame_points.reserve(triangulated_points.cols);
    for (int point_idx = 0; point_idx < triangulated_points.cols; point_idx++) {
        auto point = triangulated_points.col(point_idx);
        frame_points.emplace_back(point.at<double>(0), point.at<double>(1), point.at<double>(2));
    }
    return frame_points;
}

class ImageSubscriberNode : public rclcpp::Node {

  public:
    ImageSubscriberNode() : Node("exact_time_subscriber") {
        // subscription_pointcloud = create_subscription<sensor_msgs::msg::PointCloud2>(
        //     "pointcloud_topic", rclcpp::SensorDataQoS(),
        //     std::bind(&ImageSubscriberNode::RegisterFrame, this, std::placeholders::_1));

        subscription_left_image.subscribe(this, "/cam0/image_raw");
        subscription_right_image.subscribe(this, "/cam1/image_raw");
        sync_ = std::make_shared<
            message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(
            subscription_left_image, subscription_right_image, 3);
        sync_->registerCallback(std::bind(&ImageSubscriberNode::stereo_image_callback, this,
                                          std::placeholders::_1, std::placeholders::_2));

        rclcpp::QoS qos(rclcpp::KeepLast{100});
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("odometry", qos);
        frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("frame", qos);
        map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("local_map", qos);
    }

  private:
    void stereo_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg_left,
                               const sensor_msgs::msg::Image::ConstSharedPtr &img_msg_right) {
        cv::Mat image_left(img_msg_left->height, img_msg_left->width, CV_8UC1,
                           const_cast<unsigned char *>(img_msg_left->data.data()),
                           img_msg_left->step);

        cv::Mat image_right(img_msg_right->height, img_msg_right->width, CV_8UC1,
                            const_cast<unsigned char *>(img_msg_right->data.data()),
                            img_msg_right->step);

        // call by reference -> return

        equalizeStereoHist(image_left, image_right, 1, false);

        std::vector<cv::Point2f> matched_left, matched_right;
        // 50 -> config->min_matched_num
        obtainCorrespondingPoints(image_left, image_right, matched_left, matched_right, 200, false);
        std::vector<cv::Point2f> undistort_left, undistort_right;
        undistortKeyPoints(matched_left, matched_right, undistort_left, undistort_right, K_left,
                           K_right, D_left, D_right);

        std::vector<cv::Point3f> matched_left_homogeneous, matched_right_homogeneous;
        cv::convertPointsToHomogeneous(matched_left, matched_left_homogeneous);
        cv::convertPointsToHomogeneous(matched_right, matched_right_homogeneous);

        Eigen::MatrixXd matched_left_eigen = convertToEigenMatrix(matched_left_homogeneous);
        Eigen::MatrixXd matched_right_eigen = convertToEigenMatrix(matched_right_homogeneous);

        Eigen::MatrixXd F =
            computeFundamentalmatrixNormalized(matched_left_eigen, matched_right_eigen);
        Eigen::Vector3d p1 = matched_left_eigen.row(0);
        Eigen::Vector3d p2 = matched_right_eigen.row(0);

        Eigen::Vector3d e1 = compute_epipole(F);
        Eigen::Vector3d e2 = compute_epipole(F.transpose());

        // img2 is image right
        std::pair<Eigen::Matrix3d, Eigen::Matrix3d> homographies = compute_matching_homographies(
            e2, F, image_right, matched_left_eigen, matched_right_eigen);

        Eigen::MatrixXd new_points1 =
            divideByZ(homographies.first * matched_left_eigen.transpose()).transpose();
        Eigen::MatrixXd new_points2 =
            divideByZ(homographies.second * matched_right_eigen.transpose()).transpose();

        // End Pre-processing

        // Convert homogeneous points into non-homogeneous points
        int matched_num = new_points1.rows();
        std::vector<cv::Point2d> matched_keypoints_left, matched_keypoints_right;
        matched_keypoints_left.reserve(matched_num);
        matched_keypoints_right.reserve(matched_num);
        for (int i = 0; i < matched_num; ++i) {
            double x = new_points1(i, 0) / new_points1(i, 2);
            double y = new_points1(i, 1) / new_points1(i, 2);
            matched_keypoints_left.emplace_back(cv::Point2d(x, y));
        }

        for (int i = 0; i < matched_num; ++i) {
            double x = new_points2(i, 0) / new_points2(i, 2);
            double y = new_points2(i, 1) / new_points2(i, 2);
            matched_keypoints_right.emplace_back(cv::Point2d(x, y));
        }

        auto triangulated_points =
            triangulate_points(matched_keypoints_left, matched_keypoints_right);

        auto [cloud_map, cloud_keypoints, new_pose] = register_frame(triangulated_points);
        RCLCPP_INFO(this->get_logger(), "cloud_map size: %d cloud_keypoints size: %d",
                    cloud_map.size(), cloud_keypoints.size());
        RCLCPP_INFO(this->get_logger(), "new pose", new_pose.matrix());
        update_map(cloud_map, new_pose);
        slam_ctx.poses_.push_back(new_pose);

        Eigen::Vector3d translation = new_pose.matrix().block<3, 1>(0, 3);
        Eigen::Matrix3d rotation = new_pose.matrix().block<3, 3>(0, 0);

        // Convert rotation matrix to unit quaternion
        Eigen::Quaterniond quaternion(rotation);
        geometry_msgs::msg::TransformStamped transform_msg;

        transform_msg.header.stamp = this->now();
        transform_msg.header.frame_id = odom_frame_;
        transform_msg.child_frame_id = child_frame_;
        transform_msg.transform.rotation.x = quaternion.x();
        transform_msg.transform.rotation.y = quaternion.y();
        transform_msg.transform.rotation.z = quaternion.z();
        transform_msg.transform.rotation.w = quaternion.w();
        transform_msg.transform.translation.x = translation.x();
        transform_msg.transform.translation.y = translation.y();
        transform_msg.transform.translation.z = translation.z();
        tf_broadcaster_->sendTransform(transform_msg);

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = this->now();
        odom_msg.header.frame_id = odom_frame_;
        odom_msg.child_frame_id = child_frame_;
        odom_msg.pose.pose.orientation.x = quaternion.x();
        odom_msg.pose.pose.orientation.y = quaternion.y();
        odom_msg.pose.pose.orientation.z = quaternion.z();
        odom_msg.pose.pose.orientation.w = quaternion.w();
        odom_msg.pose.pose.position.x = translation.x();
        odom_msg.pose.pose.position.y = translation.y();
        odom_msg.pose.pose.position.z = translation.z();
        odom_publisher_->publish(odom_msg);

        sensor_msgs::msg::PointCloud2 frame_msg;
        frame_msg.header.stamp = this->now();
        frame_msg.header.frame_id = child_frame_;
        frame_publisher_->publish(std::move(EigenToPointCloud2(cloud_keypoints, frame_msg.header)));

        sensor_msgs::msg::PointCloud2 map_msg;
        map_msg.header.stamp = this->now();
        map_msg.header.frame_id = odom_frame_;
        // auto local_map_header = msg->header;
        // local_map_header.frame_id = odom_frame_;
        map_publisher_->publish(
            std::move(EigenToPointCloud2(slam_ctx.get_registered_map(), map_msg.header)));
    }

  public:
  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_pointcloud;
    message_filters::Subscriber<sensor_msgs::msg::Image> subscription_left_image;
    message_filters::Subscriber<sensor_msgs::msg::Image> subscription_right_image;
    std::shared_ptr<
        message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>
        sync_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // what is tf broadcasters role?

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
    std::string odom_frame_{"world"};
    std::string child_frame_{"base_link"};
    std::string map_frame_{"frame"};
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    cv::namedWindow("view");
    cv::startWindowThread();
    cv::namedWindow("histogram equalization");
    cv::startWindowThread();
    cv::namedWindow("Matched Features");
    cv::startWindowThread();
    cv::namedWindow("result");
    cv::startWindowThread();

    auto node = std::make_shared<ImageSubscriberNode>();

    rclcpp::spin(node);
    rclcpp::shutdown();

    node = nullptr;

    return 0;
}