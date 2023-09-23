#define FMT_HEADER_ONLY
#include "kiss_icp.hpp"
#include <fmt/format.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tsl/robin_map.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
} // namespace Eigen

namespace kiss_icp {
SLAM_Context slam_ctx;
// TODO: Manage parameters
// Constant Parameters
constexpr int ESTIMATION_THRESHOLD_ = 0.0001;
constexpr int MAX_NUM_ITERATIONS_ = 100;
// Configurable Parameters
double initial_threshold_ = 2.0;   // adptive
const double min_motion_th_ = 0.1; // constant

const double mid_pose_timestamp = 0.05;
double model_error_sse2_ = 0;
int num_samples_ = 0;
const double min_range_ = 5.0;   // constant
const double max_range_ = 100.0; // constant
const double voxel_size_ = 1;    // constant
double max_distance_ = 100.0;
int max_points_per_voxel_ = 20;

Sophus::SE3d model_deviation_ = Sophus::SE3d();

// ICP

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
    transform_cloud(pose_initial_guess, source); // point to point x point to map

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
    RemovePointsFarFromLocation(origin);
}
////////////////////////////////////////// Unverified code
std::vector<Eigen::Vector3d>
deskew_pointcloud(const std::vector<Eigen::Vector3d> &cloud,
                  const std::vector<Eigen::Vector3d> imu_angular_velocity,
                  const std::vector<double> &timestamps) {
    std::vector<Eigen::Vector3d> deskewed(cloud.size());

    Eigen::Vector3d rotation;
    rotation.setZero();
    for (size_t i = 0; i < imu_angular_velocity.size(); ++i) {
        Eigen::Vector3d angular_velocity = imu_angular_velocity[i];
        double time_delta = timestamps[i];
        rotation += time_delta * angular_velocity;
    }
    rotation *= -1;
    std::cout << "rotation" << rotation[0] << rotation[1] << rotation[2] << "\n";
    for (int i = 0; i < cloud.size(); i++) {
        const auto &pt = cloud[i];

        // TODO: transform IMU data into the LIDAR frame
        double delta_t = 0.1 * static_cast<double>(i) / cloud.size();
        Eigen::Quaterniond delta_q(
            Eigen::AngleAxisd(delta_t / 2.0 * rotation.norm(), rotation.normalized()));

        Eigen::Vector3d pt_ = delta_q.conjugate() * pt;

        deskewed[i] = pt_;
    }

    return deskewed;
}

// This is SLAM
std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>, Sophus::SE3d> register_frame(
    const std::vector<Eigen::Vector3d> &cloud, const std::vector<Eigen::Vector3d> &T_deskew,
    const std::vector<Eigen::Vector3d> &T_motion, const std::vector<double> &timestamps) {
    auto T_initial_guess = slam_ctx.motion_model(); // identity 아니면 직전 transformation
    const auto &deskewed_cloud = deskew_pointcloud(cloud, T_deskew, timestamps);
    // Preprocess the input cloud
    const auto &cloud_cropped =
        preprocess(deskewed_cloud, max_range_, min_range_); // lidar의 range (min, max)
    // Voxelize

    const auto cloud_downsampled =
        voxelize_downsample(cloud_cropped, voxel_size_ * 0.5); // map으로 저장할것
    const auto cloud_source = voxelize_downsample(cloud_cropped, voxel_size_ * 1.5); // 연산에 쓸거

    const double sigma = get_adaptive_threshold(); // Need to see

    const auto last_pose = !slam_ctx.poses_.empty() ? slam_ctx.poses_.back() : Sophus::SE3d();
    const auto pose_initial_guess = last_pose * T_initial_guess;

    const Sophus::SE3d new_pose = register_frame_icp(cloud_source, slam_ctx.map_,
                                                     pose_initial_guess, 3.0 * sigma, sigma / 3.0);
    const auto model_deviation = pose_initial_guess.inverse() * new_pose;
    model_deviation_ = model_deviation;

    // /poses_.push_back(new_pose);
    return std::make_tuple(cloud_downsampled, cloud_source, new_pose);
}

} // namespace kiss_icp