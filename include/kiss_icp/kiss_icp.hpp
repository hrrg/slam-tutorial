#pragma once
#include <tuple>
#include <vector>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <tsl/robin_map.h>
namespace kiss_icp
{
extern int max_points_per_voxel_;
using Voxel = Eigen::Vector3i;

struct VoxelBlock {
    // buffer of points with a max limit of n_points
    std::vector<Eigen::Vector3d> points;
    int num_points_;
    inline void AddPoint(const Eigen::Vector3d &point) {
        if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
    }
};

struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};

class SLAM_Context{
public:
    SLAM_Context(){};
    ~SLAM_Context(){};

    Sophus::SE3d motion_model() const {
        Sophus::SE3d pred = Sophus::SE3d(); // 아무 history 없을때 identity
        const size_t N = poses_.size();
        if (N < 2) return pred;
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
extern SLAM_Context slam_ctx;

std::tuple<std::vector<Eigen::Vector3d>,std::vector<Eigen::Vector3d>,Sophus::SE3d> register_frame(const std::vector<Eigen::Vector3d> &cloud,  const std::vector<Eigen::Vector3d> &T_deskew, const std::vector<Eigen::Vector3d> &T_motion, const std::vector<double> &timestamps);
void update_map(const std::vector<Eigen::Vector3d> cloud, const Sophus::SE3d &pose);
}