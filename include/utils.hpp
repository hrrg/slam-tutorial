#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

Eigen::Matrix3d accumulate_imu(const std::vector<Eigen::Vector3d>& imu_queue) {
    Eigen::Matrix3d cumulative_rotation = Eigen::Matrix3d::Identity();  // Initialize to identity matrix

    for (const Eigen::Vector3d& angular_vector : imu_queue) {
        // Convert angular vector to rotation matrix using the exponential map
        Eigen::Matrix3d delta_rotation;
        delta_rotation = Eigen::AngleAxisd(angular_vector.norm(), angular_vector.normalized());

        // Accumulate the rotation
        cumulative_rotation = cumulative_rotation * delta_rotation;
    }

    return cumulative_rotation;
}
