#pragma once
#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <vector>
#include <regex>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include <sophus/se3.hpp>
Eigen::Vector3d accumulate_imu(const std::vector<Eigen::Vector3d>& imu_queue, const std::vector<double> duration) {
    Eigen::Vector3d rotation;
    rotation.setZero();
    for (size_t i = 0; i < imu_queue.size(); ++i) {
        Eigen::Vector3d angular_velocity = imu_queue[i];
        double time_delta = duration[i];        
        rotation += time_delta*angular_velocity;
    }
    return rotation;
}

inline std::vector<Eigen::Vector3d> PointCloud2ToEigen(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(msg->height * msg->width);
    sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
    for (size_t i = 0; i < msg->height * msg->width; ++i, ++msg_x, ++msg_y, ++msg_z) {
        points.emplace_back(*msg_x, *msg_y, *msg_z);
    }
    return points;
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

inline std::unique_ptr<sensor_msgs::msg::PointCloud2>
EigenToPointCloud2(const std::vector<Eigen::Vector3d> &cloud,
                   const std_msgs::msg::Header &header) {
    auto msg = CreatePointCloud2Msg(cloud.size(), header);
    FillPointCloud2XYZ(cloud, *msg);
    return msg;
}