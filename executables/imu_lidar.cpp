#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "message_filters/subscriber.h"

#include <tuple>
#include <memory>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <mutex>

#include <Eigen/Core>
#include "utils.hpp"

const int IMU_QUEUE_SIZE = 1000;
const double LIDAR_SCAN_TIME = 0.1;
const double IMU_SCAN_TIME = 0.003;

class SubscriberNode : public rclcpp::Node
{   
public:
    SubscriberNode() : Node("imu_lidar_subscriber"){
        imu_subscriber_ = create_subscription<sensor_msgs::msg::Imu>("/alphasense/imu", rclcpp::SensorDataQoS(), 
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {imu_callback(msg);});
        lidar_subscriber_ = create_subscription<sensor_msgs::msg::PointCloud2>(
             "/hesai/pandar", rclcpp::SensorDataQoS(), std::bind(&SubscriberNode::lidar_callback, this, std::placeholders::_1));
    }
private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        RCLCPP_INFO(this->get_logger(), "imu callback at %lf", timestamp);
        Eigen::Vector3d angular_velocity{static_cast<double>(msg->angular_velocity.x),
                                         static_cast<double>(msg->angular_velocity.y),
                                         static_cast<double>(msg->angular_velocity.z)};
        std::unique_lock<std::mutex> lock(mutex_);
        imu_se3_queue_[(imu_queue_cursor_)%IMU_QUEUE_SIZE] = angular_velocity;
        imu_timestamp_queue_[(imu_queue_cursor_)%IMU_QUEUE_SIZE] = timestamp;
        imu_queue_cursor_++;
        lock.unlock();
    }

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        std::vector<Eigen::Vector3d> imu_deskew;
        std::vector<Eigen::Vector3d> imu_motion;
        RCLCPP_INFO(this->get_logger(), "lidar callback at %lf %lf", timestamp, prev_timestamp_);
        std::unique_lock<std::mutex> lock(mutex_);
        
        // collect imu data for cloud deskewing
        int imu_idx = imu_queue_cursor_-1;
        for (int i = 0; i < IMU_QUEUE_SIZE; i++){
            double imu_timestamp = imu_timestamp_queue_[imu_idx];
            if ( timestamp - LIDAR_SCAN_TIME < imu_timestamp && imu_timestamp < timestamp)
                imu_deskew.push_back(imu_se3_queue_[imu_idx]);
            imu_idx--;
            if(imu_idx < 0) imu_idx = IMU_QUEUE_SIZE-1;
        }        

        // collect imu data for icp initial guess
        imu_idx = imu_queue_cursor_-1;
        for (int i = 0; i < IMU_QUEUE_SIZE; i++){
            double imu_timestamp = imu_timestamp_queue_[imu_idx];
            if ( prev_timestamp_ < imu_timestamp && imu_timestamp < timestamp)
                imu_motion.push_back(imu_se3_queue_[imu_idx]);
            imu_idx--;
            if(imu_idx < 0) imu_idx = IMU_QUEUE_SIZE-1;
        }

        auto T_deskew = accumulate_imu(imu_deskew);
        auto T_motion = accumulate_imu(imu_motion);
        //std::ostringstream oss;
        //oss << T_deskew;
        //RCLCPP_INFO(this->get_logger(),"T_deskew %s", oss.str().c_str());
        //oss << T_motion;
        //RCLCPP_INFO(this->get_logger(),"T_motion %s", oss.str().c_str());
        
        lock.unlock();
        prev_timestamp_ = timestamp;
    }   
private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    
    std::mutex mutex_;
    std::array<Eigen::Vector3d, IMU_QUEUE_SIZE> imu_se3_queue_;
    std::array<double, IMU_QUEUE_SIZE> imu_timestamp_queue_;
    int imu_queue_cursor_ = 0;
    double prev_timestamp_ = 0;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SubscriberNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();

    return 0;
}
