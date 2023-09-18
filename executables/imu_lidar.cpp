#include "message_filters/subscriber.h"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"


#include <tuple>
#include <memory>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <mutex>

#include <Eigen/Core>
#include "utils.hpp"
#include "kiss_icp/kiss_icp.hpp"


const int IMU_QUEUE_SIZE = 300;
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
        
        rclcpp::QoS qos(rclcpp::KeepLast{100});
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("odometry", qos);
        frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("scan", qos);
        map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("local_map", qos);
        path_publisher_ = create_publisher<nav_msgs::msg::Path>("path", qos);

        path_msg_.header.frame_id = odom_frame_;
        
        std::fill(std::begin(imu_timestamp_queue_), std::end(imu_timestamp_queue_), 0);
        
    }
private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        //RCLCPP_INFO(this->get_logger(), "imu callback at %lf", timestamp);
        Eigen::Vector3d angular_velocity{static_cast<double>(msg->angular_velocity.x),
                                         static_cast<double>(msg->angular_velocity.y),
                                         static_cast<double>(msg->angular_velocity.z)};
        Eigen::Vector3d acceleration{static_cast<double>(msg->linear_acceleration.x),
                                         static_cast<double>(msg->linear_acceleration.y),
                                         static_cast<double>(msg->linear_acceleration.z)};
        std::unique_lock<std::mutex> lock2(mutex2_);
        acceleration[1] = acceleration[1]  - sin(rpy[1]) * cos(rpy[0]) * 9.81;
        acceleration[2] = acceleration[2]  + cos(rpy[1]) * cos(rpy[0]) * 9.81;
        acceleration[0] = acceleration[0]   + sin(rpy[1]) * 9.81;
;
        lock2.unlock();
        std::unique_lock<std::mutex> lock(mutex_);
        
        imu_acc_queue_[(imu_queue_cursor_)] = acceleration; // REMOVE
        
        imu_se3_queue_[(imu_queue_cursor_)] = angular_velocity;
        imu_timestamp_queue_[(imu_queue_cursor_)] = timestamp;
        imu_queue_cursor_ = (imu_queue_cursor_+1)%IMU_QUEUE_SIZE;
        lock.unlock();
        /* PREPROCESS */
        int imu_ptr_back = (imu_queue_cursor_ - 1 + IMU_QUEUE_SIZE) % IMU_QUEUE_SIZE;
        
        double time_diff = 0.0025;
        imu_shift[imu_queue_cursor_] = imu_shift[imu_ptr_back] + imu_velocity[imu_ptr_back] * time_diff + acceleration * time_diff * time_diff * 0.5;
        imu_velocity[imu_queue_cursor_] = imu_velocity[imu_ptr_back] + acceleration * time_diff;
        imu_rotation[imu_queue_cursor_] = imu_rotation[imu_ptr_back] + angular_velocity * time_diff;
        
        
        RCLCPP_INFO(this->get_logger(), "%lf %lf %lf",imu_shift[imu_queue_cursor_][0],imu_shift[imu_queue_cursor_][1],imu_shift[imu_queue_cursor_][2]);
        geometry_msgs::msg::TransformStamped transform_msg;
        Eigen::Quaterniond quaternion;
        quaternion = Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX()) *
                 Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ());
        transform_msg.header.stamp = msg->header.stamp;
        transform_msg.header.frame_id = imu_frame;
        transform_msg.child_frame_id = child_frame_;
        transform_msg.transform.rotation.x = quaternion.x();
        transform_msg.transform.rotation.y = quaternion.y();
        transform_msg.transform.rotation.z = quaternion.z();
        transform_msg.transform.rotation.w = quaternion.w();
        transform_msg.transform.translation.x = imu_shift[imu_queue_cursor_][0];
        transform_msg.transform.translation.y = imu_shift[imu_queue_cursor_][1];
        transform_msg.transform.translation.z = imu_shift[imu_queue_cursor_][2];
        tf_broadcaster_->sendTransform(transform_msg);
        

    }
    
    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        std::vector<Eigen::Vector3d> imu_deskew;
        std::vector<Eigen::Vector3d> imu_motion;
        std::vector<double> imu_deskew_timestamp;
        std::vector<double> imu_motion_timestamp;
        //RCLCPP_INFO(this->get_logger(), "lidar callback at %lf %lf", timestamp, prev_timestamp_);
        std::unique_lock<std::mutex> lock(mutex_);
        
        // collect imu data for cloud deskewing
        int imu_idx = imu_queue_cursor_-1;        
        for (int i = 0; i < IMU_QUEUE_SIZE; i++){
            double imu_timestamp = imu_timestamp_queue_[imu_idx];
            double time_diff = imu_timestamp_queue_[imu_idx] - imu_timestamp_queue_[imu_idx-1];
            if (time_diff > 1) continue;
            if (timestamp - LIDAR_SCAN_TIME < imu_timestamp && imu_timestamp < timestamp){
                imu_deskew.push_back(imu_se3_queue_[imu_idx]);
                imu_deskew_timestamp.push_back(time_diff);
            }
            imu_idx--;
            if(imu_idx < 0) imu_idx = IMU_QUEUE_SIZE-1;
        }        
        
        
        // collect imu data for icp initial guess
        imu_idx = imu_queue_cursor_-1;
        for (int i = 0; i < IMU_QUEUE_SIZE; i++){
            double imu_timestamp = imu_timestamp_queue_[imu_idx];
            double time_diff = imu_timestamp_queue_[imu_idx] - imu_timestamp_queue_[imu_idx-1];
            if (time_diff > 1) continue;
            if ( prev_timestamp_ < imu_timestamp && imu_timestamp < timestamp){
                imu_motion.push_back(imu_se3_queue_[imu_idx]);
                imu_motion_timestamp.push_back(time_diff);
            }
            imu_idx--;
            if(imu_idx < 0) imu_idx = IMU_QUEUE_SIZE-1;
        }
        lock.unlock();
        
        prev_timestamp_ = timestamp;
        //auto T_deskew = accumulate_imu(imu_deskew, imu_deskew_timestamp);    
        //auto T_motion = accumulate_imu(imu_motion, imu_motion_timestamp);    
        //RCLCPP_INFO(this->get_logger(), "%lf %lf %lf",T_deskew[0],T_deskew[1],T_deskew[2]);
        //RCLCPP_INFO(this->get_logger(), "%lf %lf %lf",T_motion[0],T_motion[1],T_motion[2]);
        // ICP Scan matching
        const auto points = PointCloud2ToEigen(msg);
        auto [cloud_map, cloud_keypoints, new_pose] = kiss_icp::register_frame(points, imu_deskew, imu_motion, imu_deskew_timestamp);
        kiss_icp::update_map(cloud_map, new_pose);
        kiss_icp::slam_ctx.poses_.push_back(new_pose);

        Eigen::Vector3d translation = new_pose.matrix().block<3, 1>(0, 3);
        Eigen::Matrix3d rotation = new_pose.matrix().block<3, 3>(0, 0);
        

        // Convert rotation matrix to unit quaternion
        Eigen::Quaterniond quaternion(rotation);
        std::unique_lock<std::mutex> lock2(mutex2_);
        rpy = quaternion.toRotationMatrix().eulerAngles(0, 1, 2);
        lock2.unlock();
        geometry_msgs::msg::TransformStamped transform_msg;
        
        transform_msg.header.stamp = msg->header.stamp;
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

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.pose.orientation.x = quaternion.x();
        pose_msg.pose.orientation.y = quaternion.y();
        pose_msg.pose.orientation.z = quaternion.z();
        pose_msg.pose.orientation.w = quaternion.w();
        pose_msg.pose.position.x = translation.x();
        pose_msg.pose.position.y = translation.y();
        pose_msg.pose.position.z = translation.z();
        pose_msg.header.stamp = msg->header.stamp;
        pose_msg.header.frame_id = odom_frame_;
        
        path_msg_.poses.push_back(pose_msg);
        path_publisher_->publish(path_msg_);

        //// publish odometry msg
        auto odom_msg = std::make_unique<nav_msgs::msg::Odometry>();
        odom_msg->header = pose_msg.header;
        odom_msg->header.frame_id = odom_frame_; 
        odom_msg->child_frame_id = child_frame_;
        odom_msg->pose.pose = pose_msg.pose;
        odom_publisher_->publish(std::move(odom_msg));

        //// Publish KISS-ICP internal data, just for debugging
        sensor_msgs::msg::PointCloud2 frame_msg;
        frame_msg.header.stamp = msg->header.stamp;
        frame_msg.header.frame_id = child_frame_; 
        frame_publisher_->publish(std::move(EigenToPointCloud2(cloud_keypoints, frame_msg.header)));
        
        // Map is referenced to the odometry_frame
        sensor_msgs::msg::PointCloud2 map_msg;
        map_msg.header.stamp  = msg->header.stamp;
        map_msg.header.frame_id = odom_frame_;
        map_publisher_->publish(std::move(EigenToPointCloud2(kiss_icp::slam_ctx.get_registered_map(), map_msg.header)));
        
    }   
private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    
    std::mutex mutex_;
    std::mutex mutex2_;
    std::array<Eigen::Vector3d, IMU_QUEUE_SIZE> imu_acc_queue_;
    std::array<Eigen::Vector3d, IMU_QUEUE_SIZE> imu_se3_queue_;
    Eigen::Vector3d rpy;
    std::array<Eigen::Vector3d, IMU_QUEUE_SIZE> imu_shift;
    std::array<Eigen::Vector3d, IMU_QUEUE_SIZE> imu_velocity;
    std::array<Eigen::Vector3d, IMU_QUEUE_SIZE> imu_rotation;
    

    std::array<double, IMU_QUEUE_SIZE> imu_timestamp_queue_;
    int imu_queue_cursor_ = 0;
    double prev_timestamp_ = 0;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // what is tf broadcasters role?

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    nav_msgs::msg::Path path_msg_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
    std::string imu_frame{"imu_link"};
    std::string odom_frame_{"lidar_link"};
    std::string child_frame_{"base_link"};
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
