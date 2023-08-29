#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <message_filters/subscriber.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>


#include <memory>
#include <tuple>
#include <unistd.h>
#include <chrono>
#include <cstdio>



class ImuSubscriberNode : public rclcpp::Node
{

public:
    ImuSubscriberNode()
      : Node("exact_time_subscriber")
    {
        use_imu_ = true;
        imu_subscriber_ = create_subscription<sensor_msgs::msg::Imu>(
            "imu", rclcpp::SensorDataQoS(), 
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                imu_callback(msg);
            });

        rclcpp::QoS qos(rclcpp::KeepLast{100});
    }
    
private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        if (!use_imu_) {return;}

        double roll, pitch, yaw;
        tf2::Quaternion orientation;
        tf2::fromMsg(msg->orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        float acc_x = static_cast<float>(msg->linear_acceleration.x) + sin(pitch) * 9.81;
        float acc_y = static_cast<float>(msg->linear_acceleration.y) - cos(pitch) * sin(roll) * 9.81;
        float acc_z = static_cast<float>(msg->linear_acceleration.z) - cos(pitch) * cos(roll) * 9.81;

        Eigen::Vector3f angular_velo{
        static_cast<float>(msg->angular_velocity.x),
        static_cast<float>(msg->angular_velocity.y),
        static_cast<float>(msg->angular_velocity.z)};
        Eigen::Vector3f acc{acc_x, acc_y, acc_z};
        Eigen::Quaternionf quat{
        static_cast<float>(msg->orientation.w),
        static_cast<float>(msg->orientation.x),
        static_cast<float>(msg->orientation.y),
        static_cast<float>(msg->orientation.z)};
        double imu_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        preprocess(angular_velo, acc, quat, imu_time);
    }
    void preprocess(Eigen::Vector3f angular_velo, Eigen::Vector3f acc, const Eigen::Quaternionf quat,const double imu_time);

    
private:
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;

    bool use_imu_;
    double scan_period_{0.1};
    static const int imu_que_length_{200};
    int imu_ptr_front_{0}, imu_ptr_last_{-1}, imu_ptr_last_iter_{0};

    std::array<double, imu_que_length_> imu_time_;
    std::array<float, imu_que_length_> imu_roll_;
    std::array<float, imu_que_length_> imu_pitch_;
    std::array<float, imu_que_length_> imu_yaw_;

    std::array<float, imu_que_length_> imu_acc_x_;
    std::array<float, imu_que_length_> imu_acc_y_;
    std::array<float, imu_que_length_> imu_acc_z_;
    std::array<float, imu_que_length_> imu_velo_x_;
    std::array<float, imu_que_length_> imu_velo_y_;
    std::array<float, imu_que_length_> imu_velo_z_;
    std::array<float, imu_que_length_> imu_shift_x_;
    std::array<float, imu_que_length_> imu_shift_y_;
    std::array<float, imu_que_length_> imu_shift_z_;

    std::array<float, imu_que_length_> imu_angular_velo_x_;
    std::array<float, imu_que_length_> imu_angular_velo_y_;
    std::array<float, imu_que_length_> imu_angular_velo_z_;
    std::array<float, imu_que_length_> imu_angular_rot_x_;
    std::array<float, imu_que_length_> imu_angular_rot_y_;
    std::array<float, imu_que_length_> imu_angular_rot_z_;  
};


void ImuSubscriberNode::preprocess(
    Eigen::Vector3f angular_velo, Eigen::Vector3f acc, const Eigen::Quaternionf quat,
    const double imu_time /*[sec]*/)
  {
    float roll, pitch, yaw;
    Eigen::Affine3f affine(quat);
    Eigen::Vector3f euler_angles = quat.toRotationMatrix().eulerAngles(0, 1, 2);
    roll = euler_angles[0];
    pitch = euler_angles[1];
    yaw = euler_angles[2];
    imu_ptr_last_ = (imu_ptr_last_ + 1) % imu_que_length_;

    if ((imu_ptr_last_ + 1) % imu_que_length_ == imu_ptr_front_) {
      imu_ptr_front_ = (imu_ptr_front_ + 1) % imu_que_length_;
    }

    imu_time_[imu_ptr_last_] = imu_time;
    imu_roll_[imu_ptr_last_] = roll;
    imu_pitch_[imu_ptr_last_] = pitch;
    imu_yaw_[imu_ptr_last_] = yaw;
    imu_acc_x_[imu_ptr_last_] = acc.x();
    imu_acc_y_[imu_ptr_last_] = acc.y();
    imu_acc_z_[imu_ptr_last_] = acc.z();
    imu_angular_velo_x_[imu_ptr_last_] = angular_velo.x();
    imu_angular_velo_y_[imu_ptr_last_] = angular_velo.y();
    imu_angular_velo_z_[imu_ptr_last_] = angular_velo.z();

    Eigen::Matrix3f rot = quat.toRotationMatrix();
    acc = rot * acc;

    int imu_ptr_back = (imu_ptr_last_ - 1 + imu_que_length_) % imu_que_length_;
    double time_diff = imu_time_[imu_ptr_last_] - imu_time_[imu_ptr_back];
    if (time_diff < scan_period_) {
      imu_shift_x_[imu_ptr_last_] =
        imu_shift_x_[imu_ptr_back] + imu_velo_x_[imu_ptr_back] * time_diff + acc(0) * time_diff *
        time_diff * 0.5;
      imu_shift_y_[imu_ptr_last_] =
        imu_shift_y_[imu_ptr_back] + imu_velo_y_[imu_ptr_back] * time_diff + acc(1) * time_diff *
        time_diff * 0.5;
      imu_shift_z_[imu_ptr_last_] =
        imu_shift_z_[imu_ptr_back] + imu_velo_z_[imu_ptr_back] * time_diff + acc(2) * time_diff *
        time_diff * 0.5;

      imu_velo_x_[imu_ptr_last_] = imu_velo_x_[imu_ptr_back] + acc(0) * time_diff;
      imu_velo_y_[imu_ptr_last_] = imu_velo_y_[imu_ptr_back] + acc(1) * time_diff;
      imu_velo_z_[imu_ptr_last_] = imu_velo_z_[imu_ptr_back] + acc(2) * time_diff;

      imu_angular_rot_x_[imu_ptr_last_] = imu_angular_rot_x_[imu_ptr_back] + angular_velo(0) *
        time_diff;
      imu_angular_rot_y_[imu_ptr_last_] = imu_angular_rot_y_[imu_ptr_back] + angular_velo(1) *
        time_diff;
      imu_angular_rot_z_[imu_ptr_last_] = imu_angular_rot_z_[imu_ptr_back] + angular_velo(2) *
        time_diff;
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImuSubscriberNode>();

    rclcpp::spin(node);
    rclcpp::shutdown();

    node = nullptr;

    return 0;
}