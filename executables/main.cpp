#include <cstdio>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <message_filters/subscriber.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>

void sub_callback(const sensor_msgs::msg::Image::SharedPtr left) {
    cv::Mat frame(left->height, left->width, CV_8UC1,
        const_cast<unsigned char*>(left->data.data()), left->step);
    std::cout << left->height << "," << left->width <<"\n";
    cv::imshow("view", frame);
    cv::waitKey(1);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("main_node");

    cv::namedWindow("view");
    cv::startWindowThread();

    RCLCPP_INFO(node->get_logger(), "Image Subscriber Test");
    
    auto image_subscriber = node->create_subscription<sensor_msgs::msg::Image>(
        "/alphasense/cam0/image_raw", 1000, sub_callback);
    //message_filters::Subscriber<sensor_msgs::msg::Image> image_1(node, "/alphasense/cam0/image_raw");

    rclcpp::spin(node);
    rclcpp::shutdown();
    image_subscriber = nullptr;
    node = nullptr;

    return 0;
}
