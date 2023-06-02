#include <cstdio>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>

// global variables
//cv::Mat image_left;
//cv::Mat image_right;
class ImageSubscriberNode : public rclcpp::Node
{
public:
  ImageSubscriberNode()
      : Node("exact_time_subscriber")
  {
    subscription_temp_1_.subscribe(this, "/alphasense/cam0/image_raw");
    subscription_temp_2_.subscribe(this, "/alphasense/cam1/image_raw");

    sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(subscription_temp_1_, subscription_temp_2_, 3);
    sync_->registerCallback(std::bind(&ImageSubscriberNode::topic_callback, this, std::placeholders::_1, std::placeholders::_2));
  }

private:
  void topic_callback(const sensor_msgs::msg::Image::ConstSharedPtr& tmp_1, const sensor_msgs::msg::Image::ConstSharedPtr& tmp_2) const
  {
    cv::Mat frame_left(tmp_1->height, tmp_1->width, CV_8UC1,
    const_cast<unsigned char*>(tmp_1->data.data()), tmp_1->step);
 
    cv::Mat frame_right(tmp_2->height, tmp_2->width, CV_8UC1,
    const_cast<unsigned char*>(tmp_2->data.data()), tmp_2->step);

    cv::Mat frames;
    cv::hconcat(frame_left, frame_right, frames);
    cv::imshow("view", frames);
    cv::waitKey(1);
  }
  message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_1_;
  message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_2_;
  std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>> sync_;
};

void callback_image(const sensor_msgs::msg::Image::ConstSharedPtr & image) {
    cv::Mat frame(image->height, image->width, CV_8UC1,
    const_cast<unsigned char*>(image->data.data()), image->step);
    std::cout << image->height << "," << image->width <<"\n";
    cv::imshow("view", frame);
    //cv::waitKey(0.1);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    cv::namedWindow("view");
    cv::startWindowThread();
    //auto node = rclcpp::Node::make_shared("main_node");
    auto node = std::make_shared<ImageSubscriberNode>();
    //CLCPP_INFO(node->get_logger(), "Image Subscriber Test");
    
    //image_transport::ImageTransport it(node);
    //image_transport::Subscriber image_left_sub = it.subscribe("/alphasense/cam0/image_raw", 1, callback_image);
    //image_transport::Subscriber image_right_sub = it.subscribe("/alphasense/cam3/image_raw", 1, callback_image);
    
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    node = nullptr;

    return 0;
}
