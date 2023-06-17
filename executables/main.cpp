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
#include <opencv2/features2d.hpp>
#include <chrono>
#include <unistd.h>


void detect_keypoints(cv::Mat* image_left, cv::Mat* image_right){
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints_left;
    cv::Mat descriptors_left;
    orb->detectAndCompute(*image_left, cv::noArray(), keypoints_left, descriptors_left);
    
    std::vector<cv::KeyPoint> keypoints_right;
    cv::Mat descriptors_right;
    orb->detectAndCompute(*image_right, cv::noArray(), keypoints_right, descriptors_right);

    // Match descriptors
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_left, descriptors_right, matches);
    // Filter matches using ratio test
    std::vector<cv::DMatch> filtered_matches;
    double ratio_threshold = 0.7;
    for (const auto& match : matches)
    {
        if (match.distance < ratio_threshold * matches[match.queryIdx].distance)
            filtered_matches.push_back(match);
    }
    // compute R|t from 
   cv::Mat matched_image;
    cv::drawMatches(*image_left, keypoints_left, *image_right, keypoints_right,
                    filtered_matches, matched_image, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::DEFAULT );
    
    // Display matches
    cv::imshow("Matches", matched_image); 
}


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
  void topic_callback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg_left, const sensor_msgs::msg::Image::ConstSharedPtr& img_msg_right) const
  {
    RCLCPP_INFO(this->get_logger(), "callback");
    cv::Mat image_left(img_msg_left->height, img_msg_left->width, CV_8UC1,
    const_cast<unsigned char*>(img_msg_left->data.data()), img_msg_left->step);
 
    cv::Mat image_right(img_msg_right->height, img_msg_right->width, CV_8UC1,
    const_cast<unsigned char*>(img_msg_right->data.data()), img_msg_right->step);
    detect_keypoints(&image_left, &image_right);
    
    
  }
  message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_1_;
  message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_2_;
  std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>> sync_;
};






int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    cv::namedWindow("view");
    cv::startWindowThread();
    
    auto node = std::make_shared<ImageSubscriberNode>();
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    node = nullptr;

    return 0;
}
