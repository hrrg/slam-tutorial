#include <cstdio>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "nav_msgs/msg/odometry.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <chrono>
#include <unistd.h>

cv::Mat camera_matrix_left = (cv::Mat_<double>(3,4) << 457.587, 0, 379.999, -0.0216401454975,
                                                         0, 456.134, 255.238,  -0.064676986768,
                                                         0, 0, 1, 0.00981073058949);
    
cv::Mat camera_matrix_right = (cv::Mat_<double>(3,4) << 458.654, 0, 369.215, -0.0198435579556,
                                                         0, 457.296, 248.375,  0.0453689425024,
                                                         0, 0, 1, 0.00786212447038); 

class Frame
{
public:
    Frame(){
        transformation.setIdentity();
        world_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    };
    ~Frame(){};
public:
    Eigen::Matrix4f transformation;   // Rotation and translation of Frame
    std::vector<cv::KeyPoint> keypoints_left;
    std::vector<cv::KeyPoint> keypoints_right;
    std::vector<cv::Point2d>  matched_keypoints_left;
    std::vector<cv::Point2d>  matched_keypoints_right;
    pcl::PointCloud<pcl::PointXYZ>::Ptr world_points;
};

class Tracker
{
public:
    Tracker(){
        feature_detector = cv::ORB::create();
    }
    ~Tracker(){};
    void detect_keypoints(cv::Mat* image_left, cv::Mat* image_right);
    void triangulate_points();
    Eigen::Matrix4f icp();
    void add_transformations(Eigen::Matrix4f t);
    void update_camera_trajectory();
    void print_last_camera_position();
public:
    cv::Ptr<cv::Feature2D> feature_detector; 
    std::vector<Frame> frames;
    std::vector<Eigen::Matrix4f> world_to_camera_transformation;
    std::vector<Eigen::Matrix4f> camera_to_world_transformation;
    std::vector<Eigen::Vector3f> camera_trajectory;
};

void Tracker::detect_keypoints(cv::Mat* image_left, cv::Mat* image_right) {
    Frame frame;
    cv::Mat descriptors_left;
    cv::Mat descriptors_right;
    
    // Feature detection
    feature_detector->detectAndCompute(*image_left, cv::noArray(), frame.keypoints_left, descriptors_left);    
    feature_detector->detectAndCompute(*image_right, cv::noArray(), frame.keypoints_right, descriptors_right);

    // Match descriptors
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_left, descriptors_right, matches);

    // Filter matches using ratio test
    std::vector<cv::DMatch> filtered_matches;
    //double ratio_threshold = 0.1;
    for (const auto& match : matches)
    {
        if (match.distance < 50){
            filtered_matches.push_back(match);
        }
    }
       
    // Retrieve 3D points for matched keypoints
    for (const cv::DMatch& match : filtered_matches) {
        cv::Point2f pt1 = frame.keypoints_left[match.queryIdx].pt;
        cv::Point2f pt2 = frame.
        
        keypoints_right[match.trainIdx].pt;
        frame.matched_keypoints_left.push_back(pt1);
        frame.matched_keypoints_right.push_back(pt2);     
    }
    frames.push_back(frame);
}

void Tracker::triangulate_points(){
    cv::Mat world_points;
    auto frame = frames.back();
    cv::triangulatePoints(camera_matrix_left, camera_matrix_right, frame.matched_keypoints_left, frame.matched_keypoints_right, world_points);        
    
    for (int world_point_idx = 0; world_point_idx < world_points.cols ; world_point_idx++){
        auto point = world_points.col(world_point_idx);
        pcl::PointXYZ p(point.at<double>(0),point.at<double>(1),point.at<double>(2));        
        frame.world_points->push_back(p);
    }
}

Eigen::Matrix4f Tracker::icp() {
    // run icp
    if(frames.size() < 2) return Eigen::Matrix4f::Identity();
    auto current_frame = frames.back();
    auto previous_frame = frames.at(frames.size() - 2);
    
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setMaxCorrespondenceDistance(1.0);
    icp.setTransformationEpsilon(0.001);
    icp.setMaximumIterations(1000);

    pcl::PointCloud<pcl::PointXYZ>::Ptr align(new pcl::PointCloud<pcl::PointXYZ>);
    
    icp.setInputSource(current_frame.world_points);
    icp.setInputTarget(previous_frame.world_points);
    icp.align(*align);

    double score  = icp.getFitnessScore();
    bool is_converged = icp.hasConverged();

    return icp.getFinalTransformation();
}

void Tracker::add_transformations(Eigen::Matrix4f t)
{
    // The first frame is treated as the world frame
    if (camera_to_world_transformation.size() < 1) {
        world_to_camera_transformation.push_back(t);
        camera_to_world_transformation.push_back(t.inverse());
        return;
    }

    auto last_w2c_t = world_to_camera_transformation.back();
    Eigen::Matrix4f current_w2c_t = last_w2c_t * t;
    world_to_camera_transformation.push_back(current_w2c_t);
    camera_to_world_transformation.push_back(current_w2c_t.inverse());
}

void Tracker::update_camera_trajectory()
{
    // World coordinate of camera:
    // p_w = T_wc * 0_c = T_wc.translation()
    auto last_c2w_t = camera_to_world_transformation.back();
    camera_trajectory.push_back(last_c2w_t.block<3, 1>(0, 3));
}

void Tracker::print_last_camera_position()
{
    // TODO(heejoon1130@gmail.com)
}

class ImageSubscriberNode : public rclcpp::Node
{

public:
    ImageSubscriberNode()
      : Node("exact_time_subscriber")
    {
        subscription_temp_1_.subscribe(this, "/cam0/image_raw");
        subscription_temp_2_.subscribe(this, "/cam1/image_raw");  
        sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(subscription_temp_1_, subscription_temp_2_, 3);
        sync_->registerCallback(std::bind(&ImageSubscriberNode::topic_callback, this, std::placeholders::_1, std::placeholders::_2));
        
        rclcpp::QoS qos(rclcpp::KeepLast{200});
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("odometry", qos);
    }

private:
    void topic_callback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg_left, const sensor_msgs::msg::Image::ConstSharedPtr& img_msg_right)
    {        
        cv::Mat image_left(img_msg_left->height, img_msg_left->width, CV_8UC1,
        const_cast<unsigned char*>(img_msg_left->data.data()), img_msg_left->step);

        cv::Mat image_right(img_msg_right->height, img_msg_right->width, CV_8UC1,
        const_cast<unsigned char*>(img_msg_right->data.data()), img_msg_right->step);
        
        tracker.detect_keypoints(&image_left, &image_right);
        tracker.triangulate_points();
        Eigen::Matrix4f transformation = tracker.icp();

        tracker.add_transformations(transformation);
        tracker.update_camera_trajectory();
        tracker.print_last_camera_position();

        // logging
        //auto current_frame =  tracker.frames.back();
        Eigen::Vector3f translation = transformation.block<3, 1>(0, 3);

        // Extract rotation matrix
        Eigen::Matrix3f rotation = transformation.block<3, 3>(0, 0);

        // Convert rotation matrix to unit quaternion
        Eigen::Quaternionf quaternion(rotation);
        geometry_msgs::msg::TransformStamped transform_msg;
        //transform_msg.header.stamp = msg.header.stamp;
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

        nav_msgs::msg::Odometry odom_msg;
        //odom_msg.header.stamp = msg.header.stamp;
        odom_msg.header.frame_id = odom_frame_;
        odom_msg.child_frame_id = child_frame_;
        odom_msg.pose.pose.orientation.x = quaternion.x();
        odom_msg.pose.pose.orientation.y = quaternion.y();
        odom_msg.pose.pose.orientation.z = quaternion.z();
        odom_msg.pose.pose.orientation.w = quaternion.w();
        odom_msg.pose.pose.position.x = translation.x();
        odom_msg.pose.pose.position.y = translation.y();
        odom_msg.pose.pose.position.z = translation.z();
        odom_publisher_->publish(odom_msg);
     
    }
    
public:
    Tracker tracker;
private:
    message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_1_;
    message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_2_;
    std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>> sync_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // what is tf broadcasters role?
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    std::string odom_frame_{"odom"};
    std::string child_frame_{"base_link"};
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
