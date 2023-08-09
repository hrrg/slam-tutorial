#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <regex>


class Map
{
public:
    Map(){
        map =  pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    };
    ~Map(){};
    void update_localmap(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::Matrix4f new_pose);
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr map;
};

void Map::update_localmap(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::Matrix4f new_pose){
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_transformed(new pcl::PointCloud<pcl::PointXYZ>());
    points_transformed->reserve(cloud->points.size());

    for (const auto &point : cloud->points) {
        Eigen::Vector4f point_homogeneous(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f transformed_point = new_pose * point_homogeneous;
        pcl::PointXYZ transformed_point_3d(transformed_point.x(), transformed_point.y(), transformed_point.z());
        points_transformed->push_back(transformed_point_3d);
    }

    // Assuming map is a pcl::PointCloud<pcl::PointXYZ>::Ptr
    *map += *points_transformed;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string FixFrameId(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}


inline std::unique_ptr<sensor_msgs::msg::PointCloud2> CreatePointCloud2Msg(const size_t n_points,
                                                         const std_msgs::msg::Header &header,
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

inline void FillPointCloud2XYZ(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                               sensor_msgs::msg::PointCloud2 &msg) {
    sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
    for (size_t i = 0; i < cloud->points.size(); i++, ++msg_x, ++msg_y, ++msg_z) {
        const auto p = cloud->points[i];
        *msg_x = p.x;
        *msg_y = p.y;
        *msg_z = p.z;
    }
}





inline std::unique_ptr<sensor_msgs::msg::PointCloud2>
EigenToPointCloud2(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                   const std_msgs::msg::Header &header) {
    auto msg = CreatePointCloud2Msg(cloud->points.size(), header);
    FillPointCloud2XYZ(cloud, *msg);
    return msg;
}


 cv::Mat intrinsic_left = (cv::Mat_<double>(3,3) << 458.654, 0, 367.215,
                                                          0, 457.296, 248.375,
                                                          0, 0, 1); 
 cv::Mat intrinsic_right = (cv::Mat_<double>(3,3) << 457.587, 0, 379.999,
                                                    0, 456.134, 255.238,
                                                    0, 0, 1);
 cv::Mat calib_left = (cv::Mat_<double>(3,4) << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
          0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
         -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949);
 cv::Mat calib_right = (cv::Mat_<double>(3,4) << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
          0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
         -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038);
        
// cv::Mat intrinsic_left = (cv::Mat_<double>(3,3) << 9.842439e+02, 0.000000e+00, 6.900000e+02,
//                                                  0.000000e+00, 9.808141e+02, 2.331966e+02,
//                                                   0.000000e+00, 0.000000e+00, 1.000000e+00); 

// cv::Mat intrinsic_right = (cv::Mat_<double>(3,3) << 9.895267e+02, 0.000000e+00, 7.020000e+02,
//                                                     0.000000e+00, 9.878386e+02, 2.455590e+02,
//                                                      0.000000e+00, 0.000000e+00, 1.000000e+00);
// cv::Mat calib_left = (cv::Mat_<double>(3,4) << 1.000000e+00, 0.000000e+00, 0.000000e+00, 2.573699e-16,
//                                                  0.000000e+00, 1.000000e+00, 0.000000e+00, -1.059758e-16, 
//                                                   0.000000e+00, 0.000000e+00, 1.000000e+00,  1.614870e-16
// );
// cv::Mat calib_right = (cv::Mat_<double>(3,4) << 9.993513e-01, 1.860866e-02, -3.083487e-02,-5.370000e-01,
//                                              -1.887662e-02, 9.997863e-01, -8.421873e-03,4.822061e-03,
//                                                  3.067156e-02, 8.998467e-03, 9.994890e-01,-1.252488e-02
//);
cv::Mat projection_left = intrinsic_left*calib_left;
cv::Mat projection_right = intrinsic_right*calib_right;

class Frame
{
public:
    Frame(){

        transformation = new Eigen::Matrix4f();
        transformation->setIdentity();
        world_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    };
    ~Frame(){};

  public:
    Eigen::Matrix<float, 4, 4> *transformation; // Rotation and translation of Frame
    std::vector<cv::KeyPoint> keypoints_left;
    std::vector<cv::KeyPoint> keypoints_right;
    std::vector<cv::Point2d> matched_keypoints_left;
    std::vector<cv::Point2d> matched_keypoints_right;
    pcl::PointCloud<pcl::PointXYZ>::Ptr world_points;
};

class Tracker {
  public:
    Tracker() { feature_detector = cv::ORB::create(); }
    ~Tracker(){};
    void equalize_histogram(cv::Mat &image_left, cv::Mat &image_right, int method = 0);
    void detect_keypoints(cv::Mat *image_left, cv::Mat *image_right);
    void triangulate_points();
    Eigen::Matrix4f icp();

  public:
    cv::Ptr<cv::Feature2D> feature_detector;
    std::vector<Frame> frames;

};

void Tracker::equalize_histogram(cv::Mat &image_left, cv::Mat &image_right, int method = 0) {
    if (method == 0) {
        cv::equalizeHist(image1, image1);
        cv::equalizeHist(image2, image2);
    } else if (method == 1) {
        cv::Ptr<cv::CLAHE> clahe_left = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Ptr<cv::CLAHE> clahe_right = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe_left->apply(image1, image1);
        clahe_right->apply(image2, image2);
    }
}

void Tracker::detect_keypoints(cv::Mat *image_left, cv::Mat *image_right) {
    Frame frame;
    cv::Mat descriptors_left;
    cv::Mat descriptors_right;

    // Feature detection
    feature_detector->detectAndCompute(*image_left, cv::noArray(), frame.keypoints_left,
                                       descriptors_left);
    feature_detector->detectAndCompute(*image_right, cv::noArray(), frame.keypoints_right,
                                       descriptors_right);

    // Match descriptors
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_left, descriptors_right, matches);

    // Filter matches using ratio test
    std::vector<cv::DMatch> filtered_matches;
    // double ratio_threshold = 0.1;
    for (const auto &match : matches) {
        if (match.distance < 50) {
            filtered_matches.push_back(match);
        }
    }

    // Retrieve 3D points for matched keypoints
    for (const cv::DMatch &match : filtered_matches) {
        cv::Point2f pt1 = frame.keypoints_left[match.queryIdx].pt;
        cv::Point2f pt2 = frame
                              .

                          keypoints_right[match.trainIdx]
                              .pt;
        frame.matched_keypoints_left.push_back(pt1);
        frame.matched_keypoints_right.push_back(pt2);
    }
    frames.push_back(frame);
}

void Tracker::triangulate_points() {
    cv::Mat world_points;
    auto frame = frames.back();
    cv::triangulatePoints(projection_left, projection_right, frame.matched_keypoints_left,
                          frame.matched_keypoints_right, world_points);

    for (int world_point_idx = 0; world_point_idx < world_points.cols; world_point_idx++) {
        auto point = world_points.col(world_point_idx);
        pcl::PointXYZ p(point.at<double>(0), point.at<double>(1), point.at<double>(2));
        frame.world_points->push_back(p);
    }
}

Eigen::Matrix4f Tracker::icp() {
    // run icp
    if (frames.size() < 2)
        return Eigen::Matrix4f::Identity();
    auto current_frame = &frames.back();
    auto previous_frame = &frames.at(frames.size() - 2);

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setMaxCorrespondenceDistance(1.0);
    icp.setTransformationEpsilon(0.001);
    icp.setMaximumIterations(100);

    pcl::PointCloud<pcl::PointXYZ>::Ptr align(new pcl::PointCloud<pcl::PointXYZ>);

    icp.setInputSource(current_frame->world_points);
    icp.setInputTarget(previous_frame->world_points);
    icp.align(*align);

    *current_frame->transformation =
        icp.getFinalTransformation() * (*previous_frame->transformation);
    return *current_frame->transformation;
}



class ImageSubscriberNode : public rclcpp::Node
{

public:
    ImageSubscriberNode()
      : Node("exact_time_subscriber")
    {
        //subscription_temp_1_.subscribe(this, "/kitti/camera_gray_left/image_raw");
        //subscription_temp_2_.subscribe(this, "/kitti/camera_gray_right/image_raw");  
        subscription_temp_1_.subscribe(this, "/cam0/image_raw");
        subscription_temp_2_.subscribe(this, "/cam1/image_raw");  
        sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(subscription_temp_1_, subscription_temp_2_, 3);
        sync_->registerCallback(std::bind(&ImageSubscriberNode::topic_callback, this, std::placeholders::_1, std::placeholders::_2));
        
        rclcpp::QoS qos(rclcpp::KeepLast{100});
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("odometry", qos);
        frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("frame", qos);
        map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("local_map", qos);
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg_left,
                        const sensor_msgs::msg::Image::ConstSharedPtr &img_msg_right) {
        cv::Mat image_left(img_msg_left->height, img_msg_left->width, CV_8UC1,
                           const_cast<unsigned char *>(img_msg_left->data.data()),
                           img_msg_left->step);

        cv::Mat image_right(img_msg_right->height, img_msg_right->width, CV_8UC1,

        const_cast<unsigned char*>(img_msg_right->data.data()), img_msg_right->step);
        cv::Mat image_small_left;
        cv::Mat image_small_right;
        cv::resize(image_left, image_small_left, cv::Size(img_msg_left->width/4, img_msg_left->height/4));
        cv::resize(image_right, image_small_right, cv::Size(img_msg_left->width/4, img_msg_left->height/4));
        tracker.detect_keypoints(&image_small_left, &image_small_right);

        tracker.triangulate_points();
        
        tracker.icp();
        auto current_frame = &tracker.frames.back();
        auto world_points = current_frame->world_points;
        Eigen::Matrix4f transfomation = *current_frame->transformation;
        std::stringstream ss;
        RCLCPP_INFO(this->get_logger(), "callback");
        // logging
        // auto current_frame =  tracker.frames.back();
        // 이전 frame의 p_2=T*p_1

        Eigen::Vector3f translation = transfomation.block<3, 1>(0, 3);

        // Extract rotation matrix
        Eigen::Matrix3f rotation = transfomation.block<3, 3>(0, 0);
        map.update_localmap(world_points, transfomation);

        // Convert rotation matrix to unit quaternion
        Eigen::Quaternionf quaternion(rotation);
        geometry_msgs::msg::TransformStamped transform_msg;

        transform_msg.header.stamp = this->now();
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
        odom_msg.header.stamp = this->now();
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


        sensor_msgs::msg::PointCloud2 frame_msg;
        frame_msg.header.stamp = this->now();
        frame_msg.header.frame_id = child_frame_;        
        frame_publisher_->publish(std::move(EigenToPointCloud2(world_points, frame_msg.header)));

        sensor_msgs::msg::PointCloud2 map_msg;
        map_msg.header.stamp = this->now();
        map_msg.header.frame_id = odom_frame_;
        //auto local_map_header = msg->header;
        //local_map_header.frame_id = odom_frame_;
        map_publisher_->publish(std::move(EigenToPointCloud2(map.map, map_msg.header)));
    }


    
public:
    Tracker tracker;
    Map map;
private:

    message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_1_;
    message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_2_;
    std::shared_ptr<
        message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>
        sync_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // what is tf broadcasters role?
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
    std::string odom_frame_{"world"};
    std::string child_frame_{"base_link"};
    std::string map_frame_{"frame"};
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
