/*
 * Created on Thu May 01 2025
 * Author: Austen Goddu, ajgoddu@mtu.edu
 * Intelligent Robotics and System Optimization Laboratory
 * Michigan Technological University
 *
 * Purpose: Runs composed with the ZED ROS2 Wrapper in order
 * to detect AprilTags and use them to correct the pose.
 * 
 * Note: I'm following the convention already used by the repository, but man
 * they should separate their declarations into a header file
 */


#include "pose_estimation.hpp"
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/camera_subscriber.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <zed_msgs/srv/set_pose.hpp>


// apriltag
#include "tag_functions.hpp"
#include <apriltag.h>

// Class to encompass the apriltag detection and pose correction
class PoseCorrectionNode : public rclcpp::Node
{
    public:
   
        PoseCorrectionNode(const rclcpp::NodeOptions& options);

        ~PoseCorrectionNode();

    private:

        const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

        apriltag_family_t* tf;
        apriltag_detector_t* const td;

        // parameter
        std::mutex mutex;
        double tag_edge_size;
        std::atomic<int> max_hamming;
        std::atomic<bool> profile;
        std::unordered_map<int, std::string> tag_frames;
        std::unordered_map<int, double> tag_sizes;

        // TF destructor
        std::function<void(apriltag_family_t*)> tf_destructor;

        // Camera subscription, AprilTag Detections, and broadcaster for transformations
        const image_transport::CameraSubscriber sub_cam;
        const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
        tf2_ros::TransformBroadcaster tf_broadcaster;

        // Service client to reset pose
        rclcpp::Client<zed_msgs::srv::SetPose>::SharedPtr _setPoseClient;

        // Function used to estimate pose
        pose_estimation_f estimate_pose = nullptr;


        /**
         * @brief Callback to detect Apriltags and call the pose correction
         * Service
         * 
         * @param img Image from the ZED X camera
         * @param cam_info Camera info from the ZED X camera
         */
        void camera_callback(
            const sensor_msgs::msg::Image::ConstSharedPtr & img,
            const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info);

        /**
         * @brief Initializes transformations
         * 
         */
        void initTFs();

        /**
         * @brief Broadcasts markers for each TF
         * 
         */
        void broadcastMarkerTFs();

        /**
         * @brief Get existing transformations from TF
         * 
         * @param targetFrame 
         * @param sourceFrame 
         * @param out_tr 
         */
        void getTransformFromTf(
            std::string targetFrame, std::string sourceFrame,
            tf2::Transform & out_tr);
        
        /**
         * @brief Calls the set_pose service from the ZED ROS2 Wrapper
         * to correct the pose.
         * 
         * @param new_pose 
         * @return true 
         * @return false 
         */
        bool resetZedPose(tf2::Transform & new_pose);
    
        /**
         * @brief Callback that is triggered when parameters are changed. Pulled from AprilTagNode
         * 
         * @param parameters 
         * @return rcl_interfaces::msg::SetParametersResult 
         */
        rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);

};
RCLCPP_COMPONENTS_REGISTER_NODE(PoseCorrectionNode)



PoseCorrectionNode::PoseCorrectionNode(const rclcpp::NodeOptions& options) : Node("apriltag_pose_correction", options)
{

}

PoseCorrectionNode::~PoseCorrectionNode(){}


rcl_interfaces::msg::SetParametersResult
PoseCorrectionNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

        IF("detector.threads", td->nthreads)
        IF("detector.decimate", td->quad_decimate)
        IF("detector.blur", td->quad_sigma)
        IF("detector.refine", td->refine_edges)
        IF("detector.sharpening", td->decode_sharpening)
        IF("detector.debug", td->debug)
        IF("max_hamming", max_hamming)
        IF("profile", profile)
    }

    mutex.unlock();

    result.successful = true;

    return result;
}