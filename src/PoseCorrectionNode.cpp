/*
 * Created on Thu May 01 2025
 * Author: Austen Goddu, ajgoddu@mtu.edu
 * Intelligent Robotics and System Optimization Laboratory
 * Michigan Technological University
 *
 * Purpose: Runs composed with the ZED ROS2 Wrapper in order
 * to detect AprilTags and use them to correct the pose.
 * 
 * Note: I'm following the convention already used by the repository, but man,
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


// Macro/Helpers for checkign and assigning parameters. Carried over from AprilTagNode
#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}
    
template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}
    
template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}

// Creates a parameter descriptor. Carriud over from AprilTagNode
rcl_interfaces::msg::ParameterDescriptor
descr(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor descr;

    descr.description = description;
    descr.read_only = read_only;

    return descr;
}

// Class to encompass the apriltag detection and pose correction
class PoseCorrectionNode : public rclcpp::Node
{
    public:
   
        PoseCorrectionNode(const rclcpp::NodeOptions& options);

        ~PoseCorrectionNode();

    private:

        // Callback handle to allow updating parameters during runtime
        const OnSetParametersCallbackHandle::SharedPtr paramCallbackHandler_;

        // Apriltag family and detector pointers
        
        const std::string tagFamilyStr_;
        apriltag_family_t* tagFamily_;

        apriltag_detector_t* const tagDetector_;

        // parameter
        std::mutex mutex_;
        double tagEdgeSize_;
        std::atomic<int> maxHamming_;
        std::atomic<bool> profile_;
        std::unordered_map<int, std::string> tagFrames_;
        std::unordered_map<int, double> tagSizes_;

        // TF destructor
        std::function<void(apriltag_family_t*)> tagFamilyDestructor_;

        // Camera subscription, AprilTag Detections, and broadcaster for transformations
        const image_transport::CameraSubscriber cameraSub_;
        const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr detectionPub_;
        tf2_ros::TransformBroadcaster tfBroadcaster_;

        // Service client to reset pose
        rclcpp::Client<zed_msgs::srv::SetPose>::SharedPtr setPoseClient_;

        // Function used to estimate pose
        pose_estimation_f estimatePose_ = nullptr;


        /**
         * @brief Callback to detect Apriltags and call the pose correction
         * Service
         * 
         * @param img Image from the ZED X camera
         * @param cam_info Camera info from the ZED X camera
         */
        void onCamera(
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



PoseCorrectionNode::PoseCorrectionNode(const rclcpp::NodeOptions& options) : Node("apriltag_pose_correction", options),
    paramCallbackHandler_(add_on_set_parameters_callback(
                            std::bind(&PoseCorrectionNode::onParameter, 
                            this, 
                            std::placeholders::_1))),
    tagDetector_(apriltag_detector_create()),
    cameraSub_(image_transport::create_camera_subscription(
                this, 
                this->get_node_topics_interface()->resolve_topic_name("image_rect"),
                std::bind(&PoseCorrectionNode::onCamera, this, std::placeholders::_1, std::placeholders::_2),
                declare_parameter("image_transport", "raw", descr({}, true)),
                rmw_qos_profile_sensor_data
    )),
    detectionPub_(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("apriltag_detections", rclcpp::QoS(1))),
    tfBroadcaster_(this)
{
    // read-only parameters
    const std::string tagFamily = declare_parameter("family", "36h11", descr("tag family", true));
    tagEdgeSize_ = declare_parameter("size", 1.0, descr("default tag size", true));


}

PoseCorrectionNode::~PoseCorrectionNode()
{
    apriltag_detector_destroy(tagDetector_);
    tagFamilyDestructor_(tagFamily_);
}

