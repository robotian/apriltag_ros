/*
 * Created on Thu May 01 2025
 * Author: Austen Goddu, ajgoddu@mtu.edu
 * Intelligent Robotics and System Optimization Laboratory
 * Michigan Technological University
 *
 * Purpose: Runs composed with the ZED ROS2 Wrapper in order
 * to detect AprilTags and use them to correct the pose of the global
 * EKF. 
 *
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
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
// #include <zed_msgs/srv/set_pose.hpp>
// #include <zed_interfaces/srv/set_pose.hpp>
#include <Eigen/Dense>

// Struct with information about a pose used to calculate the weights in the weighted average
struct detectionPose
{
     int id;
     apriltag_detection_t* detection;
     tf2::Transform localTransform; // Transform tag -> robot
     tf2::Transform globalTransform; // Transform map -> robot
};

// LINK - Class Definition
//Class to encompass the apriltag detection and pose correction
class PoseCorrectionNode : public rclcpp::Node {
public:
    PoseCorrectionNode(const rclcpp::NodeOptions& options);

    ~PoseCorrectionNode();

private:
    // Callback handle to allow updating parameters during runtime
    const OnSetParametersCallbackHandle::SharedPtr paramCallbackHandler_;

    // Apriltag family and detector pointers
    std::string tagFamilyStr_;
    apriltag_family_t* tagFamily_;
    apriltag_detector_t* const tagDetector_;

    // parameter
    std::mutex mutex_;
    double tagEdgeSize_;
    std::atomic<int> maxHamming_;
    std::atomic<bool> profile_;
    std::unordered_map<int, std::string> tagFrames_;
    std::unordered_map<int, double> tagSizes_;
    std::vector<long int> tagIDs_;
    std::vector<long int> dockIDs_;
    double maxTagDist_;

    // Maps of tag IDs to pose and frame name
    std::unordered_map<int64_t, std::vector<_Float64>> globalTagPoseMap_;
    std::unordered_map<int64_t, std::string> tagIDtoFrame_;

    // TF destructor
    std::function<void(apriltag_family_t*)> tagFamilyDestructor_;

    // Camera subscription, AprilTag Detections, and broadcasters for transformations
    const image_transport::CameraSubscriber cameraSub_;
    const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr detectionPub_;
    tf2_ros::TransformBroadcaster tfBroadcaster_;
    tf2_ros::StaticTransformBroadcaster staticBroadcaster_;

    // Pose Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr posePub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr dockingPub_;

    // Transform Buffer and Listener
    std::shared_ptr<tf2_ros::Buffer> transformBuffer_{nullptr};
    std::shared_ptr<tf2_ros::TransformListener> transformListener_;

    // Function used to estimate pose
    pose_estimation_f estimatePose_ = nullptr;

    std::string cameraName_;
    std::string worldFrame_;

    // Transforms to move between reference frames
    tf2::Transform camLeftToBase_;
    tf2::Transform imageToROS_;
    tf2::Transform rosToImage_;
    tf2::Transform tagToImage_;
    tf2::Transform imageToTag_;
    tf2::Transform camCenterToBaseLink_;

    // Map of tag frames to transforms
    std::unordered_map<std::string, tf2::Transform> tagTransformMap_;
    std::unordered_map<int, apriltag_detection_t*> detectionMap_;

    // Counter/Parameter to throttle how often the pose correction and global transform publish runs (every nth frame)
    int detectionThrottle_ = 0;
    int correct_every_n_frames_ = 10;

    // Offset & Rotation for the docking position
    std::vector<double> dockOffset_ = {0.0, 0.0, 0.0}; // XYZ
    std::vector<double> dockRotation_ = {0.0, 0.0, 0.0}; //RPY


    /**
         * @brief Callback to detect Apriltags and call the pose correction
         * Service
         *
         * @param img Image from the ZED X camera
         * @param camInfo Camera info from the ZED X camera
         */
    void onCamera(
        const sensor_msgs::msg::Image::ConstSharedPtr& img,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camInfo);

    /**
         * @brief Initializes transformations
         *
         */
    void initTFs();

    /**
         * @brief Get existing transformations from TF. This function is
         * pulled from the ZED ArUco tag localization example.
         *
         * @param targetFrame
         * @param sourceFrame
         * @param out_tr
         */
    bool getTransformFromTf(
        std::string targetFrame, std::string sourceFrame,
        tf2::Transform& out_tr);

    /**
         * @brief Takes the transform msg provided by estiamte_pose
         *
         * @param tf - Transform msg for the closest tag
         * @param id - ID of the closest tag
         * @return tf2::Transform - the transform from map to the robot using the tag
         */
    tf2::Transform computeTransform(tf2::Transform& tf, int id);

    /**
         * @brief Callback that is triggered when parameters are changed. Pulled from AprilTagNode
         *
         * @param parameters
         * @return rcl_interfaces::msg::SetParametersResult
         */
    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);

    /**
     * @brief Takes a vector of structs of global transforms for observed tags and returns a single transform as the 
     * product of weighted average.
     * @param globalTransformVec - A vector of detectionPose structs created after computeTransform calculates the global transform 
     * of the robot for each tag.
     * @return tf2::Transform - a transform created using a weighted average of each of the tags
     */
    tf2::Transform averageTransforms(std::vector<detectionPose>& globalTransformVec);

    /**
     * @brief Function to return the weighted average of quaternions using the Markley Method
     * 
     * @param quats - Vector of quaternion objects
     * @param weights - Vector of weights
     * 
     * @return Eigen::Quaterniond - The weighted average quaternion from the Markley method
     */
    Eigen::Quaterniond weightedAverageQuaternion(std::vector<Eigen::Quaterniond>& quats, std::vector<double>& weights);
};
RCLCPP_COMPONENTS_REGISTER_NODE(PoseCorrectionNode)