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
        std::vector<long int> tagIDs_;

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
         * @param camInfo Camera info from the ZED X camera
         */
        void onCamera(
            const sensor_msgs::msg::Image::ConstSharedPtr & img,
            const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camInfo);

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
    // read-only parameters, grab the tag family and edge size used
    const std::string tagFamilyStr = declare_parameter("family", "36h11", descr("tag family", true));
    tagEdgeSize_ = declare_parameter("size", 1.0, descr("default tag size", true));

    // Grab the tag ids, frame names, and sizes
    const auto tagIDs = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto tagFrames = declare_parameter("tag.frames", std::vector<std::string>{}, 
                                                descr("tag frame names per id", true));
    const auto tagSizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));

    // Grab the pose estimation method
    const std::string& poseEstimationMethod = declare_parameter("pose_estimation_method", "pnp", 
                                                                descr("pose estimation method: \"pnp\" (more accurate)"
                                                                " or \"homography\" (faster)"), true);

    
    // Check and set the pose estimation method
    if(!poseEstimationMethod.empty())
    {
        if(pose_estimation_methods.count(poseEstimationMethod))
        {
            estimatePose_ = pose_estimation_methods.at(poseEstimationMethod);
        }
        else
        {
            throw std::runtime_error("Invalid pose estimation method: " + poseEstimationMethod); 
        }
    }
    else
    {
        throw std::runtime_error("Pose estimation cannot be empty!");
    }

    // Grab all of the detector parameters
    declare_parameter("detector.threads", tagDetector_->nthreads, descr("number of threads"));
    declare_parameter("detector.decimate", tagDetector_->quad_decimate, descr("decimate resolution for quad detection"));
    declare_parameter("detector.blur", tagDetector_->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare_parameter("detector.refine", tagDetector_->refine_edges, descr("snap to strong gradients"));
    declare_parameter("detector.sharpening", tagDetector_->decode_sharpening, descr("sharpening of decoded images"));
    declare_parameter("detector.debug", tagDetector_->debug, descr("write additional debugging images to working directory"));

    // Grab the hamming and profling parameters
    declare_parameter("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare_parameter("profile", false, descr("print profiling information to stdout"));


    // Error check tag frames & ids
    if(!tagFrames.empty())
    {
        if(tagIDs.size() != tagFrames.size())
        {
            throw std::runtime_error("Number of tag frames: " + std::to_string(tagFrames.size()) +
            " and number of tag ids: " + std::to_string(tagIDs.size()) + " differ!");
        }
    }

    // Error check tag sizes & ids
    if(!tagSizes.empty())
    {
        // Verify that each id has its own size registered
        if(tagIDs.size() != tagSizes.size())
        {
            throw std::runtime_error("Number of tag sizes: " + std::to_string(tagSizes.size()) +
             " and number of tag ids: " + std::to_string(tagIDs.size()) + " differ!");
        }

        tagIDs_ = tagIDs;
        for(size_t i = 0; i < tagIDs.size(); i++)
        {
            tagSizes_[tagIDs[i]] = tagSizes[i];
        }
    }

    // Attempt to add tag family to the detector
    if(tag_fun.count(tagFamilyStr))
    {
        tagFamily_ = tag_fun.at(tagFamilyStr).first();
        tagFamilyDestructor_ = tag_fun.at(tagFamilyStr).second;
        apriltag_detector_add_family(tagDetector_, tagFamily_);
    } 
    else
    {
        throw std::runtime_error("Unsupported tag family: " + tagFamilyStr);
    }
}

PoseCorrectionNode::~PoseCorrectionNode()
{
    apriltag_detector_destroy(tagDetector_);
    tagFamilyDestructor_(tagFamily_);
}

void PoseCorrectionNode::onCamera(
    const sensor_msgs::msg::Image::ConstSharedPtr & img,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camInfo
)
{
    // camera intrinsics for rectified images
    const std::array<double, 4> intrinsics = {camInfo->p[0], camInfo->p[5], camInfo->p[2], camInfo->p[6]};

    // check for valid intrinsics
    const bool calibrated = camInfo->width && camInfo->height &&
                            intrinsics[0] && intrinsics[1] && intrinsics[2] && intrinsics[3];
 
    // Make sure the camera is calibrated
    if(!calibrated) 
    {
        RCLCPP_WARN_STREAM(get_logger(), "The camera is not calibrated!");
    }

    // Convert to an 8bit monochrome image using OpenCV
    const cv::Mat imgUint8 = cv_bridge::toCvShare(img, "mono8")->image;

    // create a struct of the image
    image_u8_t im{imgUint8.cols, imgUint8.rows, imgUint8.cols, imgUint8.data};

    // Detect apriltags (must be mutually exclusive)
    mutex_.lock();
    zarray_t* detections = apriltag_detector_detect(tagDetector_, &im);
    mutex_.unlock();

    // Set up detections message
    apriltag_msgs::msg::AprilTagDetectionArray detectionsMsg;
    detectionsMsg.header = img->header;

    // Display the timeprofile if set (not quite sure what this is)
    if(profile_)
        timeprofile_display(tagDetector_->tp);


    // Set up vector of transformations to publish
    std::vector<geometry_msgs::msg::TransformStamped> transformsToTags;

    geometry_msgs::msg::TransformStamped tfToClosestTag;
    double distanceToClosestTag = __DBL_MAX__;
    int closestTagID = -1;
        
    for(int i = 0; i < zarray_size(detections); i++)
    {
        // Grab the current detection
        apriltag_detection_t* currDetection;
        zarray_get(detections, i, &currDetection);

        //DEBUG:
        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, currDetection->family->nbits, currDetection->family->h, currDetection->id,
                     currDetection->hamming, currDetection->decision_margin);

        // Ignore tags we aren't trying to detect
        if(!tagFrames_.empty() && !tagFrames_.count(currDetection->id))
            continue;

        // Reject detections that require too many corrected bits (from parameter)
        if(currDetection->hamming > maxHamming_)
            continue;

        // Grab the detection
        apriltag_msgs::msg::AprilTagDetection tagDetection;
        tagDetection.family = std::string(currDetection->family->name);
        tagDetection.id = currDetection->id;
        tagDetection.hamming = currDetection->hamming;
        tagDetection.decision_margin = currDetection->decision_margin;
        tagDetection.centre.x = currDetection->c[0];
        tagDetection.centre.y = currDetection->c[1];
        std::memcpy(tagDetection.corners.data(), currDetection->p, sizeof(double) * 8);
        std::memcpy(tagDetection.homography.data(), currDetection->H->data, sizeof(double) * 9);
        detectionsMsg.detections.push_back(tagDetection);

        // If calibrated, then estimate the pose of the tag
        if(calibrated)
        {
            geometry_msgs::msg::TransformStamped tf;
            tf.header = img->header;
            int currTagID = currDetection->id;
            char* currTagName = currDetection->family->name;
            std::string familyAndID = std::string(currTagName) + ":" + std::to_string(currTagID);
            tf.child_frame_id = tagFrames_.count(currTagID) ? tagFrames_.at(currTagID) : familyAndID;
            const double size = tagSizes_.count(currTagID) ? tagSizes_.at(currTagID) : tagEdgeSize_;
            tf.transform = estimatePose_(currDetection, intrinsics, size);
            transformsToTags.push_back(tf);

            // TODO calculate the length of the translation
            // If the distance is less than the current closest distance, update it, the id, and the transform

        }
        else
        {
            RCLCPP_WARN(get_logger(), "Detection: %d. Camera is not calibrated, will not be estimating pose!", i);
        }


    }

    // TODO grab the closest transform, put it in terms of the map frame using the known tag position, and 
    // call the set_pose service of the ZED camera

    // Publish detections
    detectionPub_->publish(detectionsMsg);

    // Broadcast transforms
    tfBroadcaster_.sendTransform(transformsToTags);

    // Deallocate detections
    apriltag_detections_destroy(detections);

}

rcl_interfaces::msg::SetParametersResult
PoseCorrectionNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex_.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

        IF("detector.threads", tagDetector_->nthreads)
        IF("detector.decimate", tagDetector_->quad_decimate)
        IF("detector.blur", tagDetector_->quad_sigma)
        IF("detector.refine", tagDetector_->refine_edges)
        IF("detector.sharpening", tagDetector_->decode_sharpening)
        IF("detector.debug", tagDetector_->debug)
        IF("max_hamming", maxHamming_)
        IF("profile", profile_)
    }

    mutex_.unlock();

    result.successful = true;

    return result;
}



void PoseCorrectionNode::initTFs()
{

}

void PoseCorrectionNode::broadcastMarkerTFs()
{

}


void PoseCorrectionNode::getTransformFromTf(
    std::string /*targetFrame*/, std::string /*sourceFrame*/,
    tf2::Transform & /*out_tr*/)
{

}


bool PoseCorrectionNode::resetZedPose(tf2::Transform & /*new_pose*/)
{
    return true;
}