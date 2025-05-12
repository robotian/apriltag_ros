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
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <zed_msgs/srv/set_pose.hpp>
#include <Eigen/Dense>

using Eigen::Vector3f;

using namespace std::chrono_literals;
using namespace std::placeholders;

#define TIMEZERO_ROS rclcpp::Time(0, 0, RCL_ROS_TIME)

#ifndef DEG2RAD
#define DEG2RAD 0.017453293
#define RAD2DEG 57.295777937
#endif

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

// Creates a parameter descriptor. Carried over from AprilTagNode
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

        // Transform Buffer and Listener
        std::shared_ptr<tf2_ros::Buffer> transformBuffer_{nullptr};
        std::shared_ptr<tf2_ros::TransformListener> transformListener_;
        

        // Service client to reset pose
        rclcpp::Client<zed_msgs::srv::SetPose>::SharedPtr setPoseClient_;

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

        // Map of tag frames to transforms
        std::unordered_map<std::string, tf2::Transform> tagTransformMap_;




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
         * @brief Get existing transformations from TF. This function is
         * pulled from the ZED ArUco tag localization example. 
         * 
         * @param targetFrame 
         * @param sourceFrame 
         * @param out_tr 
         */
        bool getTransformFromTf(
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
    tfBroadcaster_(this),
    staticBroadcaster_(this)
{

    // Construct the transform buffer and listener
    transformBuffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    transformListener_ = std::make_unique<tf2_ros::TransformListener>(*transformBuffer_);

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

    cameraName_ = declare_parameter("camera_name", "", descr("Camera name for TF lookup", true));
    worldFrame_ = declare_parameter("world_frame_id", "", descr("World frame ID ", true));


    // Grab the global poses for each tag
    for(int64_t id : tagIDs)
    {
        std::string currTagPoseParam = "global_pose_list.tag_";
        currTagPoseParam += std::to_string(id);
        std::vector<_Float64> currTagPose = declare_parameter(currTagPoseParam, std::vector<_Float64>{});
        globalTagPoseMap_[id] = currTagPose;        
    }                                       

    // Check we have a transform defined for each tag
    if(globalTagPoseMap_.size() != tagIDs.size())
    {
        throw std::runtime_error("Size Mismatch between the passed tag IDs and their global transforms!");
    }

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

        // Populate the map from tag ID to frames
        for(size_t i = 0; i < tagIDs.size(); i++)
        {
            tagIDtoFrame_[tagIDs[i]] = tagFrames[i];
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

    // Initialize transformations
    initTFs();

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

            // Calculate the length of the translation, and determine if its the closest tag
            Vector3f transVec(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
            double tagDist = transVec.norm();
            if(tagDist < distanceToClosestTag)
            {
                distanceToClosestTag = tagDist;
                tfToClosestTag = tf;
                closestTagID = currDetection->id;
            }

        }
        else
        {
            RCLCPP_WARN(get_logger(), "Detection: %d. Camera is not calibrated, will not be estimating pose!", i);
        }


    }

    if(closestTagID != -1)
    {
        //TODO
        RCLCPP_INFO(get_logger(), "Correcting pose using Tag ID: %d", closestTagID);
    }

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
    // Grab the camera coptical frame to camera base frame
    std::string camLeftFrame = cameraName_ + "_left_camera_frame";
    std::string camBaseFrame = cameraName_ + "_camera_center";
    bool tfOk = getTransformFromTf(camLeftFrame, camBaseFrame, camLeftToBase_);

    if(!tfOk)
    {
        RCLCPP_ERROR(get_logger(), "Could not grab transform '%s' -> '%s', Please verify the parameters and the status "
        "of the 'ZED State Publisher' node!", camBaseFrame.c_str(), camLeftFrame.c_str());
        exit(EXIT_FAILURE);
    }

    // double r, p, y;
    tf2::Matrix3x3 basis;

    // Set up AprilTag coordinate system to Image coordinate system, and vice versa (NOTE: we might need to change these)
    basis = tf2::Matrix3x3(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0); // flip x and y, leave z
    imageToTag_.setIdentity();
    imageToTag_.setBasis(basis);
    tagToImage_ = imageToTag_.inverse();

    // Set up ROS coordinate system to image coordinate system, and vice versa (NOTE: we might need to change these)
    basis = tf2::Matrix3x3(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0);
    rosToImage_.setIdentity();
    rosToImage_.setBasis(basis);
    imageToROS_ = rosToImage_.inverse();

    for(int64_t id : tagIDs_)
    {
        // Create and send the global transform for each tag
        // There's probably a way to do this in less lines of code
        geometry_msgs::msg::TransformStamped globalTagTransform;
        globalTagTransform.header.stamp = this->get_clock()->now();
        globalTagTransform.header.frame_id = worldFrame_;
        globalTagTransform.child_frame_id = tagIDtoFrame_[id];
        std::vector<_Float64> currTransform = globalTagPoseMap_[id];
        tf2::Quaternion globalRot; 
        globalRot.setRPY(currTransform[4], currTransform[5], currTransform[6]);
        globalTagTransform.transform.translation.x = currTransform[0];
        globalTagTransform.transform.translation.y = currTransform[1];
        globalTagTransform.transform.translation.z = currTransform[2];
        globalTagTransform.transform.rotation.x = globalRot.getX();
        globalTagTransform.transform.rotation.y = globalRot.getY();
        globalTagTransform.transform.rotation.z = globalRot.getZ();
        globalTagTransform.transform.rotation.w = globalRot.getW();

        RCLCPP_INFO(get_logger(), "Sending Static transform from '%s' to '%s': (%f %f %f), (%f, %f, %f, %f)", 
            worldFrame_.c_str(), tagIDtoFrame_[id].c_str(),
            globalTagTransform.transform.translation.x, globalTagTransform.transform.translation.y,
            globalTagTransform.transform.translation.z, globalTagTransform.transform.rotation.x, 
            globalTagTransform.transform.rotation.y, globalTagTransform.transform.rotation.z, 
            globalTagTransform.transform.rotation.w);

        staticBroadcaster_.sendTransform(globalTagTransform);
    }

    // Grab the tag transform as tf2::Transforms (also verified that we successfully broadcast the static transforms)
    for(int64_t id : tagIDs_)
    {
        tfOk = getTransformFromTf(tagIDtoFrame_[id], worldFrame_, tagTransformMap_[tagIDtoFrame_[id]]);
        if(!tfOk)
        {
            RCLCPP_ERROR(get_logger(), "Could not grab transform '%s' -> '%s', Please verify the AprilTag pose " 
            "parameters!", worldFrame_.c_str(), tagIDtoFrame_[id].c_str());
            exit(EXIT_FAILURE);
        }
    }

    RCLCPP_ERROR(get_logger(), "Number of tag transforms grabbed from tf: %ld", tagTransformMap_.size());


}

// Modified slightly from the ZED ArUco localization example code
bool PoseCorrectionNode::getTransformFromTf(
    std::string targetFrame, std::string sourceFrame,
    tf2::Transform & out_tr)
{

  std::string msg;
  geometry_msgs::msg::TransformStamped transf_msg;

  try 
  {
    transformBuffer_->canTransform(
      targetFrame, sourceFrame, TIMEZERO_ROS, 1000ms,
      &msg);
    RCLCPP_INFO_STREAM(
      get_logger(), "[getTransformFromTf] canTransform '"
        << targetFrame.c_str() << "' -> '"
        << sourceFrame.c_str()
        << "':" << msg.c_str());
    std::this_thread::sleep_for(3ms);

    transf_msg =
      transformBuffer_->lookupTransform(targetFrame, sourceFrame, TIMEZERO_ROS, 1s);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(
      this->get_logger(),
      "[getTransformFromTf] Could not transform '%s' to '%s': %s",
      targetFrame.c_str(), sourceFrame.c_str(), ex.what());
    return false;
  }

  tf2::Stamped<tf2::Transform> tr_stamped;
  tf2::fromMsg(transf_msg, tr_stamped);
  out_tr = tr_stamped;
  double r, p, y;
  out_tr.getBasis().getRPY(r, p, y, 1);

  RCLCPP_INFO(
    get_logger(),
    "[getTransformFromTf] '%s' -> '%s': \n\t[%.3f,%.3f,%.3f] - "
    "[%.3f°,%.3f°,%.3f°]",
    sourceFrame.c_str(), targetFrame.c_str(), out_tr.getOrigin().x(),
    out_tr.getOrigin().y(), out_tr.getOrigin().z(), r * RAD2DEG,
    p * RAD2DEG, y * RAD2DEG);

  return true;
}


bool PoseCorrectionNode::resetZedPose(tf2::Transform & /*new_pose*/)
{
    return true;
}