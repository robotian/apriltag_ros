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

#include "PoseCorrectionNode.hpp"
#include <cmath>

#define INFO(...) RCLCPP_INFO(get_logger(), __VA_ARGS__)
#define WARN(...) RCLCPP_WARN(get_logger(), __VA_ARGS__)
#define FATAL(...) RCLCPP_FATAL(get_logger(), __VA_ARGS__)

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


// LINK - Constructor (I find this format a lot more readable, but it can be changed otherwise
PoseCorrectionNode::PoseCorrectionNode(const rclcpp::NodeOptions& options) : Node("apriltag_pose_correction", options),
    paramCallbackHandler_
    (
        add_on_set_parameters_callback
        (
            std::bind(&PoseCorrectionNode::onParameter,this, std::placeholders::_1)
        )
    ),
    tagDetector_(apriltag_detector_create()),
    cameraSub_
    (
        image_transport::create_camera_subscription
        (
            this,
            this->get_node_topics_interface()->resolve_topic_name("image_rect"),
            std::bind(&PoseCorrectionNode::onCamera, this, std::placeholders::_1, std::placeholders::_2),
            declare_parameter("image_transport", "raw", descr({}, true)),
            rmw_qos_profile_sensor_data
        )
    ),
    detectionPub_(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1))),
    tfBroadcaster_(this),
    staticBroadcaster_(this)
{

    // Construct the transform buffer and listener
    transformBuffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    transformListener_ = std::make_unique<tf2_ros::TransformListener>(*transformBuffer_);

    // grab the tag family and edge size used
    tagFamilyStr_ = declare_parameter("family", "36h11", descr("tag family", true));
    tagEdgeSize_ = declare_parameter("size", 1.0, descr("default tag size", true));

    // Grab the tag ids, frame names, and sizes
    const auto tagIDs = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto tagFrames = declare_parameter("tag.frames", std::vector<std::string>{},
                                             descr("tag frame names per id", true));
    const auto tagSizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));
    const auto dockIDs = declare_parameter("tag.dock_ids", std::vector<int64_t>{}, descr("tag ids corresponding to docks, must be in tag.ids", true));

    maxTagDist_ = declare_parameter("tag_transform_threshold_dist: 2.0", 2.0, descr("Max Distance at which tag transforms are considered", true));

    // Grab the pose estimation method
    const std::string& poseEstimationMethod = declare_parameter("pose_estimation_method", "pnp",
                                                                descr("pose estimation method: \"pnp\" (more accurate)"
                                                                      " or \"homography\" (faster)"),
                                                                true);
    
    // parameters to set the camera name, world frame, and how often to run pose correction
    cameraName_ = declare_parameter("camera_name", "", descr("Camera name for TF lookup", true));
    worldFrame_ = declare_parameter("world_frame_id", "", descr("World frame ID ", true));
    correct_every_n_frames_ = declare_parameter("correct_every_n_frames", 1, descr("Runs the pose correction every n frames" , true));

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

    // Check that each dock id is also listed in tag.ids
    for(auto id : dockIDs)
    {
        if(std::find(tagIDs.begin(), tagIDs.end(), id) == tagIDs.end())
        {
            throw std::runtime_error("ID specified in tag.dock_ids is missing from tag.ids!");
        }
    }
    dockIDs_ = dockIDs;

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

        // Copy the tag ids and their respective sizes into internal data structures
        tagIDs_ = tagIDs;
        for(size_t i = 0; i < tagIDs.size(); i++) 
        {
            tagSizes_[tagIDs[i]] = tagSizes[i];
        }
    }

    // Attempt to add tag family to the detector
    if(tag_fun.count(tagFamilyStr_)) 
    {
        tagFamily_ = tag_fun.at(tagFamilyStr_).first();
        tagFamilyDestructor_ = tag_fun.at(tagFamilyStr_).second;
        apriltag_detector_add_family(tagDetector_, tagFamily_);
    }
    else 
    {
        throw std::runtime_error("Unsupported tag family: " + tagFamilyStr_);
    }

    // Initialize static transformations 
    initTFs();

    // Create Pose Publishers to reset EKF and send the pose for docking
    posePub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("set_pose", 10);
    dockingPub_ = create_publisher<geometry_msgs::msg::PoseStamped>("detected_dock_pose", 10);
}

// Destructor
PoseCorrectionNode::~PoseCorrectionNode()
{
    apriltag_detector_destroy(tagDetector_);
    tagFamilyDestructor_(tagFamily_);
}


// LINK - onCamera: Image callback for tag detection, transform publishing, and pose correction
void PoseCorrectionNode::onCamera(
    const sensor_msgs::msg::Image::ConstSharedPtr& img,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camInfo)
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


    // Set up a pose message for docking
    geometry_msgs::msg::PoseStamped dockingTagPose;
    dockingTagPose.header = img->header;

    // Set up vector of transformations to publish
    std::vector<geometry_msgs::msg::TransformStamped> transformsToTags;
    std::vector<int> consideredTagIDs_;

    // Don't waste any time if we don't have any tags
    if(zarray_size(detections) == 0)
    {
        apriltag_detections_destroy(detections);
        detectionMap_.clear();
        return;
    }

    // LINK - tag detection loop: loops through each detection to publish transforms and estimate pose
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
            // INFO("### Estimating pose of detection %d!", i);
            geometry_msgs::msg::TransformStamped tf;
            tf.header = img->header;
            int currTagID = currDetection->id;
            char* currTagName = currDetection->family->name;
            std::string familyAndID = std::string(currTagName) + ":" + std::to_string(currTagID);
            tf.child_frame_id = tagFrames_.count(currTagID) ? tagFrames_.at(currTagID) : familyAndID;
            const double size = tagSizes_.count(currTagID) ? tagSizes_.at(currTagID) : tagEdgeSize_;
            tf.transform = estimatePose_(currDetection, intrinsics, size);

            // Calculate the length of the translation, and determine if its the closest tag
            Vector3f transVec(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
            double tagDist = transVec.norm();
            if(tagDist < maxTagDist_) 
            {
                transformsToTags.push_back(tf);
                consideredTagIDs_.push_back(currDetection->id);
                detectionMap_[currDetection->id] = currDetection;
            }
            else
            {
                continue;
            }
        }
        else 
        {
            RCLCPP_WARN(get_logger(), "Detection: %d. Camera is not calibrated, will not be estimating pose!", i);
        }
    }

    // Publish detections whenever we see them
    detectionPub_->publish(detectionsMsg);

    // Throttle how often the tag transforms and pose correction parts run by returning early
    if(detectionThrottle_++ != 0)
    {
        apriltag_detections_destroy(detections);
        detectionMap_.clear();
        detectionThrottle_ %= correct_every_n_frames_;
        return;
    }

    // Don't waste any effort if we haven't found any tags within our max transform distance
    if(consideredTagIDs_.size() == 0)
    {
        apriltag_detections_destroy(detections);
        detectionMap_.clear();
        return;
    }

    // Broadcast transforms 
    tfBroadcaster_.sendTransform(transformsToTags);

    // LINK - Compute global pose from each detected tag
    std::vector<detectionPose> transformVec;
    for(auto id : consideredTagIDs_)
    {
            // Use TF to get the local transform in the right coordinate frame
            tf2::Transform tagToHusky;
            getTransformFromTf("tag" + tagFamilyStr_ + ":" + std::to_string(id), "base_link", tagToHusky);

            // Compute the global transform
            tf2::Transform out = computeTransform(tagToHusky, id);
            detectionPose pose;
            pose.id = id;
            pose.detection = detectionMap_[id];
            pose.globalTransform = out;
            pose.localTransform = tagToHusky;
            transformVec.push_back(pose);

            // LINK - Publish docking pose if it's one of the docking tags
            if(std::find(dockIDs_.begin(), dockIDs_.end(), id) != dockIDs_.end())
            {
                // Pitch because its in the apriltag's coordinate frame w.r.t the map frame
                double r=0, p=1.57, y=0;
                tf2::Quaternion rotation;
                rotation.setRPY(r, p, y);
                rotation.normalize();

                // // Convert to pose message and publish
                geometry_msgs::msg::PoseStamped dockMsg;                

                dockMsg.header.stamp = this->get_clock()->now();
                dockMsg.header.frame_id = "tag" + tagFamilyStr_ + ":" + std::to_string(id);
                // dockMsg.pose.position.x = tagToHusky.getOrigin().getX();
                // dockMsg.pose.position.y = tagToHusky.getOrigin().getY();
                // dockMsg.pose.position.z = 0.2;
                dockMsg.pose.orientation.w = rotation.getW();
                dockMsg.pose.orientation.x = rotation.getX();
                dockMsg.pose.orientation.y = rotation.getY();
                dockMsg.pose.orientation.z = rotation.getZ();
                dockingPub_->publish(dockMsg);
            }

    }

    // Compute the weighted average of all global transforms from each detected tag
    tf2::Transform globalTf = averageTransforms(transformVec);

    // Publish the message to reset the EKF pose with the averaged transform
    geometry_msgs::msg::PoseWithCovarianceStamped poseMsg;
    poseMsg.header.stamp = this->get_clock()->now();
    poseMsg.header.frame_id = "map";
    poseMsg.pose.pose.position.x = globalTf.getOrigin().getX();
    poseMsg.pose.pose.position.y = globalTf.getOrigin().getY();
    poseMsg.pose.pose.position.z = globalTf.getOrigin().getZ();
    poseMsg.pose.pose.orientation.w = globalTf.getRotation().getW();
    poseMsg.pose.pose.orientation.x = globalTf.getRotation().getX();
    poseMsg.pose.pose.orientation.y = globalTf.getRotation().getY();
    poseMsg.pose.pose.orientation.z = globalTf.getRotation().getZ();
    posePub_->publish(poseMsg);

    // Cleanup
    apriltag_detections_destroy(detections);
    detectionMap_.clear();

}

// Set parameters on the fly if changed using Parameter Detector
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


// LINK - initTfs: sets up and publishes static transforms for global pose correction
void PoseCorrectionNode::initTFs()
{
    // Grab the camera optical frame to camera base frame
    std::string camLeftFrame = cameraName_ + "_left_camera_frame";
    std::string camBaseFrame = cameraName_ + "_camera_link";
    bool tfOk = getTransformFromTf(camLeftFrame, camBaseFrame, camLeftToBase_);

    if(!tfOk) 
    {
        RCLCPP_ERROR(get_logger(), "Could not grab transform '%s' -> '%s', Please verify the parameters and the status "
                                    "of the 'ZED State Publisher' node!",
                        camBaseFrame.c_str(), camLeftFrame.c_str());
        exit(EXIT_FAILURE);
    }

    // Get a transform between the center and base link to use when correcting the EKF pose
    tfOk = getTransformFromTf(cameraName_ + "_camera_center", "base_link", camCenterToBaseLink_);
    if(!tfOk) 
    {
        RCLCPP_ERROR(get_logger(), "Could not grab transform '%s' -> '%s', Please verify the parameters and the status "
                                    "of the 'ZED State Publisher' node!",
                        (cameraName_ + "_camera_link").c_str(), "base_link");
        exit(EXIT_FAILURE);
    }

    for(int64_t id : tagIDs_) 
    {
        // Create and send the static global transform for each tag defined in the config file
        geometry_msgs::msg::TransformStamped globalTagTransform;
        globalTagTransform.header.stamp = this->get_clock()->now();
        globalTagTransform.header.frame_id = worldFrame_;
        globalTagTransform.child_frame_id = tagIDtoFrame_[id];
        std::vector<_Float64> currTransform = globalTagPoseMap_[id];
        tf2::Quaternion globalRot;
        globalRot.setRPY(currTransform[3], currTransform[4], currTransform[5]);
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
}


// LINK - compute transform: given a local transform to a tag and its id, compute the robot's global position
tf2::Transform PoseCorrectionNode::computeTransform(tf2::Transform & tf, int id)
{
    // Change the reference frame (basis) from the tag to the camera
    tf2::Transform cameraMapPose = tf;
    tf2::Matrix3x3 basis = tf2::Matrix3x3(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0);
    tf2::Transform coordAlign;// aligns the coordinate frames of the detected tag to the known global tag position transform
    coordAlign.setIdentity();
    coordAlign.setBasis(basis);
    cameraMapPose.mult(coordAlign.inverse(), cameraMapPose);

    // Set up vector of transformations to publish
    std::vector<geometry_msgs::msg::TransformStamped> tfs_vec;

    // Publish the global tag pose from the camera
    geometry_msgs::msg::TransformStamped globalTestTf;
    globalTestTf.header.stamp = this->get_clock()->now();
    globalTestTf.header.frame_id = tagIDtoFrame_[id];
    globalTestTf.child_frame_id = tagFamilyStr_ + ":" + std::to_string(id) + "_to_" + cameraName_;
    globalTestTf.transform.translation.x = cameraMapPose.getOrigin().getX();
    globalTestTf.transform.translation.y = cameraMapPose.getOrigin().getY();
    globalTestTf.transform.translation.z = cameraMapPose.getOrigin().getZ();
    globalTestTf.transform.rotation.x = cameraMapPose.getRotation().getX();
    globalTestTf.transform.rotation.y = cameraMapPose.getRotation().getY();
    globalTestTf.transform.rotation.z = cameraMapPose.getRotation().getZ();
    globalTestTf.transform.rotation.w = cameraMapPose.getRotation().getW();
    tfs_vec.push_back(globalTestTf);
    tfBroadcaster_.sendTransform(tfs_vec);
    tfs_vec.clear();

    // Use the previously broadcasted local transform to get the location of the robot in the map frame
    tf2::Transform out;
    getTransformFromTf(globalTestTf.child_frame_id, "map", out);
    out = out.inverse();
    out.setRotation(out.getRotation().normalize());

    globalTestTf.header.stamp = this->get_clock()->now();
    globalTestTf.header.frame_id = "map";
    globalTestTf.child_frame_id = "pose_correction_test";
    globalTestTf.transform.translation.x = out.getOrigin().getX();
    globalTestTf.transform.translation.y = out.getOrigin().getY();
    globalTestTf.transform.translation.z = out.getOrigin().getZ();
    globalTestTf.transform.rotation.x = out.getRotation().getX();
    globalTestTf.transform.rotation.y = out.getRotation().getY();
    globalTestTf.transform.rotation.z = out.getRotation().getZ();
    globalTestTf.transform.rotation.w = out.getRotation().getW();
    tfs_vec.push_back(globalTestTf);

    // Broadcast the global transform of the robot
    tfBroadcaster_.sendTransform(tfs_vec);

    return out;
}

// LINK - averageTransforms: combines multiple global pose estimates for the robot into one, as a weighted average
// using distance to the tag as the weighting metric.
// NOTE - a better metric is probably how much of the camera frame a tag takes up, this accounts for distance
// and tag size, but need to do some extra steps to compute that
tf2::Transform PoseCorrectionNode::averageTransforms(std::vector<detectionPose>& transformVec)
{
    // Initialize weights as the inverse of the squared distance (Distant tags fall off faster)
    std::vector<double> weightVec;
    std::vector<Eigen::Quaterniond> quaternionVec;
    double weightSum = 0;
    for(long unsigned int i = 0; i < transformVec.size(); i++)
    {
        tf2::Transform local = transformVec[i].localTransform;
        Vector3f transVec(local.getOrigin().getX(), local.getOrigin().getY(), local.getOrigin().getZ());
        double tagDist = transVec.norm();
        double weight = 1 / std::pow(tagDist, 2);
        weightVec.push_back(weight);
        weightSum += weight;
    }

    // Normalize weights
    for(long unsigned int i = 0; i < weightVec.size(); i++)
    {
        weightVec[i] /= weightSum;
    }

    // Perform weighted average of translation, break out quaternions into a separate vector
    Eigen::Vector3d transSum(0, 0, 0);
    for(long unsigned int i = 0; i < transformVec.size(); i++)
    {
        // Grab weight and transform for readability
        double w_i = weightVec[i]; 
        tf2::Transform currTf = transformVec[i].globalTransform;
        
        // Sum our transform vector using eigen
        Eigen::Vector3d currTrans(w_i * currTf.getOrigin().getX(), w_i * currTf.getOrigin().getY(), w_i * currTf.getOrigin().getZ());
        transSum += currTrans;

        // Convert quaternion into a different type and add it to a vector
        Eigen::Quaterniond q(currTf.getRotation().getW(), currTf.getRotation().getX(), currTf.getRotation().getY(), currTf.getRotation().getZ());
        quaternionVec.push_back(q);
    }

    // Perform weighted average of rotations using the Markley method
    Eigen::Quaterniond avgQuaternion = weightedAverageQuaternion(quaternionVec, weightVec);

    // Convert back to tf2::Transform
    tf2::Transform finalTf;
    finalTf.setOrigin(tf2::Vector3(transSum(0), transSum(1), transSum(2)));
    finalTf.setRotation(tf2::Quaternion(avgQuaternion.x(), avgQuaternion.y(), avgQuaternion.z(), avgQuaternion.w()));

    return finalTf;
}

// LINK - weightedAverageQuaternion
Eigen::Quaterniond PoseCorrectionNode::weightedAverageQuaternion(std::vector<Eigen::Quaterniond>& quats, std::vector<double>& weights)
{
    // This function is an adaptation of tbirdal's algorithm for this process, based on Markley et al. 2007
    // https://tbirdal.blogspot.com/2019/10/i-allocate-this-post-to-providing.html

    // Verify we were passed quaternions and weights, and that those sizes match
    Eigen::Quaterniond quaternionOut;
    if(quats.size() == 0 || weights.size() == 0)
    {
        throw std::invalid_argument("Quaternion list or weight list has no entries!");
    }
    if(quats.size() != weights.size())
    {
        throw std::invalid_argument("Quaternion list and weight list have different lengths!");
    }

    // Build matrix of the outer products of the quaternion by its weight
    double totalWeight = 0;
    Eigen::Matrix4d accumulator = Eigen::Matrix4d::Zero();
    for(long unsigned int i = 0; i < quats.size(); i++)
    {
        // Break quaternion into a vector 
        Eigen::Vector4d q(quats[i].w(), quats[i].x(), quats[i].y(), quats[i].z());
        q.normalize();

        // Correct sign to have everything on the same hemisphere
        if(q(0) < 0)
        {
            q = -q;
        }

        // Calculate the outer product, scale by weight, and add to matrix
        accumulator += weights[i] * (q * q.transpose());
        totalWeight += weights[i];
    }

    // Normalize the matrix
    accumulator /= totalWeight;

    // Solve for the eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(accumulator);
    if(eigensolver.info() != Eigen::Success)
    {
        throw std::runtime_error("Failed to solve for eigen values!");
    }
        
    // Grab the eigen vector for the largest eigenvalue
    Eigen::Vector4d average = eigensolver.eigenvectors().col(3);

    return Eigen::Quaterniond(average(0), average(1), average(2), average(3));
}


// LINK - getTransformFromTf : helper function to grab a broadcast transform from tf2
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
        transf_msg =
            transformBuffer_->lookupTransform(targetFrame, sourceFrame, TIMEZERO_ROS, 1s);
    }
    catch(const tf2::TransformException& ex) 
    {
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

    return true;
}
