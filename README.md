# AprilTag ROS 2 Node

This is a ROS 2 wrapper for the AprilTag library. It detects AprilTags in images and publishes their pose, ID, and other metadata. This node is designed to be a flexible and efficient solution for robotics applications that require visual fiducial markers.

**Key Features:**

*   **ROS 2 Integration:** Seamlessly integrates with ROS 2, providing topics for tag detections and transforms.
*   **High Performance:** Utilizes the AprilTag library for efficient tag detection.
*   **Composable Node:** Can be used as a composable node for efficient intra-process communication.
*   **Configurable:** Highly configurable through a YAML file, allowing you to customize tag families, sizes, and more.

## Getting Started

### Prerequisites

*   ROS 2 (Foxy Fitzroy or later)
*   OpenCV
*   `apriltag` library
*   `apriltag_msgs` messages

### Building

To build the `apriltag_ros` package, clone this repository into your ROS 2 workspace and build it with `colcon`:

```bash
cd ~/ros2_ws/src
git clone https://github.com/AprilRobotics/apriltag_ros.git
cd ..
colcon build
```

## Usage

### Topics

#### Subscriptions

| Topic | Type | Description |
|---|---|---|
| `image_rect` | `sensor_msgs/msg/Image` | The input image. |
| `image_rect/compressed` | `sensor_msgs/msg/CompressedImage` | The compressed input image. |
| `camera_info` | `sensor_msgs/msg/CameraInfo` | The camera calibration information. |

The node uses an `image_transport::CameraSubscriber` to subscribe to the input image and camera info. The specific topic names depend on the `image_transport` parameter.

#### Publishers

| Topic | Type | Description |
|---|---|---|
| `/tf` | `tf2_msgs/msg/TFMessage` | The transforms of the detected AprilTags. |
| `detections` | `apriltag_msgs/msg/AprilTagDetectionArray` | An array of detected AprilTags, including their ID, pose, and other metadata. |

### Launching the Node

There are two ways to run the AprilTag node: as a standalone node or as a composable node.

#### Standalone Node

You can run the AprilTag node as a standalone executable using `ros2 run`:

```bash
ros2 run apriltag_ros apriltag_node --ros-args \
    -r image_rect:=/camera/image_rect \
    -r camera_info:=/camera/camera_info \
    --params-file /path/to/your/config.yaml
```

Make sure to replace `/camera/image_rect`, `/camera/camera_info`, and `/path/to/your/config.yaml` with your actual topic names and configuration file path.

#### Composable Node

For improved performance, you can run the AprilTag node as a composable node within a container process. This avoids data serialization and deserialization when passing messages between nodes in the same process.

The `apriltag_ros` package provides an example launch file that demonstrates how to run the AprilTag node as a composable node with a camera node from the `camera_ros` package. To run the example, you first need to install the `camera_ros` package:

```bash
sudo apt install ros-$ROS_DISTRO-camera-ros
```

Then, you can launch the example:

```bash
ros2 launch apriltag_ros camera_36h11.launch.yml
```

This will launch a container process with the `AprilTagNode` and `camera::CameraNode` components, with intra-process communication enabled.

## Configuration

The node can be configured using a YAML file. For more information on the ROS parameter syntax, please refer to the [ROS 2 documentation](https://docs.ros.org/en/humble/How-To-Guides/Using-Parameters-in-a-Class-CPP.html).

An example configuration file can be found at `cfg/tags_36h11.yaml`.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_transport` | string | "raw" | The image transport to use. Can be "raw" or "compressed". |
| `family` | string | "36h11" | The AprilTag family to detect. Supported families: `16h5`, `25h9`, `36h11`, `Circle21h7`, `Circle49h12`, `Custom48h12`, `Standard41h12`, `Standard52h13`. |
| `size` | double | 1.0 | The default edge size of the AprilTags in meters. |
| `profile` | bool | false | Whether to print profiling information to the console. |
| `max_hamming` | int | 0 | The maximum allowed Hamming distance for a tag detection. |
| `pose_estimation_method` | string | "pnp" | The method for estimating the tag pose. Can be "pnp" (more accurate) or "homography" (faster). Set to an empty string to disable pose estimation. |
| `detector.threads` | int | 1 | The number of threads to use for tag detection. |
| `detector.decimate` | double | 2.0 | The decimation factor for the quad detection. |
| `detector.blur` | double | 0.0 | The sigma of the Gaussian blur for the quad detection. |
| `detector.refine` | int | 1 | Whether to refine the edges of the detected quads. |
| `detector.sharpening` | double | 0.25 | The sharpening factor for the decoded images. |
| `detector.debug` | int | 0 | Whether to write additional debugging images to the current working directory. |
| `tag.ids` | array of ints | `[]` | A list of tag IDs to publish transforms for. If empty, all detected tags are published. |
| `tag.frames` | array of strings | `[]` | A list of frame names to use for the tag transforms. The order of the frame names must correspond to the order of the tag IDs in `tag.ids`. |
| `tag.sizes` | array of doubles | `[]` | A list of tag sizes to use for the tag transforms. The order of the sizes must correspond to the order of the tag IDs in `tag.ids`. |

## Contributing

Contributions are welcome! If you have a bug fix or a new feature, please open a pull request.

### Pull Request Guidelines

*   Please make sure your code follows the style of the existing code.
*   Please run the linters and tests before submitting your pull request.
*   Please provide a clear and concise description of your changes.
