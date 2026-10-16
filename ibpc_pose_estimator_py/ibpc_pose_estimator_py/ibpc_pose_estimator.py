# todo(Yadunund): Add copyright.
import os
import torch
from cv_bridge import CvBridge
import numpy as np
import json
from scipy.spatial.transform import Rotation
import sys
from typing import List, Optional, Union
from geometry_msgs.msg import Pose as PoseMsg
from ibpc_interfaces.msg import Camera as CameraMsg
from ibpc_interfaces.msg import Photoneo as PhotoneoMsg
from ibpc_interfaces.msg import PoseEstimate as PoseEstimateMsg
from ibpc_interfaces.srv import GetPoseEstimates

import rclpy
from rclpy.node import Node

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = "/opt/ros/overlay/src/ibpc_pose_estimator_py/ibpc_pose_estimator_py"
sys.path.append(BASE_DIR)

ISM_DIR = os.path.join(BASE_DIR, 'Instance_Segmentation_Model')
sys.path.append(ISM_DIR)
sys.path.append(os.path.join(ISM_DIR, 'provider'))
sys.path.append(os.path.join(ISM_DIR, 'utils'))
sys.path.append(os.path.join(ISM_DIR, 'model'))
sys.path.append(os.path.join(ISM_DIR, 'segment_anything'))

PEM_DIR = os.path.join(BASE_DIR, 'Pose_Estimation_Model')
sys.path.append(PEM_DIR)
sys.path.append(os.path.join(PEM_DIR, 'config'))
sys.path.append(os.path.join(PEM_DIR, 'utils'))
sys.path.append(os.path.join(PEM_DIR, 'model'))
sys.path.append(os.path.join(PEM_DIR, 'model', 'pointnet2'))

RENDER_DIR = os.path.join(BASE_DIR, 'Render')
sys.path.append(RENDER_DIR)
sys.path.append(os.path.join(RENDER_DIR, 'blender'))

from Instance_Segmentation_Model.run_inference_custom import ISM
from Pose_Estimation_Model.run_inference_custom import PEM
from Render.template_renderer import TemplateRenderer

# Helper functions
def ros_pose_to_mat(pose: PoseMsg):
    r = Rotation.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    matrix = r.as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = matrix
    pose_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return pose_matrix


class Camera:
    """
    Represents a camera with its pose, intrinsics, and image data,
    initialized from either a CameraMsg or a PhotoneoMsg ROS message.

    Attributes:
        name (str): The name of the camera (taken from the message's frame_id).
        pose (np.ndarray): The 4x4 camera pose matrix (world-to-camera transformation),
                          converted from the ROS message's pose.
        intrinsics (np.ndarray): The 3x3 camera intrinsics matrix, reshaped from the
                                  message's K matrix.
        rgb (np.ndarray): The RGB image data, converted from the ROS message.
        depth (np.ndarray): The depth image data, converted from the ROS message.
        aolp (np.ndarray, optional): The Angle of Linear Polarization data,
                                     converted from the CameraMsg (None for PhotoneoMsg).
        dolp (np.ndarray, optional): The Degree of Linear Polarization data,
                                     converted from the CameraMsg (None for PhotoneoMsg).
    """

    def __init__(self, msg: Union[CameraMsg, PhotoneoMsg]):
        """
        Initializes a new Camera object from a ROS message.

        Args:
            msg: Either a CameraMsg or a PhotoneoMsg ROS message containing
                 camera information.

        Raises:
           TypeError: If the input `msg` is not of the expected type.
        """
        try:
            br = CvBridge()
        except Exception as e:
            raise RuntimeError("Failed to initialize CvBridge") from e

        if not isinstance(msg, (CameraMsg, PhotoneoMsg)):
            raise TypeError("Input message must be of type CameraMsg or PhotoneoMsg")

        self.name: str = (msg.info.header.frame_id,)
        self.pose: np.ndarray = ros_pose_to_mat(msg.pose)
        self.intrinsics: np.ndarray = np.array(msg.info.k).reshape(3, 3)
        self.rgb = br.imgmsg_to_cv2(msg.rgb)
        self.depth = br.imgmsg_to_cv2(msg.depth)
        if isinstance(msg, CameraMsg):
            self.aolp: Optional[np.ndarray] = br.imgmsg_to_cv2(msg.aolp)
            self.dolp: Optional[np.ndarray] = br.imgmsg_to_cv2(msg.dolp)
        else:  # PhotoneoMsg
            self.aolp: Optional[np.ndarray] = None
            self.dolp: Optional[np.ndarray] = None


class PoseEstimator(Node):

    def __init__(self):
        super().__init__("bpc_pose_estimator")
        self.get_logger().info("Starting bpc_pose_estimator...")
        # Declare parameters
        self.model_dir = (
            self.declare_parameter("model_dir", "").get_parameter_value().string_value
        )
        if self.model_dir == "":
            raise Exception("ROS parameter model_dir not set.")
        self.get_logger().info(f"Model directory set to {self.model_dir}.")

        # 3d_models directory
        self.cads_dir = "/opt/ros/underlay/install/3d_models"
        render = TemplateRenderer(
            blender_path=os.path.join(RENDER_DIR, "blender"),
            models_dir=self.cads_dir,
            output_dir=self.model_dir)
        render.verify_templates()
        
        # Configure the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")
        
        # Initialize the ISM models
        self.checkpoints_path = os.path.join(self.model_dir, "checkpoints")
        self.ism = ISM(checkpoints_path=self.checkpoints_path, segmentor_model="sam", device=self.device)

        # Initialize the PEM models
        self.pem = PEM(checkpoints_path=self.checkpoints_path, device=self.device)

        self.image_count = 0
        self.ism_error_count = 0
        self.pem_error_count = 0
        self.ism_cuda_oom_count = 0
        self.pem_cuda_oom_count = 0
        self.pem_zero_est_count = 0
        
        # Create a Service
        srv_name = "/get_pose_estimates"
        self.get_logger().info(f"Pose estimates can be queried over srv {srv_name}.")
        self.srv = self.create_service(GetPoseEstimates, srv_name, self.srv_cb)
        self.get_logger().info(f"Service {srv_name} created.")

    def srv_cb(self, request, response):
        if len(request.object_ids) == 0:
            self.get_logger().warn("Received request with empty object_ids.")
            return response
        if len(request.cameras) < 3:
            self.get_logger().warn("Received request with insufficient cameras.")
            return response
        try:
            self.get_logger().info("Received request for pose estimates.")
            cam_1 = Camera(request.cameras[0])
            cam_2 = Camera(request.cameras[1])
            cam_3 = Camera(request.cameras[2])
            photoneo = Camera(request.photoneo)
            self.get_logger().info("Cameras initialized successfully.")
            response.pose_estimates = self.get_pose_estimates(
                request.object_ids, cam_1, cam_2, cam_3, photoneo
            )
        except:
            self.get_logger().error("Error calling get_pose_estimates.")
        return response

    def get_pose_estimates(
        self,
        object_ids: List[int],
        cam_1: Camera,
        cam_2: Camera,
        cam_3: Camera,
        photoneo: Camera,
    ) -> List[PoseEstimateMsg]:
        
        try:
            self.get_logger().info("Starting pose estimation...")
            cam_info = {}
            cam_info["cam_K"] = list(np.array(photoneo.intrinsics).flatten())
            cam_info["depth_scale"] = 0.1
        except Exception as e:
            self.get_logger().error(f"Error in camera info: {e}")
            return []
        
        self.image_count += 1
        pose_estimates = []
        self.get_logger().info("Looping through object IDs...")
        for obj_id in object_ids:
            try:
                self.get_logger().info(f"Processing object ID: {obj_id}")
                cad_path = os.path.join(self.cads_dir, f"obj_{obj_id:06d}.ply")
                if not os.path.exists(cad_path):
                    self.get_logger().error(f"CAD file not found: {cad_path}")
                    continue
            except Exception as e:
                self.get_logger().error(f"Error in CAD path: {e}")
                continue
            # Run the instance segmentation model
            try:
                self.get_logger().info("Running instance segmentation model...")
                ism_results = self.ism.run_inference(
                    output_dir=self.model_dir,
                    cad_path=cad_path,
                    rgb=photoneo.rgb,
                    depth=photoneo.depth,
                    cam_info=cam_info,
                    seg_score_thresh=0.35)

            except Exception as e:
                self.get_logger().error(f"Error in instance segmentation: {e}")
                self.ism_error_count += 1
                if "CUDA out of memory" in str(e):
                    self.ism_cuda_oom_count += 1
                continue
            self.get_logger().info("Instance segmentation model completed.")
            
            # Run the pose estimation model
            try:
                self.get_logger().info("Running pose estimation model...")
                detections = self.pem.run_inference(
                    output_dir=self.model_dir,
                    cad_path=cad_path,
                    rgb=photoneo.rgb,
                    depth=photoneo.depth,
                    cam_info=cam_info,
                    seg_data=ism_results,
                    est_score_thresh=0.3)
                pose_estimate_msgs = self.detections_to_pose_msgs(detections, obj_id)
                pose_estimates.extend(pose_estimate_msgs)
            except Exception as e:
                self.get_logger().error(f"Error in pose estimation: {e}")
                self.pem_error_count += 1
                if "CUDA out of memory" in str(e):
                    self.pem_cuda_oom_count += 1
                if "non-empty TensorList" in str(e):
                    self.pem_zero_est_count += 1
                continue
            self.get_logger().info("Pose estimation completed.")
        
        if self.image_count % 3 == 0:
            self.get_logger().info(f"ISM error count: {self.ism_error_count}")
            self.get_logger().info(f"PEM error count: {self.pem_error_count}")
            self.get_logger().info(f"ISM CUDA OOM error count: {self.ism_cuda_oom_count}")
            self.get_logger().info(f"PEM CUDA OOM error count: {self.pem_cuda_oom_count}")
            self.get_logger().info(f"PEM zero estimates error count: {self.pem_zero_est_count}")
            
        return pose_estimates

    def detections_to_pose_msgs(self, detections, obj_id):
        pose_estimate_msgs = []
        
        for detection in detections:
            # Create a new Pose message
            pose_estimate = PoseEstimateMsg()

            pose_estimate.obj_id = int(obj_id)
            pose_estimate.score = detection['score']
            
            # Get rotation matrix and translation vector
            rot_matrix = np.array(detection['R']).reshape(3, 3)
            trans_vector = np.array(detection['t'])
            
            # Convert rotation matrix to quaternion
            r = Rotation.from_matrix(rot_matrix)
            quat = r.as_quat()  # Returns x, y, z, w
            
            # Set orientation (quaternion)
            pose_estimate.pose.orientation.x = quat[0]
            pose_estimate.pose.orientation.y = quat[1]
            pose_estimate.pose.orientation.z = quat[2]
            pose_estimate.pose.orientation.w = quat[3]
            
            # Set position (translation)
            pose_estimate.pose.position.x = trans_vector[0]
            pose_estimate.pose.position.y = trans_vector[1]
            pose_estimate.pose.position.z = trans_vector[2]
            
            # Add to list of pose messages
            pose_estimate_msgs.append(pose_estimate)
        
        return pose_estimate_msgs

def main(argv=sys.argv):
    rclpy.init(args=argv)
    pose_estimator = PoseEstimator()
    rclpy.spin(pose_estimator)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
