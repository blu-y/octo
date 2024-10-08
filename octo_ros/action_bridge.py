import cv2
import jax
import tensorflow_datasets as tfds
import tqdm
import mediapy
import numpy as np
from octo.model.octo_model import OctoModel

# ACTION_DIM_LABELS = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'grasp']

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Float32
from cv_bridge import CvBridge

class TwistPublisher(Node):
    def __init__(self):
        super().__init__('twist_publisher')
        self.get_logger().info("TwistPublisher node started")
        self.get_logger().info("Loading OctoModel...")
        self.octo = OctoModel.load_pretrained("./octo-small-1.5")
        self.get_logger().info("OctoModel loaded")
        self.action_pub = self.create_publisher(Twist, '/twist_controller/commands', 10)
        self.task_sub = self.create_subscription(String, '/task', self.task_cb, 1)
        self.action_pub = self.create_publisher(Twist, '/twist_controller/commands', 4)
        self.wrist_sub = self.create_subscription( Image, '/camera/color/image_raw', self.wrist_cb, 1)
        self.pause_sub = self.create_subscription(Bool, '/pause', self.pause_cb, 1)
        self.linx_sub = self.create_subscription(Float32, '/x_lin', self.x_lin_cb, 1)
        self.angx_sub = self.create_subscription(Float32, '/x_ang', self.x_ang_cb, 1)
        self.get_logger().info("\tros2 topic pub /pause std_msgs/msg/Bool 'data: true' to pause")
        self.get_logger().info("\tros2 topic pub /pause std_msgs/msg/Bool 'data: false' to resume")
        self.get_logger().info("\tros2 topic pub /x_lin std_msgs/msg/Float32 'data: 1.0' to set linear multiplier")
        self.get_logger().info("\tros2 topic pub /x_ang std_msgs/msg/Float32 'data: 1.0' to set angular multiplier")
        self.pause = False
        self.wrist_sub  # prevent unused variable warning
        self.br = CvBridge()
        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.x_lin = 5.0
        self.x_ang = 5.0
        self.pred_actions = []
        self.frames = []
        self.wrist_frames = []
        self.task = None
        self.task = self.octo.create_tasks(texts=["Pick up the pepsi bottle"])
        self.stat = {
            "mask": np.array([ True,  True,  True,  True,  True,  True, False]),
            "max": np.array([0.41691166, 0.25864795, 0.21218234, 3.12220192, 1.86181128, 6.28047848, 1.]),
            "mean": np.array([ 2.17586465e-04,  1.25082981e-04, -1.71083258e-04, -1.61711389e-04, -2.52485683e-04,  2.51578749e-04,  5.87948442e-01]),
            "min": np.array([-0.40075102, -0.13874775, -0.225539, -3.20107865, -1.86181128, -6.27907562, 0.]),
            "std": np.array([0.00963238, 0.01350064, 0.0125106, 0.02814521, 0.03028243, 0.07585602, 0.48771909])
        }

    def x_lin_cb(self, msg):
        self.x_lin = msg.data
        self.get_logger().info(f"linear multiplier: {self.x_lin}")

    def x_ang_cb(self, msg):
        self.x_ang = msg.data
        self.get_logger().info(f"angular multiplier: {self.x_ang}")

    def pause_cb(self, msg):
        self.pause = msg.data
        if self.pause: self.get_logger().info("Pausing")
        else: self.get_logger().info("Resuming")

    def task_cb(self, msg):
        language_instruction = msg.data
        self.get_logger().info(f"Received task: {language_instruction}")
        self.task = self.octo.create_tasks(texts=[language_instruction])

    def wrist_cb(self, msg):
        if not self.task:
            self.get_logger().info("No task received yet.")
            self.get_logger().info("\tros2 topic pub /task std_msgs/msg/String 'data: \"Pick up the block\"'")
            # ros2 topic pub /task std_msgs/msg/String 'data: "Pick up the pepsi bottle"' -1
            return
        wrist_frame = self.br.imgmsg_to_cv2(msg)
        ret, frame = self.cap.read()
        wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("wrist", wrist_frame)
        cv2.imshow("camera", frame)
        cv2.waitKey(1)
        if self.pause:
            return
        if not ret:
            print("Error: Could not read frame.")
            return
        self.frames.append(cv2.resize(frame, (256, 256)))
        self.wrist_frames.append(cv2.resize(wrist_frame, (128, 128)))
        if len(self.frames) < 2 or len(self.wrist_frames) < 2:
            return
        self.frames = self.frames[-2:]
        self.wrist_frames = self.wrist_frames[-2:]
        actions = self.sample_actions()
        self.publish_actions(actions)
        self.pred_actions.append(actions)

    def sample_actions(self):
        input_images = np.stack(self.frames)[None]
        wrist_images = np.stack(self.wrist_frames)[None]
        # self.get_logger().info(f"type(input_images): {type(input_images)}")
        # self.get_logger().info(f"type(wrist_images): {type(wrist_images)}")
        # self.get_logger().info(f"input_images.shape: {input_images.shape}")
        # self.get_logger().info(f"wrist_images.shape: {wrist_images.shape}")
        observation = {
            'image_primary': input_images,
            'image_wrist': wrist_images,
            'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
        }
        actions = self.octo.sample_actions(observation, self.task, unnormalization_statistics=self.stat, rng=jax.random.PRNGKey(0))
        actions = actions[0] # remove batch dimension
        return actions

    def publish_actions(self, actions):
        msg = Twist()
        actions = actions[0]
        actions = [float(actions[i]) for i in range(6)]
        actions[:3] = [actions[i] * self.x_lin for i in range(3)]
        actions[3:] = [actions[i] * self.x_ang for i in range(3)]
        self.get_logger().info(f"actions: [{actions[0]:.5f}, {actions[1]:.5f}, {actions[2]:.5f}, {actions[3]:.5f}, {actions[4]:.5f}, {actions[5]:.5f}]")
        msg.linear.x = actions[0]
        msg.linear.y = actions[1]
        msg.linear.z = actions[2]
        msg.angular.x = actions[3]
        msg.angular.y = actions[4]
        msg.angular.z = actions[5]
        self.action_pub.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg)

def main(args=None):
    rclpy.init(args=args)
    twist_publisher = TwistPublisher()
    rclpy.spin(twist_publisher)
    twist_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()