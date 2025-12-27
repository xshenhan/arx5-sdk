import time

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import numpy as np


def main(model: str = "L5", interface: str = "can0"):

    # To initialize robot with different configurations,
    # you can create RobotConfig and ControllerConfig by yourself and modify based on it
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    # Modify the default configuration here
    # controller_config.controller_dt = 0.01 # etc.

    USE_MULTITHREADING = True
    if USE_MULTITHREADING:
        # Will create another thread that communicates with the arm, so each send_recv_once() will take no time
        # for the main thread to execute. Otherwise (without background send/recv), send_recv_once() will block the
        # main thread until the arm responds (usually 2ms).
        controller_config.background_send_recv = True
    else:
        controller_config.background_send_recv = False

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    # arx5_joint_controller.reset_to_home()
    while True:
        joint = arx5_joint_controller.get_joint_state()
        eef = arx5_joint_controller.get_eef_state()
        home_pose = arx5_joint_controller.get_home_pose()
        print(f"home_pose: {home_pose}")
        print(f"eef: {eef.gripper_pos}")
        print(f"joint: {joint.pos()}")
        time.sleep(0.1)

if __name__ == "__main__":
    main()