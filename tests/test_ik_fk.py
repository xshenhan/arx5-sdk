# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from typing import List

import numpy as np
import pink
import pinocchio as pin
import qpsolvers
from loguru import logger
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask

class LPConstrainedSE3Filter:
    def __init__(self, alpha, dt=1 / 60):
        self.alpha = alpha
        self.dt = dt
        self.is_init = False

        self.prev_pos = None
        self.prev_vel = None

        self.max_vel = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
        self.max_acc = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3])

    def next(self, target):
        if not self.is_init:
            self.prev_pos = pin.SE3(target)
            self.prev_vel = np.zeros(6)
            self.prev_acc = np.zeros(6)
            self.is_init = True
            return np.array(self.prev_pos)

        ip_pos = pin.SE3.Interpolate(self.prev_pos, pin.SE3(target), self.alpha)
        ip_vel = pin.log(ip_pos.actInv(self.prev_pos)).vector / self.dt
        ip_acc = (ip_vel - self.prev_vel) / self.dt

        acc = np.clip(ip_acc, -self.max_acc, self.max_acc)
        vel = np.clip(self.prev_vel + acc * self.dt, -self.max_vel, self.max_vel)
        pos = self.prev_pos * (
            ~pin.exp(vel * self.dt)
        )  # Caution! * means matrix multiplication in pinocchio

        self.prev_pos = pos
        self.prev_vel = vel

        return np.array(self.prev_pos)

    def reset(self):
        self.y = None
        self.is_init = False

class SingleArmIK:
    def __init__(
        self,
        urdf_path: str,
        package_dirs: list,
        init_q: np.ndarray = None,  # 允许传入初始关节位置
        orientation_cost: float = 1.0,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
        joint_names_to_lock: List[str] = [],
        ee_link_name: str = "",
    ):
        # 加载完整机器人模型
        self.full_robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=package_dirs,
            root_joint=None,
        )

        # 保存关节信息
        self.joint_names = [name for name in self.full_robot.model.names]
        self.active_joint_ids = []
        self.locked_joint_ids = []

        # 确定活动关节和锁定关节
        for joint_id in range(1, self.full_robot.model.njoints):  # 跳过universe关节
            joint_name = self.full_robot.model.names[joint_id]
            if joint_name in joint_names_to_lock:
                self.locked_joint_ids.append(joint_id)
            else:
                self.active_joint_ids.append(joint_id)

        # 保存完整机器人的默认配置
        self.full_q0 = np.copy(self.full_robot.q0)
        self.full_q0[0] = -1.57
        self.full_q0[1] = -1.57

        # 如果提供了初始姿态，使用它来更新默认配置
        if init_q is not None:
            assert len(init_q) == len(
                self.full_q0
            ), f"初始位置长度不匹配: {len(init_q)} vs {len(self.full_q0)}"
            self.full_q0 = np.copy(init_q)

        # 构建精简模型
        self.robot = self.full_robot.buildReducedRobot(self.locked_joint_ids)

        # 映射从完整机器人到精简机器人的关节位置
        self.reduced_indices = []
        reduced_joint_names = [
            self.robot.model.names[i] for i in range(self.robot.model.njoints)
        ]
        for joint_id in self.active_joint_ids:
            joint_name = self.full_robot.model.names[joint_id]
            if joint_name in reduced_joint_names:
                self.reduced_indices.append(reduced_joint_names.index(joint_name))

        # 创建适用于精简模型的初始配置
        self.reduced_q0 = np.copy(self.robot.q0)

        # 从完整配置更新精简配置中的活动关节
        self._update_reduced_q_from_full(self.full_q0)

        # 初始化配置
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.reduced_q0
        )

        # 初始化任务
        self.ee_task = FrameTask(
            ee_link_name,  # 需根据实际URDF调整
            position_cost=1.0,
            orientation_cost=orientation_cost,
        )
        self.posture_task = PostureTask(cost=1e-3)

        # 从当前配置设置姿态任务目标
        self.posture_task.set_target_from_configuration(self.configuration)

        self.tasks = [
            self.ee_task,
            self.posture_task,
        ]

        self.ee_link_name = ee_link_name

        # QP求解器配置
        self.solver = qpsolvers.available_solvers[0]
        self.dt = 0.01
        self.damp = damp

        # 初始化滤波器
        if use_in_lp_filter:
            self.filter = LPConstrainedSE3Filter(in_lp_alpha)
        else:
            self.filter = None

        # 输出关节信息
        print(f"活动关节数量: {len(self.active_joint_ids)}")
        print(
            f"活动关节: {[self.full_robot.model.names[i] for i in self.active_joint_ids]}"
        )

    def _update_reduced_q_from_full(self, full_q):
        """从完整配置更新精简模型配置"""
        # 遍历活动关节，将值从完整配置复制到精简配置
        for i, joint_id in enumerate(self.active_joint_ids):
            joint_name = self.full_robot.model.names[joint_id]
            joint_idx = self.full_robot.model.getJointId(joint_name)

            # 获取关节在完整模型配置中的索引范围
            joint = self.full_robot.model.joints[joint_idx]
            q_idx = self.full_robot.model.idx_qs[joint_idx]
            nq = joint.nq

            # 获取关节在精简模型中的索引
            reduced_idx = self.robot.model.getJointId(joint_name)
            reduced_q_idx = self.robot.model.idx_qs[reduced_idx]

            # 复制配置
            self.reduced_q0[reduced_q_idx : reduced_q_idx + nq] = full_q[
                q_idx : q_idx + nq
            ]

    def set_init_position(self, init_q):
        """设置新的初始位置"""
        assert len(init_q) == len(
            self.full_q0
        ), f"初始位置长度不匹配: {len(init_q)} vs {len(self.full_q0)}"
        self.full_q0 = np.copy(init_q)

        # 更新精简模型的初始配置
        self._update_reduced_q_from_full(self.full_q0)

        # 更新当前配置
        self.configuration.q = np.copy(self.reduced_q0)

        # 重置任务目标
        self.posture_task.set_target_from_configuration(self.configuration)

        # 重置滤波器
        if self.filter is not None:
            self.filter.reset()

        return self.configuration.q

    def solve(
        self,
        wrist_transform,
        joints,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        """求解双臂逆运动学

        Args:
            wrist_transform: 4x4 齐次变换矩阵，表示ee目标位姿
            orientation_cost: 可选方向权重覆盖值

        Returns:
            (关节位置)
        """
        # 更新方向权重
        device = "cuda"
        if isinstance(joints, torch.Tensor):
            device = joints.device
            joints = joints.cpu().numpy()
        if isinstance(wrist_transform, torch.Tensor):
            device = wrist_transform.device
            wrist_transform = wrist_transform.cpu().numpy()
        if orientation_cost is not None:
            self.ee_task.orientation_cost = orientation_cost

        # 应用平滑滤波
        if self.filter is not None:
            transform = self.filter.next(wrist_transform)
        else:
            transform = wrist_transform

        self.reduced_q0 = joints
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.reduced_q0
        )

        # 设置任务目标
        self.ee_task.set_target(pin.SE3(transform))

        initial_error = np.linalg.norm(
            self.ee_task.compute_error(self.configuration)
        ) + np.linalg.norm(self.ee_task.compute_error(self.configuration))
        error_norm = initial_error

        logger.trace(f"Initial error norm: {error_norm}")

        nb_steps = 0
        damping_factor = 1e-8  #  1e-3
        prev_error = float("inf")
        current_dt = dt
        try:
            while error_norm > stop_thres and (
                nb_steps < max_try_times
                or (nb_steps < max_try_times and prev_error - error_norm > 0.003)
            ):
                dv = solve_ik(
                    self.configuration,
                    tasks=self.tasks,
                    dt=current_dt,
                    damping=damping_factor,
                    safety_break=False,
                    solver="quadprog",
                )
                if nb_steps % 5 == 0:
                    damping_factor *= 1.5
                # 如果误差减小不明显，减小步长
                if prev_error - error_norm < error_norm * 0.01:
                    current_dt *= 0.8
                prev_error = error_norm
                q_out = pin.integrate(
                    self.robot.model, self.configuration.q, dv * current_dt
                )
                self.configuration = pink.Configuration(
                    self.robot.model, self.robot.data, q_out
                )
                pin.updateFramePlacements(self.robot.model, self.robot.data)
                error_norm = np.linalg.norm(
                    self.ee_task.compute_error(self.configuration)
                )
                nb_steps += 1
        except Exception as e:
            logger.error(f"IK failed: {e}")
            return torch.from_numpy(self.configuration.q).to(device)

        # print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
        return torch.from_numpy(self.configuration.q).to(device)

if __name__ == "__main__":
    robot = SingleArmIK(
        urdf_path="models/L5.urdf",
        package_dirs=["models"],
        use_in_lp_filter=False,
    )
    # import roboticstoolbox as rtb

    # rtb_robot = rtb.Robot.URDF(f"models/L5.urdf")
    full_robot = robot.full_robot

    for i in range(100):
        q0 = full_robot.q0 + np.random.randn(full_robot.model.nq) * 0.1
        full_robot.framesForwardKinematics(q0)
        # rtb_ee = rtb_robot.fkine(q0, end="link10").A
        pin_ee = np.eye(4)
        pin_ee[:3, 3] = full_robot.data.oMf[
            full_robot.model.getFrameId("link10")
        ].translation
        pin_ee[:3, :3] = full_robot.data.oMf[
            full_robot.model.getFrameId("link10")
        ].rotation
        assert np.allclose(rtb_ee, pin_ee, rtol=1e-3, atol=1e-3)

    print(1)