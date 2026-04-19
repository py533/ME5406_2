import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import namedtuple
import math
import random

class UR5RobotiqEnv(gym.Env):
    def __init__(self, headless=False, reach_threshold_m=0.03):
        super(UR5RobotiqEnv, self).__init__()

        self.headless = headless
        self.reach_threshold_m = float(reach_threshold_m)
        # In headless mode: use DIRECT (no window), skip all sleeps
        self._sleep = (lambda x: None) if headless else time.sleep

        # Connect to PyBullet
        self.physics_client = p.connect(p.DIRECT if headless else p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Set the simulation time step to 1/300 for faster calculations
        p.setTimeStep(1 / 300)
        # Action space: [x, y] target position for the end-effector
        self.action_space = spaces.Box(low=np.array([0.3, -0.3]), high=np.array([0.7, 0.3]), dtype=np.float64)

        # Observation space: [cube_x, cube_y, eef_x, eef_y]
        # Including EEF position lets the agent know where its arm currently is,
        # which is essential for learning accurate Q-value estimates.
        self.observation_space = spaces.Box(
            low=np.array( [0.3, -0.3, 0.2, -0.4]),
            high=np.array([0.7,  0.3, 0.8,  0.4]),
            dtype=np.float64,
        )

        # Load environment objects
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        self.tray_id = p.loadURDF("tray/tray.urdf", [0.5, 0.9, 0.6], p.getQuaternionFromEuler([0, 0, 0]))
        self.cube_id2 = p.loadURDF("cube.urdf", [0.5, 0.9, 0.3], p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.6, useFixedBase=True)

        if not headless:
            # Load decorative objects (visual enrichment only, fixed inside red boundary box)
            self.decor_cylinder_id = p.loadURDF(
                "./urdf/cylinder_orange.urdf",
                [0.32, 0.27, 0.675],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True
            )
            self.decor_sphere_id = p.loadURDF(
                "./urdf/sphere_green.urdf",
                [0.68, -0.27, 0.657],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True
            )
            # Set GUI viewing angle
            self.set_gui_view()

        # Load the robot
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        if not headless:
            # Apply custom colors to robot links
            self._color_robot()

        # Initialize cube
        self.cube_id = None
        self._boundary_line_ids = []
        # Set the maximum number of steps
        self.max_steps = 100
        self.current_step = 0
        self.gripper_range = [0, 0.085]  # [fully closed, fully open]

        # change gripper friction
        for link_id in [12, 17]:
            p.changeDynamics(self.robot.id, link_id,
                     lateralFriction=1000.0,
                     spinningFriction=1.0,
                     frictionAnchor=1)

    def set_gui_view(self):
        """
        Set the GUI camera view (not actual camera capture)
        """
        camera_distance = 1.5      # Slightly further for wider view
        camera_yaw = 45            # Diagonal front-right perspective
        camera_pitch = -32         # Shallower tilt for better depth perception
        camera_target = [0.5, 0, 0.68]  # Raised slightly to show workspace better

        p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                     cameraYaw=camera_yaw,
                                     cameraPitch=camera_pitch,
                                     cameraTargetPosition=camera_target)

    def _color_robot(self):
        """
        Apply custom colors to UR5 arm and Robotiq85 gripper links.
        Arm: steel blue; Gripper fingers: silver-grey.
        """
        num_joints = p.getNumJoints(self.robot.id)
        arm_color   = [0.20, 0.38, 0.68, 1.0]   # Steel blue for arm body
        joint_color = [0.28, 0.28, 0.32, 1.0]   # Dark anthracite for joint housings
        gripper_color = [0.72, 0.72, 0.76, 1.0] # Silver-grey for gripper

        # Base link (index -1) → dark anthracite
        p.changeVisualShape(self.robot.id, -1, rgbaColor=joint_color)

        for i in range(num_joints):
            info = p.getJointInfo(self.robot.id, i)
            link_name = info[12].decode("utf-8")
            if any(kw in link_name for kw in ["finger", "knuckle", "inner", "pad", "tip"]):
                p.changeVisualShape(self.robot.id, i, rgbaColor=gripper_color)
            elif i < 6:
                # UR5 arm links alternate blue / anthracite for joint caps
                color = arm_color if i % 2 == 0 else joint_color
                p.changeVisualShape(self.robot.id, i, rgbaColor=color)
            else:
                p.changeVisualShape(self.robot.id, i, rgbaColor=gripper_color)

    def draw_boundary(self, x_range, y_range, z_height):
        for line_id in self._boundary_line_ids:
            p.removeUserDebugItem(line_id)
        self._boundary_line_ids.clear()

        corners = [
            [x_range[0], y_range[0], z_height],
            [x_range[1], y_range[0], z_height],
            [x_range[1], y_range[1], z_height],
            [x_range[0], y_range[1], z_height],
        ]
        for i in range(len(corners)):
            line_id = p.addUserDebugLine(corners[i], corners[(i + 1) % len(corners)], [1, 0, 0], lineWidth=2)
            self._boundary_line_ids.append(line_id)

    def _sample_decor_pos(self, occupied_xy_list, min_dist=0.12):
        """
        Sample a random (x, y) inside the red boundary box that keeps at least
        min_dist away from every position in occupied_xy_list.
        """
        for _ in range(200):  # max attempts before giving up
            x = np.random.uniform(0.31, 0.69)
            y = np.random.uniform(-0.28, 0.28)
            pos = np.array([x, y])
            if all(np.linalg.norm(pos - np.array(occ)) >= min_dist
                   for occ in occupied_xy_list):
                return x, y
        # Fallback: return the last sampled position if rejection keeps failing
        return x, y

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        """
        self.current_step = 0
        self.robot.original_position(self.robot)
        # Reset cube position
        x_range = np.arange(0.4, 0.7, 0.2)
        y_range = np.arange(-0.3, 0.3, 0.2)

        cube_start_pos = [
            np.random.choice(x_range),
            np.random.choice(y_range),
            0.63
        ]
        x_draw_range = [0.3, 0.7]
        y_draw_range = [-0.3, 0.3]
        # Draw the boundary box
        self.draw_boundary(x_draw_range, y_draw_range, 0.63)
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        if self.cube_id:
            p.resetBasePositionAndOrientation(self.cube_id, cube_start_pos, cube_start_orn)
        else:
            self.cube_id = p.loadURDF("./urdf/cube_blue.urdf", cube_start_pos, cube_start_orn)

        # Randomize decorative objects inside the red box, avoiding the target cube
        if not self.headless:
            cube_xy = cube_start_pos[:2]
            cyl_x, cyl_y = self._sample_decor_pos([cube_xy])
            sph_x, sph_y = self._sample_decor_pos([cube_xy, [cyl_x, cyl_y]])
            orn = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(
                self.decor_cylinder_id, [cyl_x, cyl_y, 0.675], orn)
            p.resetBasePositionAndOrientation(
                self.decor_sphere_id,   [sph_x, sph_y, 0.657], orn)

        # Store the initial position of the cube for comparison
        self.initial_cube_pos = np.array(cube_start_pos[:2])

        # Get initial cube position for observation
        self.target_pos = np.array(cube_start_pos[:2])

        # Query current EEF position after arm reset, so the agent immediately
        # knows where its arm is and prev_distance is properly initialised.
        eef_state   = self.robot.get_current_ee_position()
        eef_xy      = np.clip(np.array(eef_state[0])[:2],
                              [0.2, -0.4], [0.8, 0.4])
        self.prev_distance = float(np.linalg.norm(eef_xy - self.target_pos))

        observation = np.concatenate([self.target_pos, eef_xy])
        info = {}
        return observation, info

    def gripper_close(self):
        grip_value = self.gripper_range[1]  

        while True:
            contact_point = p.getContactPoints(bodyA=self.robot.id)

            force = {}
            if len(contact_point) > 0:
                for i in contact_point:
                    link_index = i[2]
                    if force.get(link_index) is None:
                        force[link_index] = {17: 0, 12: 0}
                    if i[3] == 17:
                        if i[9] > force[link_index][17]:
                            force[link_index][17] = i[9]
                    elif i[3] == 12:
                        if i[9] > force[link_index][12]:
                            force[link_index][12] = i[9]

            #  Stop immediately when force is detected
            for link_index in force:
                if force[link_index][17] > 3 and force[link_index][12] > 3:
                    if not self.headless:
                        print(f"[Grasped] Link {link_index}: joint 17 = {force[link_index][17]:.2f}, joint 12 = {force[link_index][12]:.2f}")
                    return True

            #  Print current force status (for debugging, GUI mode only)
            if not self.headless:
                for link_index in force:
                    for joint in [17, 12]:
                        if force[link_index][joint] > 0:
                            print(f"Link {link_index}, joint {joint} force: {force[link_index][joint]:.2f}")

            if grip_value <= self.gripper_range[0]:  # Already fully closed
                break

            grip_value -= 0.001
            self.robot.move_gripper(grip_value)

            for _ in range(60):
                p.stepSimulation()

        return False

    def step(self, action):
        """
        Perform an action in the environment.
        :param action: [x, y, z] target position for the end-effector
        """
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reached = False
        grasp_success = False
        cube_z = 0.0

        eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
        eef_position = eef_state[0]
        eef_orientation = eef_state[1]

        target_pos = np.array([action[0], action[1], 0.88]) 
        self.robot.move_arm_ik(target_pos, eef_orientation)
        for _ in range(100):
            p.stepSimulation()

        eef_state = self.robot.get_current_ee_position()
        eef_position = np.array(eef_state[0])[:2]

        distance_to_target = abs(np.linalg.norm(eef_position - self.target_pos))

        # ── potential-based improvement bonus ─────────────────────────────────
        # Reward the agent for reducing distance at each step.  This adds a
        # dense learning signal without changing the optimal policy (Ng 1999).
        improvement_bonus = 0.0
        if self.prev_distance is not None:
            improvement_bonus = 8.0 * max(0.0, self.prev_distance - distance_to_target)
        self.prev_distance = distance_to_target
        # ─────────────────────────────────────────────────────────────────────

        if distance_to_target <= self.reach_threshold_m:
            reached = True
            steps_taken = self.max_steps - self.current_step
            reward = 80
            reward += max(0, (steps_taken * 1))
            
            if not self.headless:
                print(f"Cube picked. {self.target_pos[0], self.target_pos[1]} picked successfully, distance {distance_to_target}, reward: {reward}")
        
            target_pos = np.array([action[0], action[1], 0.8])
            self.robot.move_arm_ik(target_pos, eef_orientation)
            for _ in range(100):
                p.stepSimulation()
                self._sleep(0.01)

            success = self.gripper_close()

            if success:
                if not self.headless:
                    print("Grasp successful!")
                self._sleep(0.5)
                self.lift_object_slowly(
                    start_pos=np.array([action[0], action[1], 0.8]),
                    end_z=1.0,
                    eef_orientation=eef_orientation
                )
                cube_z = p.getBasePositionAndOrientation(
                    self.cube_id, physicsClientId=self.physics_client
                )[0][2]
                grasp_success = cube_z > 0.80
                if grasp_success:
                    reward += 120
                else:
                    reward -= 20
                if not self.headless:
                    p.addUserDebugText(f"Success Pick", textColorRGB=[0, 0, 255], textPosition=[0.5, -1.1, 0.9],
                                    textSize=2, lifeTime=1)
            else:
                reward -= 40
                if not self.headless:
                    print("Grasp failed.")

            self._sleep(0.5)
            done = True
        elif self.current_step >= self.max_steps:
            reward = -10 * distance_to_target + improvement_bonus
            done = True
        else:
            reward = -10 * distance_to_target + improvement_bonus
            done = False
     
        if not self.headless:
            print(f"reward:{reward}\n")
            print(f"Distance difference: {distance_to_target}")

        # 4-D observation: [cube_x, cube_y, eef_x, eef_y]
        eef_xy_clipped = np.clip(eef_position, [0.2, -0.4], [0.8, 0.4])
        observation = np.concatenate([self.target_pos, eef_xy_clipped])
        truncated = False
        info = {
            "reached": reached,
            "grasp_success": grasp_success,
            "cube_z": cube_z,
        }
 
        return observation, reward, done, truncated, info

    def lift_object_slowly(self, start_pos, end_z, eef_orientation,
                            steps=30, sim_steps_per_move=5, sleep_time=0.005):
        """
        Faster smooth lifting
        """
        for i in range(steps):
            intermediate_z = start_pos[2] + (end_z - start_pos[2]) * (i + 1) / steps
            lift_pos = np.array([start_pos[0], start_pos[1], intermediate_z])
            self.robot.move_arm_ik(lift_pos, eef_orientation)

            for _ in range(sim_steps_per_move):
                p.stepSimulation()
                if sleep_time > 0:
                    self._sleep(sleep_time)

    def close(self):
        p.disconnect()


class UR5Robotiq85:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 10

    def load(self):
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()
        
    def __parse_joint_info__(self):
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            )

        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, open_length):
        """
        Control the gripper to open or close.
        :param open_length: Target width for gripper opening (0 ~ 0.085m)
        """
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=50, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id, self.eef_id, target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i], maxVelocity=self.max_velocity)

    def get_current_ee_position(self):
        return p.getLinkState(self.id, self.eef_id)

    def original_position(self, robot):
        # Set the initial posture for the robot arm to approach the cube
        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        for _ in range(100):
            p.stepSimulation()
        self.move_gripper(0.085)
        for _ in range(3500):
            p.stepSimulation()
