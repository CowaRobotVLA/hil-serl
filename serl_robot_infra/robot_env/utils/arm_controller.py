import os, sys
# print(sys.path)
# for i, p in enumerate(sys.path):
#     if ".local" in p:
#         sys.path.pop(i)
import time
import numpy as np
import threading
from loguru import logger
import pycrmw
from msgs import wheel_pb2
# from .cr_node_util import ThreadSafeStack, ArmStateDecoder
# from .IK_solver_pybullet import robot_solver
from .cr_node_util import ThreadSafeStack, ArmStateDecoder, RawImageDecoder
# from .IK_solver_pybullet import robot_solver
from .robot_solver_pybullet import RobotSolver


def sign(value):
    if value < 0:
        return -1
    else:
        return 1

def fp(l):
    if l is None:
        return l
    str_l = [format(v, '.3f') for v in l]
    return '['+',\t'.join(str_l)+']'

class ArmController:
    def __init__(self, node, dof_num:int, control_mode:str="vel", receive_interval=0.002, publish_interval=0.01, timeout_interval=0.2):
        self.sequence = 0
        self.dof_num = dof_num
        self.target_joints = [0]*dof_num
        self.control_mode = control_mode
        self.real_joints_pos = None
        self.real_joints_speed = None
        self.ee_pose = None
        self.rl_action = None
        self.last_receive_time = 0
        self.last_action = None
        self.receive_interval = receive_interval # 500Hz
        self.publish_interval = publish_interval # 100Hz
        self.timeout_interval = timeout_interval
        self.in_control = False # 通过set target开启
        self.arm_image = None
        self.arm_image_lock = threading.Lock()
        self.arm_state_lock = threading.Lock()
        self.target_lock = threading.Lock() 
        self.arm_state_stack = ThreadSafeStack(max_size=1)
        arm_state_decoder = ArmStateDecoder(stack=self.arm_state_stack, freq=100)
        self.ee_stack = ThreadSafeStack(max_size=1)
        ee_decoder = ArmStateDecoder(stack = self.ee_stack, freq=100)
        self.arm_state_reader = node.CreateReader("/RL/base_info/arm", arm_state_decoder)
        self.img_stack = ThreadSafeStack(5)
        img_decoder = RawImageDecoder(stack=self.img_stack, freq=30) #帧率要求大于20（发布频率）不然可能会导致漏过关键帧，导致出的图片模糊
        self.camera_reader = node.CreateReader("/camera/panorama/3/image_raw", img_decoder)
        self.eepoes_reader = node.CreateReader("/RL/base_info/arm_eepos", ee_decoder)
        self.action_writer = node.CreateWriter("/RL/action/arm", wheel_pb2.RLAction)
        self.robot_solver = RobotSolver(urdf_path='assets/cowarm/urdf/cowa_4rad_w_arm_6dof.urdf')
        self.update_arm_state_thread = threading.Thread(target=self._update_arm_state, daemon=True)
        self.update_arm_state_thread.start()
        self.update_arm_image_thread = threading.Thread(target=self._update_arm_image, daemon=True)
        self.update_arm_image_thread.start()
        self.pub_target_thread = threading.Thread(target=self._publish_target, daemon=True)
        self.pub_target_thread.start()

    def get_arm_state(self):
        with self.arm_state_lock:
            joint_pos = self.real_joints_pos
            joint_vel = self.real_joints_speed
            return joint_pos, joint_vel
        
    def get_eepos_state(self):
        with self.arm_state_lock:
            ee_pose = self.ee_pose
            return ee_pose
        
    def get_arm_image(self):
        with self.arm_image_lock:
            return self.arm_image

    def set_arm_state(self, arm_state):
        with self.arm_state_lock:
            self.real_joints_pos = np.array(arm_state['joints_pos'])
            self.real_joints_speed = np.array(arm_state['joints_vel'])
            self.last_receive_time = arm_state['timestamp']

    def set_ee_state(self, ee):
        with self.arm_state_lock:
            self.ee_pose = np.array(ee['eepose'])

    def set_arm_image(self, arm_image):
        with self.arm_image_lock:
            self.arm_image = arm_image

    def _update_arm_state(self):
        while True:
            flag, arm_state = self.arm_state_stack.pop()
            while not flag:
                time.sleep(0.01)
                flag, arm_state = self.arm_state_stack.pop()
            self.set_arm_state(arm_state)
            time.sleep(1/30)

    def _update_ee_state(self):
        while True:
            flag, ee = self.ee_stack.pop()
            while not flag:
                time.sleep(0.01)
                flag, ee = self.ee_stack.pop()
            self.set_ee_state(ee)
            time.sleep(1/30)

    def _update_arm_image(self):
        while True:
            flag, arm_image = self.img_stack.peek()
            while not flag:
                time.sleep(1/30)
                flag, arm_image = self.img_stack.peek()
            self.set_arm_image(arm_image)
            time.sleep(self.receive_interval)

    def get_q_by_ee_pos(self, xyz, quat, gripper, current_q=None):
        ''' xyz + quat + gripper '''
        q = self.robot_solver.inverse_kinematics(xyz, quat, rest_poses=current_q)
        return np.concatenate(([gripper], q))

    def get_ee_pos_by_q(self, q):
        " return xyz and quat relative to robot base link"
        xyz, quat, _ = self.robot_solver.forward_kinematics(q)
        return xyz, quat
        
    def filter_action(self, action: np.ndarray, epsilon = 0.1):
        f_action = epsilon * self.last_action + (1-epsilon) * action if self.last_action is not None else action
        f_action[0] = action[0]
        self.last_action = f_action
        return f_action.tolist()

    
    def build_msg_by_pos(self):
        with self.target_lock:
            target = self.target_joints.copy()
        gripper_cmd = np.clip(target[0],5,95)
        # gripper_cmd = 0.7 if target[0] < 50 else -0.7
        pos_cmd = np.concatenate([[gripper_cmd], target[1:]])
        rl_action = wheel_pb2.RLAction()
        if self.control_mode == "vel":
            rl_action.control_mode = wheel_pb2.VELOCITY_MODE
        elif self.control_mode == "torque":
            rl_action.control_mode = wheel_pb2.TORQUE_MODE
        rl_action.joint_num = self.dof_num
        rl_action.timestamp = int(time.time()*1e9)
        rl_action.action.cmd.extend(pos_cmd)
        rl_action.sequence = self.sequence
        self.rl_action = rl_action
        self.sequence += 1
        return rl_action

    def set_target(self, q):
        with self.target_lock:
            self.in_control = True
            self.target_joints = q

    def can_control(self):
        with self.arm_state_lock:
            # print("time:", abs(time.time() - self.last_receive_time))
            if abs(time.time() - self.last_receive_time) > self.timeout_interval:
                self.real_joints_pos = None
                self.real_joints_speed = None
                return True
            else:
                return False

    def _publish_target(self):
        while True:
            if self.in_control:
                timeout_flag = self.can_control()
                # with self.target_lock:
                #     print("arm controller pub target: ", self.target_joints)
                msg = self.build_msg_by_pos()
                self.action_writer.Write(msg)
                # logger.info('--------------------------------------------')
                # logger.info(f'real: {fp(self.real_joints_pos)}')
                # logger.info(f'publ: {fp(self.target_joints)}')
            time.sleep(self.publish_interval)




if __name__ == "__main__":
    start_pos = np.array([100.0, -0.073, -0.834, 2.177, -0.11, -0.22, 0.06])
    pycrmw.Init(sys.argv)
    if not pycrmw.IsOK():
        os._exit(0)
    node  = pycrmw.Node("ArmController")
    arm_controller = ArmController(node, 7)
    print("arm controller start")
    arm_controller.set_target(start_pos.copy())
    while 1:
        flag, arm_state = arm_controller.arm_state_stack.peek()
        if flag:
            print(arm_state, arm_controller.arm_state_stack.size())
        time.sleep(0.002)
