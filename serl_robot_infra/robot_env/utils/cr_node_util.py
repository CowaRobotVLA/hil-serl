from typing import Any, Optional
import time
import io
# import av
import threading
from collections import deque
import abc 
import sys
import numpy as np
print(sys.executable)
import numpy as np
from loguru import logger
import pycrmw
import crpilot
from queue import Queue
from msgs import record_pb2, pose_pb2, chassis_pb2, wheel_pb2, wheel_env_pb2, shared_msg_pb2, arm_eepos_pb2, stream_pb2
# from msgs import wheel_pb2
from queue import Empty
import struct

class ThreadSafeStack:
    def __init__(self, max_size: int = 1000):
        """
        线程安全的栈实现
        
        Args:
            max_size: 栈的最大容量
        """
        self._stack = deque()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._condition = threading.Condition(self._lock)
    
    def push(self, item: Any) -> bool:
        """
        入栈操作
        
        Args:
            item: 要入栈的元素
            
        Returns:
            bool: 成功返回True，栈满返回False
        """
        with self._condition:
            if len(self._stack) >= self._max_size:
                self._stack.popleft()
            self._stack.append(item)
            self._condition.notify_all()  # 通知等待的消费者
            return True
    
    def pop(self, timeout: Optional[float] = None) -> tuple[bool, Any]:
        """
        出栈操作
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            tuple[bool, Any]: (是否成功, 元素值)
        """
        with self._condition:
            # 等待直到栈不为空或超时
            while len(self._stack) == 0:
                if not self._condition.wait(timeout):
                    return False, None
            
            item = self._stack.pop()
            return True, item
    
    def peek(self) -> tuple[bool, Any]:
        """
        查看栈顶元素但不弹出
        
        Returns:
            tuple[bool, Any]: (是否成功, 元素值)
        """
        # with self._lock:
        with self._condition:
            if len(self._stack) == 0:
                return False, None
            return True, self._stack[-1]
    
    def size(self) -> int:
        """获取栈的大小"""
        with self._condition:
            return len(self._stack)
    
    def is_empty(self) -> bool:
        """检查栈是否为空"""
        with self._condition:
            return len(self._stack) == 0
    
    def is_full(self) -> bool:
        """检查栈是否已满"""
        with self._condition:
            return len(self._stack) >= self._max_size
    
    def clear(self) -> None:
        """清空栈"""
        with self._condition:
            self._stack.clear()
    
    def capacity(self) -> int:
        """获取栈的容量"""
        return self._max_size
    
    def remaining_capacity(self) -> int:
        """获取剩余容量"""
        with self._condition:
            return self._max_size - len(self._stack)

class DropOldestQueue(Queue):
    def put(self, item, block=True, timeout=None):
        """
        当队列满时，自动移除最早的元素，然后再插入新元素
        """
        with self.mutex:  # 获取内部锁
            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                # 丢弃最早的元素
                self._get()  # 从内部队列取出最早的
            self._put(item)    # 放入新元素
            self.unfinished_tasks += 1
            self.not_empty.notify()

class WayPoint:
    def __init__(self, position_xyz, rotation_quat, v, omega):
        self.position = position_xyz
        self.rotation_quat = rotation_quat
        self.v = v
        self.omega = omega

class BasicDecoder(abc.ABC):
    def __init__(self, stack: ThreadSafeStack, freq: int):
        self.stack = stack
        self.freq = freq
        self.cache = DropOldestQueue(maxsize=100)
        self.stop_flag = True # 在外界call时打开，开始解码
        self.last_push_time = 0.0
        self.min_interval = 1.0 / freq if freq > 0 else 0.0  # 计算最小时间间隔
        # 启动解码线程
        self.decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.decode_thread.start()
    
    @abc.abstractmethod
    def _decode_loop(self):
        """
        Abstract method: Subclasses must implement.
        """
        raise NotImplementedError

    def __call__(self, msg: Any):
        self.stop_flag = False
        self.cache.put_nowait(msg)

    def stop(self):
        self.stop_flag = True
        self.decode_thread.join()

class H264Decoder(BasicDecoder):
    """
    高效 H.264 解码器：
    - 独立解码线程
    - RGB 输出
    - 非阻塞控制帧率
    """
    def __init__(self, stack: ThreadSafeStack, freq: int):
        super().__init__(stack, freq)
        self.wait_first_keyframe = True

    def __call__(self, msg: record_pb2.ImageFrame):
        super().__call__(msg)

    def _decode_loop(self):
        container = None
        codec = None
        rawData = io.BytesIO()
        rawData_pos = 0
        while 1:
            if not self.stop_flag:
                try:
                    # 队列空则等待
                    packet_bytes = self.cache.get(timeout=0.01).data
                    rawData.write(packet_bytes)
                    rawData.seek(rawData_pos)

                    if rawData_pos == 0:
                        container = av.open(rawData, format='h264', mode='r')
                        video_stream = container.streams.video[0]
                        codec = av.codec.CodecContext.create(video_stream.name, 'r')
                    rawData_pos += len(packet_bytes)

                    for packet in container.demux():
                        if packet.size == 0:
                            continue
                        if self.wait_first_keyframe:
                            if packet.is_keyframe:
                                self.wait_first_keyframe = False
                                logger.success("检测到关键帧，开始解码")
                        
                        if not self.wait_first_keyframe:
                            frames = codec.decode(packet)
                            for frame in frames:
                                frame_array = frame.to_ndarray(format='bgr24')
                                # 控制频率
                                current_time = time.perf_counter()
                                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                                    self.stack.push(frame_array)
                                    self.last_push_time = current_time
                except Exception as e:
                    # print(e)
                    time.sleep(0.001)

class RawImageDecoder(BasicDecoder):
    def __call__(self, msg: shared_msg_pb2.Image2):
        super().__call__(msg)
        
    def _decode_loop(self):
        while 1:
            if not self.stop_flag:
                try:
                    msg = self.cache.get(timeout=0.01)
                except Empty:
                    continue
                data = msg.SerializeToString()
                img_obj = crpilot.Image2()
                img_obj.ParseFromString(data)
                image_raw = img_obj.image

                # 控制频率
                current_time = time.perf_counter()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push(image_raw)
                    self.last_push_time = current_time

class SamImageDecoder(BasicDecoder):
    def __call__(self, msg: wheel_env_pb2.SegmentImage):
        super().__call__(msg)
        
    def _decode_loop(self):
        while 1:
            if not self.stop_flag:
                img_msg = self.cache.get(timeout=1)
                if img_msg.encoding not in ("rgb8", "bgr8", "mask"):
                    raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
                image = np.frombuffer(img_msg.data, dtype=np.uint8)
                if img_msg.encoding in ("rgb8", "bgr8"):
                    # 重建 numpy 数组
                    image = image.reshape((img_msg.height, img_msg.width, 3))
                if img_msg.encoding == "mask":
                    image = image.reshape((img_msg.height, img_msg.width))
                if img_msg.encoding == "bgr8":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 控制频率
                current_time = time.perf_counter()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push(image)
                    self.last_push_time = current_time

class LidarDecoder(BasicDecoder):
    def __call__(self, msg: pycrmw.PointCloud3):
        super().__call__(msg)
        
    def _decode_loop(self):
        while 1:
            if not self.stop_flag:
                msg = self.cache.get(timeout=0.2)
                xyz = [(p['x'],p['y'],p['z']) for p in msg.point]
                
                # 控制频率
                current_time = time.perf_counter()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push(xyz)
                    self.last_push_time = current_time
            
class OdomDecoder(BasicDecoder):
    def __call__(self, msg: pose_pb2.PoseStamped):
        super().__call__(msg)

    def _decode_loop(self):
        while 1:
            if not self.stop_flag:
                msg = self.cache.get(timeout=0.01)  # 没消息会常见性超时
                waypoint = WayPoint((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z), msg.pose.rotation, msg.velocity.linear, msg.velocity.angular)
                # 控制频率
                current_time = time.perf_counter()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push(waypoint)
                    self.last_push_time = current_time
    
class BaseInfoDecoder(BasicDecoder):
    def __call__(self, msg: chassis_pb2.VehicleInfo):
        super().__call__(msg)

    def _decode_loop(self):
        while 1:
            if not self.stop_flag:
                msg = self.cache.get(timeout=0.01)
                drive = msg.drive
                speed = (drive.speed, drive.steer_speed)
                # 控制频率
                current_time = time.perf_counter()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push(speed)
                    self.last_push_time = current_time

class LeaderArmStateDecoder(BasicDecoder):
    def __call__(self, msg: wheel_pb2.RobotRLInfo):
        super().__call__(msg)

    def _decode_loop(self):
        while 1:
            # print(f"{self.stop_flag=}")
            if not self.stop_flag:
                try:
                    # print("cache length: ", self.cache.qsize())
                    msg = self.cache.get(timeout=0.01)  # 没消息会常见性超时
                except Empty:
                    continue  # 正常情况：这轮没消息，继续下一轮
                joints_pos = [999]*7 #6关节加1夹爪
                updated = False
                for idx, pos in zip(msg.joint_state.index, msg.joint_state.pos):
                    joints_pos[idx] = pos
                    # TODO: 增加常见error code的处理           

                    if 0 <= idx < 7:        # 防越界
                        joints_pos[idx] = pos
                        updated = True
                    
                if not updated:
                    continue  # 这条消息没有效数据，跳过

                # 控制频率
                current_time = time.time()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push({"joints_pos": joints_pos.copy(), "timestamp": time.time()})
                    self.last_push_time = current_time
            else:
                time.sleep(self.min_interval)

class ArmStateDecoder(BasicDecoder):
    def __call__(self, msg: wheel_pb2.RobotRLInfo):
        super().__call__(msg)

    def _decode_loop(self):
        while 1:
            # print(f"{self.stop_flag=}")
            if not self.stop_flag:
                try:
                    # print("cache length: ", self.cache.qsize())
                    msg = self.cache.get(timeout=0.01)  # 没消息会常见性超时
                except Empty:
                    continue  # 正常情况：这轮没消息，继续下一轮
                joints_pos = [999]*7 #6关节加1夹爪
                joints_vel = [999]*7
                joint_torque = [999]*7
                updated = False
                for idx, pos, speed, torque, error_code in zip(msg.joint_state.index, msg.joint_state.pos, msg.joint_state.speed, msg.joint_state.torque, msg.joint_state.error_code):
                    joints_pos[idx] = pos
                    joints_vel[idx] = speed
                    joint_torque[idx] = torque   
                    # TODO: 增加常见error code的处理           

                    if 0 <= idx < 7:        # 防越界
                        joints_pos[idx] = pos
                        joints_vel[idx] = speed
                        joint_torque[idx] = torque   
                        updated = True
                    
                if not updated:
                    continue  # 这条消息没有效数据，跳过

                # 控制频率
                current_time = time.time()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push({"joints_pos": joints_pos.copy(), "joints_vel": joints_vel.copy(), "joint_torque": joint_torque.copy(), "timestamp": time.time()})
                    self.last_push_time = current_time
            else:
                time.sleep(self.min_interval)

class ExpertStateDecoder(BasicDecoder):
    def __call__(self, msg: wheel_pb2.MultiMotorInfo):
        super().__call__(msg)

    def _decode_loop(self):
        while 1:
            # print(f"{self.stop_flag=}")
            if not self.stop_flag:
                try:
                    # print("cache length: ", self.cache.qsize())
                    msg = self.cache.get(timeout=0.01)  # 没消息会常见性超时
                except Empty:
                    continue  # 正常情况：这轮没消息，继续下一轮
                expert_state = None
                if msg.operation_mode == wheel_pb2.AUTO_MODE_REMOTE:
                    expert_state = True
                else:
                    expert_state = False
                # 控制频率
                current_time = time.time()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push({"expert_state": expert_state,  "timestamp": time.time()})
                    self.last_push_time = current_time
            else:
                time.sleep(self.min_interval)

class LeaderCommandDecoder(BasicDecoder):
    
    def __call__(self, msg: stream_pb2.Stream):
        super().__call__(msg)

    def deserialize_double_array(self,send_data):
        # 验证
        if len(send_data) % 8 != 0:
            raise ValueError("Invalid send data length for double array")
        if len(send_data) == 0:
            raise ValueError("gr control data length is 0")
        
        # 解析
        count = len(send_data) // 8
        cmds = struct.unpack(f'{count}d', send_data)
        
        # 创建16元素列表，填充前15个
        cmd_list = [0.0] * 16
        for i in range(min(count, 15)):
            cmd_list[i] = cmds[i]
        
        return cmd_list
    def _decode_loop(self):
        while 1:
            # print(f"{self.stop_flag=}")
            if not self.stop_flag:
                try:
                    # print("cache length: ", self.cache.qsize())
                    msg = self.cache.get(timeout=0.1)  # 没消息会常见性超时
                except Empty:
                    continue  # 正常情况：这轮没消息，继续下一轮
                action = np.array(self.deserialize_double_array(msg.buffer[0].send[0]))[:7]
                # 控制频率
                current_time = time.time()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push({"expert_action": action,  "timestamp": time.time()})
                    self.last_push_time = current_time
            else:
                time.sleep(self.min_interval)

class EeposeDecoder(BasicDecoder):
    def __call__(self, msg: arm_eepos_pb2.ArmeePos):
        super().__call__(msg)
    def quat_xyz_to_homogeneous(self, x, y, z, qx, qy, qz, qw):
        # 归一化（非常重要）
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T
    def _decode_loop(self):
        while 1:
            if not self.stop_flag:
                msg = self.cache.get(timeout=1)
                quat = msg.quat
                pose = [msg.x, msg.y, msg.z]
                rpy = [msg.roll,msg.pitch,msg.yaw]
                eepose = np.concatenate((pose, np.array((quat.x,quat.y,quat.z,quat.w))))
                T = self.quat_xyz_to_homogeneous(pose[0],pose[1],pose[2],quat.x,quat.y,quat.z,quat.w)
                current_time = time.time()
                if current_time - self.last_push_time >= self.min_interval or self.last_push_time==0.0:
                    self.stack.push({"ee_pose": eepose, "ee_position": pose, "ee_quat": quat,"ee_rpy": rpy, "ee_T": T, "timestamp": time.time()})
                    self.last_push_time = current_time

if __name__ == "__main__":
    import cv2
    def test_basic_stack():
        stack = ThreadSafeStack(max_size=5)
        
        # 测试入栈
        for i in range(6):
            success = stack.push(f"item_{i}")
            print(stack.size())
            print(f"Push item_{i}: {success}")
        
        # 测试出栈
        for i in range(3):
            success, item = stack.pop()
            print(f"Pop: {success}, item: {item}")
        
        # 测试栈状态
        print(f"Size: {stack.size()}")
        print(f"Is empty: {stack.is_empty()}")
    
    def test_image_decoder(node):
        img_stack = ThreadSafeStack(10)
        h264_decoder = H264Decoder(stack=img_stack, freq=30) #帧率要求大于20（发布频率）不然可能会导致漏过关键帧，导致出的图片模糊
        # camera_reader = node.CreateReader("/camera/surround/front/h264", h264_decoder)
        camera_reader = node.CreateReader("/camera/panorama/3/h264", h264_decoder)
        while 1:
            flag, img = img_stack.peek()
            if flag:
                cv2.imwrite("test_img_0.png", img)
                time.sleep(0.05)
    
    def test_arm_state_decoder(node):
        stack = ThreadSafeStack(10)
        decoder = ArmStateDecoder(stack=stack, freq=100)
        reader = node.CreateReader("/RL/base_info/arm", decoder)
        while 1:
            flag, arm_state = stack.peek()
            if flag:
                print(arm_state, stack.size())
            time.sleep(0.002)

    def test_dropQ():
        q = DropOldestQueue(maxsize=3)
        q.put(1)
        q.put(2)
        q.put(3)
        q.put(4)  # 会丢掉最早的1，队列变为 [2,3,4]
        print(list(q.queue))  # 直接访问内部deque查看结果

    def test_rawimage_decoder(node):
        img_stack = ThreadSafeStack(10)
        Raw_decoder = RawImageDecoder(stack=img_stack, freq=30)
        reader = node.CreateReader("/camera/panorama/3/image_raw", Raw_decoder)
        while 1:
            flag, img = img_stack.peek()
            if flag:
                cv2.imwrite("test_img_raw.png", img)
                time.sleep(0.05)

    def test_segimage_decoder(node):
        img_stack = ThreadSafeStack(10)
        Sam_decoder = SamImageDecoder(stack=img_stack, freq=30)
        reader = node.CreateReader("/segment/image", Sam_decoder)
        while 1:
            flag, img = img_stack.peek()
            if flag:
                cv2.imwrite("test_img_raw.png", img)
                time.sleep(0.05)
    def test_eeposo_decoder(node):
        eepose_stack = ThreadSafeStack(10)
        eepose_decoder = EeposeDecoder(stack=eepose_stack, freq=500)
        reader = node.CreateReader("/RL/base_info/arm_eepos", eepose_decoder)
        while 1:
            flag, ee_info = eepose_stack.peek()
            if flag:
                a = ee_info['ee_pose']
                print(a)

    def test_leaderarm_decoder(node):
        eepose_stack = ThreadSafeStack(10)
        eepose_decoder = LeaderArmStateDecoder(stack=eepose_stack, freq=100)
        reader = node.CreateReader("/RL/base_info/leader_arm", eepose_decoder)
        while 1:
            flag, ee_info = eepose_stack.peek()
            if flag:
                a = ee_info['joints_pos']
                print(a)

    def test_expert_state(node):
        expert_stack = ThreadSafeStack(10)
        expert_decoder = LeaderCommandDecoder(stack=expert_stack, freq=10)
        reader = node.CreateReader("/gr/control", expert_decoder)
        while 1:
            flag, expert_state = expert_stack.peek()
            if flag:
                a = expert_state['expert_action']
                time.sleep(0.05)


    import sys,os
    pycrmw.Init(sys.argv)
    if not pycrmw.IsOK():
        os._exit(0)
    node = pycrmw.Node("test")

    # test_basic_stack()
    try:
        # test_arm_state_decoder(node)
        # test_expert_state(node)
        test_leaderarm_decoder(node)
        # test_dropQ()
    except KeyboardInterrupt:
        pycrmw.AsyncShutdown() 
