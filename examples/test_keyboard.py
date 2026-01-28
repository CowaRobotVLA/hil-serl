import sys
import tty
import termios
import select
import pycrmw
from msgs.crmw_pb2 import Service
class KeyboardListener:
    def __init__(self, parent_instance):
        self.parent = parent_instance

    def get_key(self):
        """获取单个按键，非阻塞模式"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            # 使用 select 检查是否有输入，超时时间 0.1s
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                key = sys.stdin.read(1)
                return key
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run_loop(self):
        print("监听中... (按 'e' 发送 exit_remote, 'r' 发送 remote, 's', 'f', 'n', 'q' 退出)")
        try:
            while True:
                key = self.get_key()
                if key == 'r':
                    print("\n发送: remote")
                    self.parent.RpcRequest(b'remote')
                elif key == 'e':
                    print("\n发送: exit_remote")
                    self.parent.RpcRequest(b'exit_remote')
                elif key == 's':
                    print("\n发送: success")
                    self.parent.RpcRequest(b'success')
                elif key == 'f':
                    print("\n发送: fail")
                    self.parent.RpcRequest(b'fail')
                elif key == 'n':
                    print("\n发送: next")
                    self.parent.RpcRequest(b'next')
                elif key == 'q':
                    print("\n退出监听")
                    break
        except KeyboardInterrupt:
            pass

# 假设这是你的主类
class YourApp:
    def RpcRequest(self,cmd):
        # cmd: b'support' or b'release_support' 
        rpc = pycrmw.ServiceFind("hil_serl_arg", Service, Service)
        # rpc = pycrmw.ServiceFind("WheelLegRpc", crmw_pb2.Service, crmw_pb2.Service)
        arg = Service()
        arg.arg.append(cmd)
        try:
            r = rpc(arg)
            print("send")
            return True
        except:
            print("fail to send")
            return False

if __name__ == "__main__":
    app = YourApp()
    listener = KeyboardListener(app)
    listener.run_loop()