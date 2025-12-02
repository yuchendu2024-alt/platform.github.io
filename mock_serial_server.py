import socket
import threading
import time
import argparse

class MockSerialServer:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.client_sockets = []
        print(f"模拟串口服务器初始化，将在 {host}:{port} 监听")
    
    def start(self):
        # 创建TCP服务器套接字
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # 绑定地址和端口
            self.server_socket.bind((self.host, self.port))
            # 开始监听连接
            self.server_socket.listen(5)
            print(f"模拟串口服务器已启动，监听端口 {self.port}...")
            
            self.running = True
            # 接受连接的主循环
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print(f"接收到来自 {client_address} 的连接")
                    
                    # 将新的客户端连接添加到列表中
                    self.client_sockets.append(client_socket)
                    
                    # 为每个客户端创建一个新线程处理
                    client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
                    client_thread.daemon = True
                    client_thread.start()
                except socket.error as e:
                    if not self.running:
                        break
                    print(f"接受连接时出错: {e}")
        except Exception as e:
            print(f"服务器启动失败: {e}")
            self.stop()
    
    def handle_client(self, client_socket, client_address):
        buffer = ""
        try:
            while self.running:
                # 接收数据
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                # 处理完整的命令行
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        print(f"[{client_address}] 接收: {line}")
                        # 处理命令
                        response = self.process_command(line)
                        # 发送响应
                        if response:
                            client_socket.sendall((response + '\r\n').encode('utf-8'))
                            print(f"[{client_address}] 发送: {response}")
        except Exception as e:
            print(f"处理客户端 {client_address} 时出错: {e}")
        finally:
            # 关闭客户端连接
            client_socket.close()
            if client_socket in self.client_sockets:
                self.client_sockets.remove(client_socket)
            print(f"客户端 {client_address} 已断开连接")
    
    def process_command(self, command):
        """处理接收到的命令并返回响应"""
        # 转换为大写以便不区分大小写
        command = command.upper()
        
        # 简单命令响应 - 直接输出信号
        command_responses = {
            "MOVE": "OK: 移动定位信号已发送",
            "ROTATE": "OK: 零件转动信号已发送",
            "FLIP": "OK: 翻转气缸信号已发送",
            "MOTOR_START": "OK: 电机启动信号已发送",
            "MOTOR_RESET": "OK: 电机复位信号已发送",
            "CYLINDER_1": "OK: 一号气缸动作信号已发送",
            "CYLINDER_2": "OK: 二号气缸动作信号已发送",
            "CYLINDER_3": "OK: 三号气缸动作信号已发送"
        }
        
        # 检查是否是简单命令
        if command in command_responses:
            # 对于某些命令，模拟执行时间
            if command in ["MOVE", "ROTATE", "FLIP"]:
                time.sleep(0.3)
            return command_responses[command]
        
        # 保留原有的命令格式支持，以兼容旧版本
        # 模拟步进电机命令
        if command.startswith("MOTOR"):
            parts = command.split()
            if len(parts) >= 2:
                action = parts[1]
                
                if action == "FORWARD":
                    steps = parts[2] if len(parts) >= 3 else "100"
                    speed = parts[3] if len(parts) >= 4 else "500"
                    return f"OK: 步进电机向前运动 {steps} 步，速度 {speed}"
                
                elif action == "BACKWARD":
                    steps = parts[2] if len(parts) >= 3 else "100"
                    speed = parts[3] if len(parts) >= 4 else "500"
                    return f"OK: 步进电机向后运动 {steps} 步，速度 {speed}"
                
                elif action == "RESET":
                    return "OK: 步进电机复位完成"
        
        # 模拟翻转命令
        elif command.startswith("FLIP"):
            parts = command.split()
            if len(parts) >= 2:
                direction = parts[1]
                if direction in ["LEFT", "RIGHT"]:
                    return f"OK: 翻转操作执行 {direction}"
                else:
                    return "ERROR: 无效的翻转方向"
        
        # 模拟气缸命令
        elif command.startswith("CYLINDER"):
            parts = command.split()
            if len(parts) >= 2:
                action = parts[1]
                if action == "EXTEND":
                    # 模拟气缸伸出需要一些时间
                    time.sleep(0.5)
                    return "OK: 气缸已伸出"
                elif action == "RETRACT":
                    return "OK: 气缸已缩回"
        
        # 未知命令
        return f"OK: 命令 '{command}' 已接收并执行"
    
    def stop(self):
        print("正在停止模拟串口服务器...")
        self.running = False
        
        # 关闭所有客户端连接
        for client_socket in self.client_sockets:
            try:
                client_socket.close()
            except:
                pass
        self.client_sockets.clear()
        
        # 关闭服务器套接字
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("模拟串口服务器已停止")

# 使用说明
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模拟串口服务器，用于测试机械控制界面')
    parser.add_argument('--host', type=str, default='localhost', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8888, help='服务器监听端口')
    args = parser.parse_args()
    
    # 创建并启动服务器
    server = MockSerialServer(host=args.host, port=args.port)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止服务器...")
    finally:
        server.stop()