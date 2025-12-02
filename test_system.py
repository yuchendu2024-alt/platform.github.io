import subprocess
import time
import os
import sys

# 启动模拟串口服务器
def start_mock_server():
    print("正在启动模拟串口服务器...")
    # 使用新进程启动模拟服务器
    server_process = subprocess.Popen([sys.executable, "mock_serial_server.py"])
    print("模拟串口服务器已启动，PID:", server_process.pid)
    return server_process

# 检查系统依赖
def check_dependencies():
    print("正在检查系统依赖...")
    try:
        # 尝试导入主要依赖
        import PyQt5
        import cv2
        import numpy
        import serial
        import pyzbar
        print("所有依赖都已安装")
        return True
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行 'pip install -r requirements.txt' 安装所需依赖")
        return False

# 检查参考图像文件
def check_reference_images():
    print("正在检查参考图像...")
    required_images = ["1.jpg", "2.jpg", "3.jpg"]
    missing_images = []
    
    for img in required_images:
        if not os.path.exists(img):
            missing_images.append(img)
    
    if missing_images:
        print(f"缺少以下参考图像: {', '.join(missing_images)}")
        print("警告: 没有参考图像将导致图案识别功能无法正常工作")
    else:
        print("所有参考图像都存在")
    
    return len(missing_images) == 0

# 主测试函数
def main():
    print("===== 机械视觉控制系统测试 =====")
    
    # 检查系统
    dependencies_ok = check_dependencies()
    check_reference_images()
    
    if not dependencies_ok:
        print("测试失败: 缺少必要的依赖")
        return
    
    # 启动模拟服务器
    server_process = start_mock_server()
    
    try:
        # 等待服务器启动
        time.sleep(2)
        print("\n测试准备就绪！")
        print("1. 模拟串口服务器已在 localhost:8888 运行")
        print("2. 要测试系统，请运行:")
        print("   python control_interface.py")
        print("3. 在界面中，选择 'localhost:8888' 作为串口设备进行连接")
        print("\n按 Ctrl+C 停止模拟服务器...")
        
        # 保持运行直到用户中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止测试...")
    finally:
        # 终止服务器进程
        if server_process:
            print("停止模拟串口服务器...")
            server_process.terminate()
            server_process.wait()
    
    print("测试完成！")

if __name__ == "__main__":
    main()