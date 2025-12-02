import sys
import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
                            QTabWidget, QGridLayout, QTextEdit, QSplitter)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pyzbar.pyzbar import decode

# 导入现有的视觉识别功能相关代码
tipIds = [4, 8, 12, 16, 20]
total = [0, 0, 0, 0, 0]
lower_red1 = np.array([0, 50, 50])
lower_red2 = np.array([170, 50, 50])
lower_blue = np.array([100, 50, 50])
higher_red1 = np.array([10, 255, 255])
higher_red2 = np.array([180, 255, 255])
higher_blue = np.array([125, 255, 255])
kernel = np.ones((5, 5), np.float32)

def red_detect(contours_red):
    # 遍历所有红色轮廓
    for contours_reds in contours_red:
        # 计算轮廓的面积
        area = cv2.contourArea(contours_reds)
        # 如果面积大于1800，则认为检测到红色区域
        if (area > 1800):
            return True
    return False

def blue_detect(contours_blue):
    # 遍历所有蓝色轮廓
    for contours_blues in contours_blue:
        # 计算轮廓的面积
        area = cv2.contourArea(contours_blues)
        # 如果面积大于1800，则认为检测到蓝色区域
        if (area > 1800):
            return True
    return False

# 摄像头线程类，用于实时处理视频流
class CameraThread(QThread):
    image_update = pyqtSignal(QImage)
    detection_result = pyqtSignal(str, str, str)  # 发送颜色、图案、角度信息
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.capture = None
        self.error = [0]
        self.reference_images = []
        self.load_reference_images()
        
    def load_reference_images(self):
        try:
            img1 = cv2.imread("1.jpg")
            img2 = cv2.imread("2.jpg")
            img3 = cv2.imread("3.jpg")
            if img1 is not None:
                self.reference_images.append(img1)
            if img2 is not None:
                self.reference_images.append(img2)
            if img3 is not None:
                self.reference_images.append(img3)
        except Exception as e:
            print(f"加载参考图像失败: {e}")
    
    def run(self):
        self.running = True
        self.capture = cv2.VideoCapture(0)
        
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            # 进行视觉识别处理
            processed_frame, color, angle = self.process_image(frame)
            
            # 发送处理后的图像
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_update.emit(qt_image)
            
            # 发送识别结果
            pattern = "已识别" if angle else "未识别"
            self.detection_result.emit(color, pattern, angle)
            
            # 控制循环频率
            time.sleep(0.02)
        
        if self.capture:
            self.capture.release()
    
    def stop(self):
        self.running = False
        self.wait()
    
    def process_image(self, frame):
        # 基于final.py的视觉识别功能
        color = "未知"
        angle = ""
        current_img = None
        
        # 尝试与参考图像匹配
        for img in self.reference_images:
            result, detected_color, detected_angle = self.ORBMatcher(img, frame.copy())
            if result:
                current_img = img
                color = detected_color
                angle = detected_angle
                break
        
        return frame, color, angle
    
    def ORBMatcher(self, img1, img2):
        img_blur = cv2.GaussianBlur(img2, (9, 9), 2)
        img_mask = np.zeros((img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
        img_mask[150:390, 180:425] = 1
        img_camera1 = cv2.multiply(img2, img_mask)
        
        # 将图像转换为HSV颜色空间
        frame_hsv = cv2.cvtColor(img_camera1, cv2.COLOR_BGR2HSV)
        
        # 创建红色和蓝色的掩码
        mask_red1 = cv2.inRange(frame_hsv, lower_red1, higher_red1)
        mask_red2 = cv2.inRange(frame_hsv, lower_red2, higher_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(frame_hsv, lower_blue, higher_blue)
        
        # 查找红色和蓝色的轮廓
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 调用红色和蓝色检测函数
        result_red = red_detect(contours_red)
        result_blue = blue_detect(contours_blue)
        
        detected_color = "未知"
        if result_red:
            detected_color = "红色"
        elif result_blue:
            detected_color = "蓝色"
        
        # 进行边缘检测
        edges = cv2.Canny(img_blur, 50, 150)
        
        # 使用霍夫圆变换检测圆
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 40, param1=20, param2=30, minRadius=100, maxRadius=115)
        if circles is not None:
            # 将检测到的圆的坐标和半径转换为整数
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # 获取圆心和半径
                center = (i[0], i[1])
                radius = i[2]
                # 计算半径误差
                error1 = f"{abs(((radius - 113.37) / 113.37) * 22.4):.2f}"
                self.error.append(error1)
                # 如果检测到红色区域，绘制红色圆
                if (result_red):
                    cv2.circle(img2, center, radius, (0, 0, 255), 3)
                # 如果检测到蓝色区域，绘制蓝色圆
                if (result_blue):
                    cv2.circle(img2, center, radius, (255, 0, 0), 3)
        
        # 创建ORB特征检测器
        ORB = cv2.ORB_create()
        # 检测并计算图像1和图像2的关键点和描述符
        keypoints1, des1 = ORB.detectAndCompute(img1, None)
        keypoints2, des2 = ORB.detectAndCompute(img2, None)
        
        # 定义FLANN匹配器的参数
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_prove_level=1)
        search_params = dict(checks=50)
        
        # 创建FLANN匹配器
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 检查描述符是否有效
        if des1 is None or des2 is None:
            return False, detected_color, ""
        
        # 进行特征匹配
        try:
            matches = flann.knnMatch(des1, des2, k=2)
            # 筛选出好的匹配点
            points_good = []
            for match in matches:
                if (len(match)) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        points_good.append(m)
            
            # 定义最小匹配点数量
            MIN_MATCH_COUNT = 30
            if len(points_good) > MIN_MATCH_COUNT:
                # 获取匹配点的坐标
                src_array = np.float32([keypoints1[m.queryIdx].pt for m in points_good]).reshape(-1, 1, 2)
                dst_array = np.float32([keypoints2[m.trainIdx].pt for m in points_good]).reshape(-1, 1, 2)
                # 计算单应性矩阵
                M, mask = cv2.findHomography(src_array, dst_array, cv2.RANSAC, 5.0)
                # 计算旋转角度
                theta = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
                angle_str = str(int(theta))
                
                # 在图像上显示颜色和角度信息
                if (result_red):
                    cv2.putText(img2, f"color:red", (400, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                if (result_blue):
                    cv2.putText(img2, f"color:blue", (400, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                cv2.putText(img2, f"angle:{int(theta)}", (210, 70), cv2.FONT_HERSHEY_PLAIN, 3, (128, 0, 128), 3)
                
                return True, detected_color, angle_str
        except Exception as e:
            print(f"特征匹配出错: {e}")
        
        return False, detected_color, ""

# 串口控制类
class SerialController:
    def __init__(self):
        self.ser = None
        self.is_connected = False
    
    def get_available_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def connect(self, port, baudrate=9600):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.is_connected = True
            return True
        except Exception as e:
            print(f"串口连接失败: {e}")
            return False
    
    def disconnect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.is_connected = False
            return True
        except Exception as e:
            print(f"串口断开失败: {e}")
            return False
    
    def send_command(self, command):
        try:
            if self.ser and self.ser.is_open:
                self.ser.write((command + '\r\n').encode())
                time.sleep(0.1)
                response = self.ser.read_all().decode()
                return response
            return "未连接"
        except Exception as e:
            print(f"发送命令失败: {e}")
            return f"错误: {str(e)}"
    
    # 步进电机控制命令
    def motor_forward(self, steps=100, speed=500):
        command = f"MOTOR FORWARD {steps} {speed}"
        return self.send_command(command)
    
    def motor_backward(self, steps=100, speed=500):
        command = f"MOTOR BACKWARD {steps} {speed}"
        return self.send_command(command)
    
    def motor_reset(self):
        command = "MOTOR RESET"
        return self.send_command(command)
    
    # 翻转控制命令
    def flip(self, direction="LEFT"):
        command = f"FLIP {direction}"
        return self.send_command(command)
    
    # 气缸控制命令
    def cylinder_extend(self):
        command = "CYLINDER EXTEND"
        return self.send_command(command)
    
    def cylinder_retract(self):
        command = "CYLINDER RETRACT"
        return self.send_command(command)

# 主窗口类
class ControlInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机械视觉控制系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化串口控制器和摄像头线程
        self.serial_controller = SerialController()
        self.camera_thread = CameraThread()
        
        # 初始化识别结果变量
        self.detected_color = "未知"
        self.detected_angle = ""
        
        # 创建主布局
        self.init_ui()
        
        # 连接信号和槽
        self.connect_signals()
        
        # 刷新可用串口列表
        self.refresh_serial_ports()
    
    def init_ui(self):
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 创建视觉识别区域
        vision_group = self.create_vision_section()
        splitter.addWidget(vision_group)
        
        # 创建控制区域
        control_group = self.create_control_section()
        splitter.addWidget(control_group)
        
        # 设置分割器比例
        splitter.setSizes([500, 300])
        
        main_layout.addWidget(splitter)
    
    def create_vision_section(self):
        group = QGroupBox("视觉识别")
        layout = QHBoxLayout()
        
        # 创建图像显示标签
        self.image_label = QLabel("等待摄像头启动...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label)
        
        # 创建识别结果面板
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        
        # 识别结果标签
        self.color_label = QLabel("颜色: 未知")
        self.pattern_label = QLabel("图案: 未识别")
        self.angle_label = QLabel("角度: ---")
        
        # 设置标签样式
        font = self.color_label.font()
        font.setPointSize(12)
        self.color_label.setFont(font)
        self.pattern_label.setFont(font)
        self.angle_label.setFont(font)
        
        # 添加到结果布局
        result_layout.addWidget(self.color_label)
        result_layout.addWidget(self.pattern_label)
        result_layout.addWidget(self.angle_label)
        result_layout.addStretch()
        
        # 启动/停止摄像头按钮
        self.camera_button = QPushButton("启动摄像头")
        result_layout.addWidget(self.camera_button)
        
        layout.addWidget(result_widget)
        group.setLayout(layout)
        return group
    
    def create_control_section(self):
        # 创建标签页控件
        tab_widget = QTabWidget()
        
        # 添加串口控制标签页
        tab_widget.addTab(self.create_serial_tab(), "串口设置")
        
        # 添加步进电机控制标签页
        tab_widget.addTab(self.create_motor_tab(), "步进电机控制")
        
        # 添加翻转控制标签页
        tab_widget.addTab(self.create_flip_tab(), "翻转控制")
        
        # 添加气缸控制标签页
        tab_widget.addTab(self.create_cylinder_tab(), "气缸控制")
        
        # 添加日志标签页
        tab_widget.addTab(self.create_log_tab(), "系统日志")
        
        return tab_widget
    
    def create_serial_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # 串口选择
        layout.addWidget(QLabel("串口:"), 0, 0)
        self.port_combo = QComboBox()
        layout.addWidget(self.port_combo, 0, 1)
        
        # 波特率选择
        layout.addWidget(QLabel("波特率:"), 0, 2)
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "115200", "38400", "57600"])
        self.baudrate_combo.setCurrentText("9600")
        layout.addWidget(self.baudrate_combo, 0, 3)
        
        # 刷新按钮
        refresh_button = QPushButton("刷新串口")
        refresh_button.clicked.connect(self.refresh_serial_ports)
        layout.addWidget(refresh_button, 0, 4)
        
        # 连接/断开按钮
        self.connect_button = QPushButton("连接串口")
        self.connect_button.clicked.connect(self.toggle_serial_connection)
        layout.addWidget(self.connect_button, 1, 0, 1, 5)
        
        # 串口状态标签
        self.serial_status_label = QLabel("状态: 未连接")
        layout.addWidget(self.serial_status_label, 2, 0, 1, 5)
        
        return widget
    
    def create_motor_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # 步进数量设置
        layout.addWidget(QLabel("步进数量:"), 0, 0)
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(1, 10000)
        self.steps_spinbox.setValue(100)
        layout.addWidget(self.steps_spinbox, 0, 1)
        
        # 速度设置
        layout.addWidget(QLabel("速度:"), 0, 2)
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(100, 2000)
        self.speed_spinbox.setValue(500)
        layout.addWidget(self.speed_spinbox, 0, 3)
        
        # 控制按钮
        forward_button = QPushButton("向前运动")
        forward_button.clicked.connect(self.motor_forward)
        layout.addWidget(forward_button, 1, 0, 1, 2)
        
        backward_button = QPushButton("向后运动")
        backward_button.clicked.connect(self.motor_backward)
        layout.addWidget(backward_button, 1, 2, 1, 2)
        
        reset_button = QPushButton("复位")
        reset_button.clicked.connect(self.motor_reset)
        layout.addWidget(reset_button, 2, 0, 1, 4)
        
        return widget
    
    def create_flip_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 自动翻转复选框
        self.auto_flip_checkbox = QPushButton("启用自动翻转")
        self.auto_flip_checkbox.setCheckable(True)
        layout.addWidget(self.auto_flip_checkbox)
        
        # 手动翻转按钮
        flip_layout = QHBoxLayout()
        left_flip_button = QPushButton("向左翻转")
        left_flip_button.clicked.connect(lambda: self.flip("LEFT"))
        flip_layout.addWidget(left_flip_button)
        
        right_flip_button = QPushButton("向右翻转")
        right_flip_button.clicked.connect(lambda: self.flip("RIGHT"))
        flip_layout.addWidget(right_flip_button)
        
        layout.addLayout(flip_layout)
        
        # 说明文本
        info_label = QLabel("说明: 启用自动翻转后，系统会根据检测到的颜色自动控制翻转方向。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        return widget
    
    def create_cylinder_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 气缸控制按钮
        extend_button = QPushButton("气缸伸出")
        extend_button.clicked.connect(self.cylinder_extend)
        layout.addWidget(extend_button)
        
        retract_button = QPushButton("气缸缩回")
        retract_button.clicked.connect(self.cylinder_retract)
        layout.addWidget(retract_button)
        
        # 测试按钮（伸出后缩回）
        test_button = QPushButton("测试动作")
        test_button.clicked.connect(self.cylinder_test)
        layout.addWidget(test_button)
        
        return widget
    
    def create_log_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 日志文本框
        self.log_textedit = QTextEdit()
        self.log_textedit.setReadOnly(True)
        layout.addWidget(self.log_textedit)
        
        # 清空日志按钮
        clear_log_button = QPushButton("清空日志")
        clear_log_button.clicked.connect(self.clear_log)
        layout.addWidget(clear_log_button)
        
        return widget
    
    def connect_signals(self):
        # 摄像头线程信号连接
        self.camera_thread.image_update.connect(self.update_image)
        self.camera_thread.detection_result.connect(self.update_detection_result)
        
        # 摄像头控制按钮
        self.camera_button.clicked.connect(self.toggle_camera)
    
    def refresh_serial_ports(self):
        ports = self.serial_controller.get_available_ports()
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        self.log_message(f"已刷新串口列表，找到 {len(ports)} 个可用串口")
    
    def toggle_serial_connection(self):
        if self.serial_controller.is_connected:
            # 断开连接
            self.serial_controller.disconnect()
            self.connect_button.setText("连接串口")
            self.serial_status_label.setText("状态: 未连接")
            self.log_message("串口已断开")
        else:
            # 建立连接
            if self.port_combo.count() == 0:
                QMessageBox.warning(self, "警告", "未找到可用串口，请先刷新串口列表")
                return
            
            port = self.port_combo.currentText()
            baudrate = int(self.baudrate_combo.currentText())
            
            if self.serial_controller.connect(port, baudrate):
                self.connect_button.setText("断开串口")
                self.serial_status_label.setText(f"状态: 已连接到 {port}")
                self.log_message(f"成功连接到串口 {port}，波特率 {baudrate}")
            else:
                QMessageBox.critical(self, "错误", f"无法连接到串口 {port}")
    
    def toggle_camera(self):
        if self.camera_thread.isRunning():
            # 停止摄像头
            self.camera_thread.stop()
            self.camera_button.setText("启动摄像头")
            self.image_label.setText("等待摄像头启动...")
            self.log_message("摄像头已停止")
        else:
            # 启动摄像头
            self.camera_thread.start()
            self.camera_button.setText("停止摄像头")
            self.log_message("摄像头已启动")
    
    def update_image(self, qt_image):
        # 更新图像显示
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def update_detection_result(self, color, pattern, angle):
        # 更新识别结果显示
        self.color_label.setText(f"颜色: {color}")
        self.pattern_label.setText(f"图案: {pattern}")
        self.angle_label.setText(f"角度: {angle if angle else '--'}")
        
        # 保存当前识别结果
        self.detected_color = color
        self.detected_angle = angle
        
        # 自动翻转逻辑
        if self.auto_flip_checkbox.isChecked() and color != "未知":
            # 根据颜色决定翻转方向
            direction = "LEFT" if color == "红色" else "RIGHT"
            self.flip(direction)
    
    def motor_forward(self):
        steps = self.steps_spinbox.value()
        speed = self.speed_spinbox.value()
        response = self.serial_controller.motor_forward(steps, speed)
        self.log_message(f"步进电机向前运动: {steps} 步, 速度: {speed}")
        self.log_message(f"响应: {response}")
    
    def motor_backward(self):
        steps = self.steps_spinbox.value()
        speed = self.speed_spinbox.value()
        response = self.serial_controller.motor_backward(steps, speed)
        self.log_message(f"步进电机向后运动: {steps} 步, 速度: {speed}")
        self.log_message(f"响应: {response}")
    
    def motor_reset(self):
        response = self.serial_controller.motor_reset()
        self.log_message("步进电机复位")
        self.log_message(f"响应: {response}")
    
    def flip(self, direction):
        response = self.serial_controller.flip(direction)
        self.log_message(f"翻转控制: {direction}")
        self.log_message(f"响应: {response}")
    
    def cylinder_extend(self):
        response = self.serial_controller.cylinder_extend()
        self.log_message("气缸伸出")
        self.log_message(f"响应: {response}")
    
    def cylinder_retract(self):
        response = self.serial_controller.cylinder_retract()
        self.log_message("气缸缩回")
        self.log_message(f"响应: {response}")
    
    def cylinder_test(self):
        self.log_message("开始气缸测试...")
        # 气缸伸出
        self.serial_controller.cylinder_extend()
        self.log_message("气缸伸出")
        # 等待1秒
        time.sleep(1)
        # 气缸缩回
        self.serial_controller.cylinder_retract()
        self.log_message("气缸缩回")
        self.log_message("气缸测试完成")
    
    def log_message(self, message):
        # 添加时间戳
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # 添加到日志
        self.log_textedit.insertPlainText(log_entry)
        self.log_textedit.verticalScrollBar().setValue(
            self.log_textedit.verticalScrollBar().maximum())
    
    def clear_log(self):
        self.log_textedit.clear()
        self.log_message("日志已清空")
    
    def closeEvent(self, event):
        # 关闭时停止摄像头和断开串口
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        
        if self.serial_controller.is_connected:
            self.serial_controller.disconnect()
        
        event.accept()

# 主函数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ControlInterface()
    window.show()
    sys.exit(app.exec_())