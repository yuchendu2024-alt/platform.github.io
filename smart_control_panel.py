import sys
import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import os
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QGroupBox, QTextEdit, QFrame, QProgressBar, QGridLayout, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 颜色检测参数设置 - 优化为更稳定的默认值
# 应用答辩代码(1).py的颜色检测参数
lower_red1 = np.array([0, 30, 20])
lower_red2 = np.array([160, 30, 20])
lower_blue = np.array([90, 80, 50])
higher_red1 = np.array([10, 255, 150])
higher_red2 = np.array([180, 255, 150])
higher_blue = np.array([130, 255, 255])
kernel = np.ones((3, 3), np.float32)

# 摄像头线程类，用于实时处理视频流
class CameraThread(QThread):
    """摄像头线程类，用于处理视频流和视觉识别，集成ORB匹配功能"""
    image_update = pyqtSignal(QImage)
    raw_image_update = pyqtSignal(np.ndarray)  # 发送原始RGB图像用于保存
    detection_result = pyqtSignal(str, str, str)  # 发送颜色、图案、角度信息
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.capture = None
        self.error = [0]
        self.reference_images = []
        self.load_reference_images()
        self.detection_mode = "camera"  # camera 或 image
        self.current_image = None
        self.auto_detect = True  # 是否自动检测
        
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
        """线程运行函数"""
        self.running = True
        
        if self.detection_mode == "camera":
            self.capture = cv2.VideoCapture(0)
            
        while self.running:
            if self.detection_mode == "camera" and self.capture:
                ret, frame = self.capture.read()
                if not ret:
                    break
                current_frame = frame
            elif self.detection_mode == "image" and self.current_image is not None:
                current_frame = self.current_image.copy()
            else:
                time.sleep(0.1)
                continue
            
            # 进行视觉识别处理
            processed_frame, color, angle = self.process_image(current_frame)
            
            # 发送处理后的图像
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_update.emit(qt_image)
            
            # 发送原始RGB图像用于保存最后一帧
            raw_rgb_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            self.raw_image_update.emit(raw_rgb_image)
            
            # 发送识别结果
            pattern = "已识别" if angle else "未识别"
            self.detection_result.emit(color, pattern, angle)
            
            # 控制循环频率
            if self.detection_mode == "image":
                time.sleep(0.5)  # 图片模式下降低处理频率
            else:
                time.sleep(0.01)  # 降低延迟以提高流畅度
        
        if self.capture:
            self.capture.release()
    
    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()
        self.wait()
    
    def red_detect(self, contours_red):
        """检测红色区域 - 使用更稳定的面积阈值"""
        for contours_reds in contours_red:
            area = cv2.contourArea(contours_reds)
            if area > 1800:  # 恢复原始稳定阈值
                return True
        return False
    
    def blue_detect(self, contours_blue):
        """检测蓝色区域 - 使用更稳定的面积阈值"""
        for contours_blues in contours_blue:
            area = cv2.contourArea(contours_blues)
            if area > 1800:  # 恢复原始稳定阈值
                return True
        return False
    
    def ORBMatcher(self, img1, img2):
        """应用答辩代码(1).py的ORB特征匹配函数，移除误差显示和蓝色框框"""
        img_blur = cv2.GaussianBlur(img2, (5, 5), 2)
        img_mask = np.zeros((img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
        img_mask[370:760, 35:410] = 1
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
        result_red = self.red_detect(contours_red)
        result_blue = self.blue_detect(contours_blue)

        if result_blue:
            result_red = False

        # 初始化颜色变量
        color = "未知"
        if result_red:
            color = "红色"
        elif result_blue:
            color = "蓝色"

        # 进行边缘检测
        edges = cv2.Canny(img_blur, 60, 180)
        # 使用霍夫圆变换检测圆
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=50, minRadius=120, maxRadius=180)

        if circles is not None:
            # 将检测到的圆的坐标和半径转换为整数
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # 获取圆心和半径
                center = (i[0], i[1])
                radius = i[2]

                # 如果检测到红色区域，绘制红色圆
                if (result_red):
                    cv2.circle(img2, center, radius, (0, 0, 255), 3)
                    cv2.putText(img2, f"circle", (200, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                # 如果检测到蓝色区域，绘制蓝色圆
                if (result_blue):
                    cv2.circle(img2, center, radius, (255, 0, 0), 3)
                    cv2.putText(img2, f"circle", (200, 35), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

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
        # 进行特征匹配
        matches = flann.knnMatch(des1, des2, k=2)
        # 筛选出好的匹配点
        points_good = []
        for match in matches:
            if (len(match)) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    points_good.append(m)
        
        MIN_MATCH_COUNT = 30
        theta = ""
        pattern = "未识别"

        if len(points_good) > MIN_MATCH_COUNT:
            # 获取匹配点的坐标
            src_array = np.float32([keypoints1[m.queryIdx].pt for m in points_good]).reshape(-1, 1, 2)
            dst_array = np.float32([keypoints2[m.trainIdx].pt for m in points_good]).reshape(-1, 1, 2)
            # 计算单应性矩阵
            M, mask = cv2.findHomography(src_array, dst_array, cv2.RANSAC, 8.0)
            # 计算旋转角度
            theta_value = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
            theta = str(int(theta_value))
            matchesMask = mask.ravel().tolist()
            
            # 定义绘制匹配点的参数（移除蓝色框框）
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(0, 0, 255),
                               matchesMask=matchesMask,
                               flags=2)
            # 绘制匹配结果
            img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, points_good, None, **draw_params)
            # 如果检测到红色区域，在图像上显示红色文字
            if (result_red):
                cv2.putText(img3, f"color:red", (400, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            # 如果检测到蓝色区域，在图像上显示蓝色文字
            if (result_blue):
                cv2.putText(img3, f"color:blue", (400, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            # 只显示角度，不显示误差
            cv2.putText(img3, f"angle:{int(theta_value)}", (210, 70), cv2.FONT_HERSHEY_PLAIN, 3, (128, 0, 128), 3)

            pattern = "已识别"
            return img3, color, theta, pattern
        else:
            # 显示清晰的中文颜色文本，无背景色
            if color != "未知":
                # 使用PIL确保中文正确显示
                text_color = (255, 0, 0) if color == "红色" else (0, 0, 255)  # BGR格式
                # 将OpenCV图像转换为PIL图像
                img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img2_pil)
                # 尝试使用不同的字体路径
                try:
                    font = ImageFont.truetype("simhei.ttf", 30)
                except:
                    try:
                        font = ImageFont.truetype("c:\\windows\\fonts\\simhei.ttf", 30)
                    except:
                        font = ImageFont.load_default()
                # 绘制中文文本
                draw.text((400, 120), "颜色:" + color, font=font, fill=(255, 0, 0) if color == "红色" else (0, 0, 255))
                # 将PIL图像转回OpenCV格式
                img2 = cv2.cvtColor(np.array(img2_pil), cv2.COLOR_RGB2BGR)
            
            return img2, color, theta, pattern
    
    def process_image(self, frame):
        """处理图像，集成ORB匹配功能，移除误差和蓝色框框"""
        color = "未知"
        angle = ""
        pattern = "未识别"
        display_frame = frame.copy()
        
        # 尝试使用参考图像进行ORB匹配
        if self.reference_images and self.detection_mode == "camera":
            for ref_img in self.reference_images:
                try:
                    matched_frame, detected_color, detected_angle, detected_pattern = self.ORBMatcher(ref_img, frame)
                    if detected_pattern == "已识别":
                        display_frame = matched_frame
                        color = detected_color
                        angle = detected_angle
                        pattern = detected_pattern
                        break
                except Exception as e:
                    # 如果匹配失败，继续尝试下一个参考图像
                    continue
        
        # 如果ORB匹配失败或不在摄像头模式，使用简单的颜色检测
        if pattern == "未识别":
            # 定义检测区域（应用答辩代码的区域设置）
            h, w = frame.shape[:2]
            img_mask = np.zeros((h, w, 3), dtype=np.uint8)
            img_mask[370:760, 35:410] = 1
            img_camera1 = cv2.multiply(frame, img_mask)
            
            # 进行颜色检测
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
            result_red = self.red_detect(contours_red)
            result_blue = self.blue_detect(contours_blue)
            
            # 根据答辩代码的逻辑，避免同时识别两种颜色
            if result_blue:
                result_red = False
            
            # 根据检测结果设置颜色
            if result_red:
                color = "红色"
                # 使用PIL绘制中文文本
                display_frame_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(display_frame_pil)
                try:
                    font = ImageFont.truetype("simhei.ttf", 30)
                except:
                    try:
                        font = ImageFont.truetype("c:\\windows\\fonts\\simhei.ttf", 30)
                    except:
                        font = ImageFont.load_default()
                draw.text((400, 120), "颜色:红色", font=font, fill=(255, 0, 0))
                display_frame = cv2.cvtColor(np.array(display_frame_pil), cv2.COLOR_RGB2BGR)
            elif result_blue:
                color = "蓝色"
                # 使用PIL绘制中文文本
                display_frame_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(display_frame_pil)
                try:
                    font = ImageFont.truetype("simhei.ttf", 30)
                except:
                    try:
                        font = ImageFont.truetype("c:\\windows\\fonts\\simhei.ttf", 30)
                    except:
                        font = ImageFont.load_default()
                draw.text((400, 120), "颜色:蓝色", font=font, fill=(0, 0, 255))
                display_frame = cv2.cvtColor(np.array(display_frame_pil), cv2.COLOR_RGB2BGR)
        
        # 更新信号发送
        return display_frame, color, angle
    
    def set_detection_mode(self, mode):
        """设置检测模式：camera 或 image"""
        self.detection_mode = mode
        if mode == "camera" and self.isRunning():
            # 如果正在运行且切换到摄像头模式，重启线程以打开摄像头
            self.stop()
            self.start()
    
    def set_current_image(self, image):
        """设置当前要检测的图像"""
        self.current_image = image
    
    def set_reference_images(self, images):
        """设置参考图像列表"""
        self.reference_images = images
    
    def load_reference_images(self):
        """加载默认参考图像"""
        try:
            # 尝试加载默认参考图像
            for i in range(1, 4):  # 尝试加载1.jpg, 2.jpg, 3.jpg
                img_path = f"d:/opencv/{i}.jpg"
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.reference_images.append(img)
        except Exception as e:
            print(f"加载默认参考图像失败: {str(e)}")
    
    # ORBMatcher函数已移除，现使用test999999.py中的color_detector进行图像处理

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

# 主窗口类
class SmartControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能制造平台")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化串口控制器和摄像头线程
        self.serial_controller = SerialController()
        self.camera_thread = CameraThread()
        
        # 初始化状态变量
        self.is_camera_on = False
        self.is_serial_connected = False
        self.current_image_path = None  # 当前加载的图片路径
        self.last_frame = None  # 保存最后一帧原始图像
        
        # 设置窗口的样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f8f8;
                color: #333333;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
        """)
        
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
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 1. 顶部区域 - 标题和图像
        header_layout = QHBoxLayout()
        header_layout.setSpacing(20)
        
        # 标题标签替代logo
        title_label = QLabel("设计与建造翻转冲压平台")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #333333;")
        title_label.setMinimumSize(250, 120)
        title_label.setMaximumSize(250, 120)
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        
        # 中间区域 - 摄像头显示
        self.image_label = QLabel("摄像头画面")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #F8F8F8; border: 1px solid #E0E0E0; color: #888888;")
        self.image_label.setMinimumSize(500, 350)
        header_layout.addWidget(self.image_label)
        
        # 右侧区域 - 识别结果
        result_group = QGroupBox("识别结果")
        result_group.setStyleSheet(self.get_groupbox_style())
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(15, 15, 15, 15)
        
        self.result_textedit = QTextEdit()
        self.result_textedit.setStyleSheet(self.get_textedit_style())
        self.result_textedit.setReadOnly(True)
        self.result_textedit.setMinimumWidth(280)
        self.result_textedit.setMinimumHeight(300)
        
        result_layout.addWidget(self.result_textedit)
        header_layout.addWidget(result_group)
        
        main_layout.addLayout(header_layout)
        
        # 2. 中间区域 - 控制区域
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(20)
        
        # 左侧 - 串口设置区域
        serial_group = QGroupBox("端口设置")
        serial_group.setStyleSheet(self.get_groupbox_style())
        serial_layout = QVBoxLayout(serial_group)
        serial_layout.setContentsMargins(15, 15, 15, 15)
        serial_layout.setSpacing(10)
        
        # 串口选择
        port_label = QLabel("选择端口:")
        port_label.setStyleSheet("color: #333333;")
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet(self.get_combo_style())
        self.port_combo.setMinimumWidth(180)
        
        # 串口按钮
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)
        self.refresh_button = QPushButton("检测端口")
        self.refresh_button.setStyleSheet(self.get_button_style())
        self.connect_button = QPushButton("打开端口")
        self.connect_button.setStyleSheet(self.get_button_style())
        self.disconnect_button = QPushButton("关闭端口")
        self.disconnect_button.setStyleSheet(self.get_button_style())
        
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.connect_button)
        button_layout.addWidget(self.disconnect_button)
        
        # 添加到串口布局
        serial_layout.addWidget(port_label)
        serial_layout.addWidget(self.port_combo)
        serial_layout.addLayout(button_layout)
        serial_layout.addStretch(1)
        
        middle_layout.addWidget(serial_group, 1)
        
        # 中间 - 摄像头控制和系统日志区域
        control_group = QGroupBox("系统控制")
        control_group.setStyleSheet(self.get_groupbox_style())
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(15)
        
        # 摄像头控制和图像检测控制
        camera_control_layout = QHBoxLayout()
        camera_control_layout.setSpacing(10)
        
        # 摄像头选择
        self.camera_combo = QComboBox()
        self.camera_combo.setStyleSheet(self.get_combo_style())
        self.camera_combo.addItem("Camera 0")
        self.camera_combo.addItem("Camera 1")
        
        # 检测模式选择
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.setStyleSheet(self.get_combo_style())
        self.detection_mode_combo.addItem("摄像头检测")
        self.detection_mode_combo.addItem("图片检测")
        
        # 功能按钮
        self.open_camera_button = QPushButton("打开摄像头")
        self.open_camera_button.setStyleSheet(self.get_button_style())
        self.close_camera_button = QPushButton("关闭摄像头")
        self.close_camera_button.setStyleSheet(self.get_button_style())
        self.insert_image_button = QPushButton("插入图片")
        self.insert_image_button.setStyleSheet(self.get_button_style())
        self.detect_image_button = QPushButton("检测图片")
        self.detect_image_button.setStyleSheet(self.get_button_style())
        
        # 添加到布局 - 更合理的分组
        control_subgroup1 = QHBoxLayout()
        control_subgroup1.addWidget(self.camera_combo)
        control_subgroup1.addWidget(self.detection_mode_combo)
        
        control_subgroup2 = QHBoxLayout()
        control_subgroup2.addWidget(self.open_camera_button)
        control_subgroup2.addWidget(self.close_camera_button)
        control_subgroup2.addWidget(self.insert_image_button)
        control_subgroup2.addWidget(self.detect_image_button)
        
        camera_control_layout.addLayout(control_subgroup1)
        camera_control_layout.addLayout(control_subgroup2)
        camera_control_layout.setContentsMargins(0, 0, 0, 0)
        
        # 系统日志
        self.log_textedit = QTextEdit()
        self.log_textedit.setStyleSheet(self.get_textedit_style())
        self.log_textedit.setReadOnly(True)
        self.log_textedit.setMinimumHeight(150)
        
        # 添加到控制布局
        control_layout.addLayout(camera_control_layout)
        control_layout.addWidget(self.log_textedit)
        
        middle_layout.addWidget(control_group, 2)
        
        # 右侧 - 平台信息区域（显示设备图片）
        info_group = QGroupBox("平台信息")
        info_group.setStyleSheet(self.get_groupbox_style())
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(15, 15, 15, 15)
        
        # 使用QLabel显示设备图片
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        # 尝试加载图片并设置到标签
        try:
            # 使用5.jpg作为显示图片
            pixmap = QPixmap("5.jpg")
            # 调整图片大小以适应标签
            scaled_pixmap = pixmap.scaled(240, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.info_label.setPixmap(scaled_pixmap)
        except Exception as e:
            # 如果图片加载失败，显示错误信息
            self.info_label.setText(f"图片加载失败\n{str(e)}")
            self.info_label.setStyleSheet("color: #ff0000; font-size: 12px; text-align: center; padding: 20px;")
            self.info_label.setWordWrap(True)
        
        info_layout.addWidget(self.info_label)
        
        middle_layout.addWidget(info_group, 1)
        
        main_layout.addLayout(middle_layout)
        
        # 3. 下部区域 - 快捷按钮
        buttons_group = QGroupBox("控制操作")
        buttons_group.setStyleSheet(self.get_groupbox_style())
        buttons_layout = QGridLayout(buttons_group)
        buttons_layout.setContentsMargins(20, 15, 20, 15)
        buttons_layout.setSpacing(12)
        
        # 创建快捷按钮 - 明确标记功能
        self.start_button = QPushButton("启动流程")
        self.start_button.setStyleSheet(self.get_button_style())
        self.start_button.setMinimumHeight(45)
        
        self.motor_button = QPushButton("启动电机")
        self.motor_button.setStyleSheet(self.get_button_style())
        self.motor_button.setMinimumHeight(45)
        
        self.reset_button = QPushButton("电机复位")
        self.reset_button.setStyleSheet(self.get_button_style())
        self.reset_button.setMinimumHeight(45)
        
        self.rotate_button = QPushButton("翻转操作")
        self.rotate_button.setStyleSheet(self.get_button_style())
        self.rotate_button.setMinimumHeight(45)
        
        # 添加冲压和识别按钮
        self.press_button = QPushButton("冲压操作")
        self.press_button.setStyleSheet(self.get_button_style())
        self.press_button.setMinimumHeight(45)
        
        self.recognize_button = QPushButton("开始识别")
        self.recognize_button.setStyleSheet(self.get_button_style())
        self.recognize_button.setMinimumHeight(45)
        
        # 按功能布局添加按钮 - 更现代化的排列
        buttons_layout.addWidget(self.start_button, 0, 0, 1, 2)
        buttons_layout.addWidget(self.motor_button, 1, 0)
        buttons_layout.addWidget(self.reset_button, 1, 1)
        buttons_layout.addWidget(self.rotate_button, 2, 0)
        buttons_layout.addWidget(self.press_button, 2, 1)
        buttons_layout.addWidget(self.recognize_button, 3, 0, 1, 2)
        
        main_layout.addWidget(buttons_group)
        
        # 4. 底部状态栏
        status_layout = QHBoxLayout()
        status_layout.setSpacing(20)
        
        # 项目内容和快捷键提示
        project_label = QLabel("项目内容: 移动 → 图像检测 → 零件转动 → 翻转气缸 | 界面功能: 启动步进电机、电机复位、翻转、冲压、识别")
        project_label.setStyleSheet("color: #666666; font-size: 12px;")
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(self.get_progress_style())
        self.progress_bar.setValue(0)
        
        # 添加到状态栏布局
        status_layout.addWidget(project_label, 1)
        status_layout.addWidget(self.progress_bar, 2)
        
        main_layout.addLayout(status_layout)
    
    def get_groupbox_style(self):
        return """
        QGroupBox {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-top: 10px;
        }
        QGroupBox::title {
            color: #333333;
            font-weight: 500;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        """
    
    def get_button_style(self):
        return """
        QPushButton {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
            border-color: #d0d0d0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
        QPushButton:disabled {
            background-color: #f5f5f5;
            color: #aaaaaa;
            border-color: #eeeeee;
        }
        """
    
    def get_combo_style(self):
        return """
        QComboBox {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 6px 10px;
            min-width: 120px;
            font-size: 13px;
        }
        QComboBox:hover {
            border-color: #d0d0d0;
        }
        QComboBox::drop-down {
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
            border-left: 1px solid #e0e0e0;
        }
        QComboBox::down-arrow {
            color: #666666;
        }
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 4px;
        }
        QComboBox QAbstractItemView::item:hover {
            background-color: #f5f5f5;
        }
        """
    
    def get_textedit_style(self):
        return """
        QTextEdit {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 10px;
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }
        """
    
    def get_progress_style(self):
        return """
        QProgressBar {
            background-color: #f0f0f0;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            text-align: center;
            color: #666666;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #4a6fa5;
            border-radius: 6px;
        }
        """
    
    def connect_signals(self):
        # 串口相关信号
        self.refresh_button.clicked.connect(self.refresh_serial_ports)
        self.connect_button.clicked.connect(self.connect_serial)
        self.disconnect_button.clicked.connect(self.disconnect_serial)
        
        # 摄像头和图像检测相关信号
        self.open_camera_button.clicked.connect(self.open_camera)
        self.close_camera_button.clicked.connect(self.close_camera)
        self.insert_image_button.clicked.connect(self.insert_image)
        self.detect_image_button.clicked.connect(self.detect_image)
        self.detection_mode_combo.currentIndexChanged.connect(self.on_detection_mode_changed)
        self.camera_thread.image_update.connect(self.update_image)
        self.camera_thread.raw_image_update.connect(self.save_last_frame)
        self.camera_thread.detection_result.connect(self.update_detection_result)
        
        # 快捷按钮信号
        self.start_button.clicked.connect(self.start_process)
        self.motor_button.clicked.connect(self.start_motor)
        self.reset_button.clicked.connect(self.reset_motor)
        self.rotate_button.clicked.connect(self.activate_cylinder2)  # 翻转功能使用气缸2
        self.press_button.clicked.connect(self.activate_cylinder3)   # 冲压功能使用气缸3
        self.recognize_button.clicked.connect(self.start_recognition)  # 识别功能
    
    def refresh_serial_ports(self):
        ports = self.serial_controller.get_available_ports()
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        self.log_message("已刷新串口列表")
    
    def connect_serial(self):
        if self.port_combo.count() == 0:
            self.log_message("错误: 未找到可用串口，请先检测端口")
            return
        
        port = self.port_combo.currentText()
        if self.serial_controller.connect(port):
            self.is_serial_connected = True
            self.connect_button.setDisabled(True)
            self.disconnect_button.setEnabled(True)
            self.log_message(f"成功连接到串口 {port}")
        else:
            self.log_message(f"错误: 无法连接到串口 {port}")
    
    def disconnect_serial(self):
        if self.serial_controller.disconnect():
            self.is_serial_connected = False
            self.connect_button.setEnabled(True)
            self.disconnect_button.setDisabled(True)
            self.log_message("串口已断开连接")
    
    def open_camera(self):
        try:
            # 检查摄像头是否已经打开
            if self.is_camera_on:
                self.log_message("摄像头已经打开")
                return
            
            # 获取选择的摄像头索引
            camera_index = self.camera_combo.currentIndex()
            
            self.log_message(f"准备启动摄像头 {camera_index}...")
            
            # 确保之前的线程已经清理
            if hasattr(self, 'camera_thread') and self.camera_thread is not None:
                self.log_message("清理残留的摄像头线程...")
                try:
                    self.camera_thread.image_update.disconnect(self.update_image)
                    self.camera_thread.raw_image_update.disconnect(self.save_last_frame)
                    self.camera_thread.detection_result.disconnect(self.update_detection_result)
                except:
                    pass
                self.camera_thread.stop()
                self.camera_thread = None
            
            # 创建并启动摄像头线程
            self.camera_thread = CameraThread()
            # 设置检测模式
            self.camera_thread.set_detection_mode("camera")
            # 连接信号
            self.camera_thread.image_update.connect(self.update_image)
            self.camera_thread.raw_image_update.connect(self.save_last_frame)
            self.camera_thread.detection_result.connect(self.update_detection_result)
            self.camera_thread.start()
            
            self.is_camera_on = True
            self.open_camera_button.setDisabled(True)
            self.close_camera_button.setEnabled(True)
            self.log_message(f"摄像头 {camera_index} 已启动")
            
        except Exception as e:
            self.log_message(f"错误: 无法启动摄像头 - {str(e)}")
            # 确保状态正确
            self.is_camera_on = False
            self.open_camera_button.setEnabled(True)
            self.close_camera_button.setEnabled(False)
    
    def close_camera(self):
        if self.is_camera_on:
            try:
                self.log_message("开始关闭摄像头...")
                if not self.is_camera_on:
                    self.log_message("摄像头已经关闭")
                    return
                
                # 安全停止摄像头线程
                if hasattr(self, 'camera_thread') and self.camera_thread is not None:
                    self.log_message("停止摄像头线程...")
                    # 断开信号连接，防止线程仍在运行时发出信号
                    try:
                        self.camera_thread.image_update.disconnect(self.update_image)
                        self.camera_thread.raw_image_update.disconnect(self.save_last_frame)
                        self.camera_thread.detection_result.disconnect(self.update_detection_result)
                    except Exception as e:
                        self.log_message(f"断开信号连接时出错: {str(e)}")
                    
                    # 停止线程
                    self.camera_thread.stop()
                    
                    # 等待线程结束
                    if self.camera_thread.isRunning():
                        self.log_message("等待线程结束...")
                        self.camera_thread.wait(2000)  # 最多等待2秒
                        if self.camera_thread.isRunning():
                            self.log_message("警告: 线程未能及时结束")
                    
                    # 清理线程对象
                    self.camera_thread = None
                    self.log_message("摄像头线程已清理")
                
                self.is_camera_on = False
                # 不清除图像标签文本，保持最后一帧显示
                # self.image_label.setText("摄像头画面")
                self.open_camera_button.setEnabled(True)
                self.close_camera_button.setDisabled(False)  # 设置为可用，允许再次关闭
                self.log_message("摄像头已关闭")
            except Exception as e:
                # 捕获所有异常，防止程序崩溃
                self.log_message(f"关闭摄像头时出错: {str(e)}")
                # 即使发生错误，也要确保状态被重置
                self.is_camera_on = False
                self.open_camera_button.setEnabled(True)
                self.close_camera_button.setDisabled(False)  # 确保可以再次操作
    
    def update_image(self, qt_image):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def save_last_frame(self, raw_rgb_image):
        """保存最后一帧原始图像，用于ESC键退出摄像头后显示"""
        self.last_frame = raw_rgb_image
    
    def update_detection_result(self, color, pattern, angle):
        # 更新结果显示，只显示颜色和角度
        result_text = f"检测结果:\n"
        result_text += f"- 颜色: {color}\n"
        result_text += f"- 角度: {angle if angle else '--'}\n"
        self.result_textedit.setText(result_text)
        
        # 在日志中记录颜色检测结果
        if color != "未知":
            self.log_message(f"颜色检测结果: {color}")
        
        # 更新进度条
        self.update_progress(50)  # 模拟进度
    
    def insert_image(self):
        """插入图片功能，修复卡退问题"""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)", options=options)
            
            if file_path:
                self.current_image_path = file_path
                try:
                    # 读取图片
                    image = cv2.imread(file_path)
                    if image is not None:
                        # 如果摄像头正在运行，先关闭
                        if self.is_camera_on:
                            self.close_camera()
                        
                        # 设置为图片检测模式
                        self.detection_mode_combo.setCurrentIndex(1)  # 切换到图片检测
                        
                        # 更新标签显示图片
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        
                        self.log_message(f"已插入图片: {file_path}")
                    else:
                        self.log_message(f"错误: 无法读取图片 {file_path}")
                except Exception as e:
                    self.log_message(f"错误: 处理图片时出错 - {str(e)}")
        except Exception as e:
            # 捕获所有异常，确保界面不会卡退
            self.log_message(f"严重错误: 插入图片时发生异常 - {str(e)}")
    
    def detect_image(self):
        """检测当前插入的图片，使用ORB匹配方式"""
        try:
            if self.current_image_path:
                try:
                    image = cv2.imread(self.current_image_path)
                    if image is not None:
                        # 如果摄像头线程没有运行，创建并启动它
                        if not self.is_camera_on:
                            self.camera_thread = CameraThread()
                            self.camera_thread.image_update.connect(self.update_image)
                            self.camera_thread.raw_image_update.connect(self.save_last_frame)
                            self.camera_thread.detection_result.connect(self.update_detection_result)
                            self.is_camera_on = True
                        
                        # 设置为图片检测模式
                        self.camera_thread.set_detection_mode("image")
                        self.camera_thread.set_current_image(image)
                        
                        # 如果线程没有启动，启动它
                        if not self.camera_thread.isRunning():
                            self.camera_thread.start()
                        
                        self.log_message(f"开始检测图片: {self.current_image_path}")
                    else:
                        self.log_message(f"错误: 无法读取图片 {self.current_image_path}")
                except Exception as e:
                    self.log_message(f"错误: 检测图片时出错 - {str(e)}")
            else:
                self.log_message("错误: 请先插入图片")
        except Exception as e:
            # 捕获所有异常，确保界面不会卡退
            self.log_message(f"严重错误: 检测图片时发生异常 - {str(e)}")
    
    def on_detection_mode_changed(self, index):
        """检测模式变更时的处理"""
        if index == 0:  # 摄像头检测
            self.log_message("切换到摄像头检测模式")
            if self.is_camera_on:
                self.camera_thread.set_detection_mode("camera")
        else:  # 图片检测
            self.log_message("切换到图片检测模式")
            if self.is_camera_on:
                self.camera_thread.set_detection_mode("image")
    
    def start_process(self):
        """启动完整处理流程"""
        # 流程控制由STM8单片机处理，这里只发送启动信号
        command = "START_PROCESS"
        if self.serial_controller and self.serial_controller.is_connected:
            self.send_command(command)
        self.log_message(f"发送电信号: {command}")
        
        # 同时启动摄像头用于识别
        if not self.camera_thread or not self.camera_thread.isRunning():
            self.open_camera()
    
    def start_motor(self):
        """启动步进电机，发送电信号给STM8单片机"""
        # 直接发送电信号指令，不在界面显示额外操作
        command = "MOTOR_START"
        if self.serial_controller and self.serial_controller.is_connected:
            self.send_command(command)
        self.log_message(f"发送电信号: {command}")
    
    def reset_motor(self):
        """复位步进电机，发送电信号给STM8单片机"""
        # 直接发送电信号指令，不在界面显示额外操作
        command = "MOTOR_RESET"
        if self.serial_controller and self.serial_controller.is_connected:
            self.send_command(command)
        self.log_message(f"发送电信号: {command}")
    
    def activate_cylinder2(self):
        """翻转操作，发送电信号给STM8单片机"""
        # 直接发送电信号指令，不在界面显示额外操作
        command = "FLIP_ACTION"
        if self.serial_controller and self.serial_controller.is_connected:
            self.send_command(command)
        self.log_message(f"发送电信号: {command}")
    
    def activate_cylinder3(self):
        """冲压操作，发送电信号给STM8单片机"""
        # 直接发送电信号指令，不在界面显示额外操作
        command = "PRESS_ACTION"
        if self.serial_controller and self.serial_controller.is_connected:
            self.send_command(command)
        self.log_message(f"发送电信号: {command}")
    
    def start_recognition(self):
        """识别操作"""
        # 直接发送电信号指令
        command = "START_RECOGNITION"
        if self.serial_controller and self.serial_controller.is_connected:
            self.send_command(command)
        self.log_message(f"发送电信号: {command}")
        
        # 根据当前检测模式进行识别
        current_mode = self.detection_mode_combo.currentIndex()
        if current_mode == 0:  # 摄像头模式
            # 启动摄像头识别
            if not self.camera_thread or not self.camera_thread.isRunning():
                self.open_camera()
        else:  # 图片模式
            # 如果有插入的图片，进行检测
            if self.current_image_path:
                self.detect_image()
            else:
                self.log_message("提示: 图片模式下请先插入图片")
    
    def send_command(self, command):
        """发送电信号命令到STM8单片机"""
        if not self.serial_controller or not self.serial_controller.is_connected:
            # 即使未连接串口，也记录信号发送（用于测试和日志记录）
            return "NOT_CONNECTED"
        
        try:
            # 直接发送命令，不等待详细响应
            self.serial_controller.send_command(command)
            return "SENT"
        except Exception as e:
            # 出错时仅记录错误，但不中断流程
            return "ERROR"
    
    def log_message(self, message):
        """记录系统日志"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_textedit.insertPlainText(log_entry)
        self.log_textedit.verticalScrollBar().setValue(
            self.log_textedit.verticalScrollBar().maximum())
    
    # 保留一个start_recognition方法，移除重复定义
            
    def update_progress(self, value):
        """更新进度条"""
        if 0 <= value <= 100:
            self.progress_bar.setValue(value)
    
    def keyPressEvent(self, event):
        """键盘事件处理，实现快捷按键功能"""
        # ESC键退出摄像头但保持界面运行
        if event.key() == Qt.Key_Escape:
            if self.is_camera_on:
                self.close_camera()
                self.log_message("已按ESC键退出摄像头，界面继续保持运行")
                # 显示最后一帧图像或默认图像
                if hasattr(self, 'last_frame') and self.last_frame is not None:
                    h, w, ch = self.last_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(self.last_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                        self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            event.accept()
        # 启动步进电机 - Ctrl+M
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_M:
            self.motor_button.click()
            event.accept()
        # 电机复位 - Ctrl+R
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_R:
            self.reset_button.click()
            event.accept()
        # 翻转气缸 - Ctrl+F
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_F:
            self.rotate_button.click()
            event.accept()
        # 冲压气缸 - Ctrl+P
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_P:
            self.press_button.click()
            event.accept()
        # 开始识别 - Ctrl+D
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_D:
            self.recognize_button.click()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        if self.is_camera_on:
            self.camera_thread.stop()
        if self.is_serial_connected:
            self.serial_controller.disconnect()
        event.accept()

# 主函数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置应用程序样式
    app.setStyle("Fusion")
    window = SmartControlPanel()
    window.show()
    sys.exit(app.exec_())