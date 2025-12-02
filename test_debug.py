import sys
import cv2
import numpy as np
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QGroupBox, QTextEdit, QFrame, QProgressBar, QGridLayout, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 颜色检测参数设置
lower_red1 = np.array([0, 50, 50])
lower_red2 = np.array([170, 50, 50])
lower_blue = np.array([100, 50, 50])
higher_red1 = np.array([10, 255, 255])
higher_red2 = np.array([180, 255, 255])
higher_blue = np.array([125, 255, 255])
kernel = np.ones((5, 5), np.float32)

# 摄像头线程类，用于实时处理视频流
class CameraThread(QThread):
    """摄像头线程类，用于处理视频流和视觉识别"""
    image_update = pyqtSignal(QImage)
    raw_image_update = pyqtSignal(np.ndarray)
    detection_result = pyqtSignal(str, str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.capture = None
        self.reference_images = []
        self.detection_mode = "camera"
        self.current_image = None
    
    def run(self):
        self.running = True
        
        if self.detection_mode == "camera":
            self.capture = cv2.VideoCapture(0)
            
        while self.running:
            if self.detection_mode == "camera" and self.capture:
                ret, frame = self.capture.read()
                if not ret:
                    break
                current_frame = frame
            else:
                time.sleep(0.1)
                continue
            
            # 简单的图像处理
            rgb_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_update.emit(qt_image)
            
            # 发送识别结果
            self.detection_result.emit("未知", "未识别", "")
            
            time.sleep(0.02)
        
        if self.capture:
            self.capture.release()
    
    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()
        self.wait()

# 主窗口类
class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("测试窗口")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化摄像头线程
        self.camera_thread = CameraThread()
        self.is_camera_on = False
        
        # 创建主布局
        self.init_ui()
        
        # 连接信号和槽
        self.connect_signals()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # 图像显示
        self.image_label = QLabel("摄像头画面")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1E1E1E; border: 1px solid #444444; color: #FFFFFF;")
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.open_camera_button = QPushButton("打开摄像头")
        self.close_camera_button = QPushButton("关闭摄像头")
        self.close_camera_button.setDisabled(True)
        
        control_layout.addWidget(self.open_camera_button)
        control_layout.addWidget(self.close_camera_button)
        
        main_layout.addLayout(control_layout)
    
    def connect_signals(self):
        self.open_camera_button.clicked.connect(self.open_camera)
        self.close_camera_button.clicked.connect(self.close_camera)
        self.camera_thread.image_update.connect(self.update_image)
        self.camera_thread.detection_result.connect(self.update_detection_result)
    
    def open_camera(self):
        if not self.is_camera_on:
            try:
                self.camera_thread = CameraThread()
                self.camera_thread.image_update.connect(self.update_image)
                self.camera_thread.detection_result.connect(self.update_detection_result)
                self.camera_thread.start()
                self.is_camera_on = True
                self.open_camera_button.setDisabled(True)
                self.close_camera_button.setEnabled(True)
                print("摄像头已启动")
            except Exception as e:
                print(f"错误: 无法启动摄像头 - {str(e)}")
    
    def close_camera(self):
        if self.is_camera_on:
            try:
                # 安全停止摄像头线程
                self.camera_thread.stop()
                self.is_camera_on = False
                self.open_camera_button.setEnabled(True)
                self.close_camera_button.setDisabled(True)
                print("摄像头已关闭")
            except Exception as e:
                # 捕获所有异常，防止程序崩溃
                print(f"关闭摄像头时出错: {str(e)}")
                self.is_camera_on = False
                self.open_camera_button.setEnabled(True)
                self.close_camera_button.setDisabled(True)
    
    def update_image(self, qt_image):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def update_detection_result(self, color, pattern, angle):
        pass
    
    def closeEvent(self, event):
        if self.is_camera_on:
            self.camera_thread.stop()
        event.accept()

# 主函数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())