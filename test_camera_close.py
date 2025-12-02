import sys
import time
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

class CameraThread(QThread):
    image_update = pyqtSignal(QImage)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        print("CameraThread initialized with camera ID:", camera_id)
    
    def run(self):
        try:
            print("Starting camera thread...")
            self.running = True
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"Error: Cannot open camera {self.camera_id}")
                self.running = False
                return
            
            print(f"Camera {self.camera_id} opened successfully")
            
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    # Convert frame to Qt format
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.image_update.emit(qt_image)
                time.sleep(0.02)
        except Exception as e:
            print(f"Error in camera thread: {str(e)}")
        finally:
            print("Camera thread finishing...")
            self.stop()
    
    def stop(self):
        try:
            self.running = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                print("Releasing camera...")
                self.cap.release()
                print("Camera released")
            time.sleep(0.1)  # Give time for release
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")
        print("Camera thread stopped")

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摄像头测试")
        self.setGeometry(100, 100, 800, 600)
        self.is_camera_on = False
        self.camera_thread = None
        
        self.init_ui()
        print("Test window initialized")
    
    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        
        # 摄像头控制
        control_layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Camera 0")
        self.camera_combo.addItem("Camera 1")
        control_layout.addWidget(self.camera_combo)
        
        self.open_camera_button = QPushButton("打开摄像头")
        self.open_camera_button.clicked.connect(self.open_camera)
        control_layout.addWidget(self.open_camera_button)
        
        self.close_camera_button = QPushButton("关闭摄像头")
        self.close_camera_button.clicked.connect(self.close_camera)
        self.close_camera_button.setEnabled(False)
        control_layout.addWidget(self.close_camera_button)
        
        # 其他操作按钮
        self.other_button1 = QPushButton("其他操作1")
        self.other_button1.clicked.connect(lambda: print("其他操作1被点击"))
        control_layout.addWidget(self.other_button1)
        
        self.other_button2 = QPushButton("其他操作2")
        self.other_button2.clicked.connect(lambda: print("其他操作2被点击"))
        control_layout.addWidget(self.other_button2)
        
        main_layout.addLayout(control_layout)
        
        # 摄像头画面
        self.image_label = QLabel("摄像头画面")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000;")
        self.image_label.setMinimumHeight(400)
        main_layout.addWidget(self.image_label)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)
        
        # 设置中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def open_camera(self):
        try:
            print("Opening camera...")
            if self.is_camera_on:
                print("Camera is already on")
                return
            
            camera_id = int(self.camera_combo.currentText().split()[1])
            
            # 创建并启动摄像头线程
            self.camera_thread = CameraThread(camera_id)
            self.camera_thread.image_update.connect(self.update_image)
            self.camera_thread.finished.connect(self.on_thread_finished)
            self.camera_thread.start()
            
            self.is_camera_on = True
            self.open_camera_button.setEnabled(False)
            self.close_camera_button.setEnabled(True)
            self.status_label.setText(f"摄像头 {camera_id} 已启动")
            print(f"Camera {camera_id} started")
        except Exception as e:
            print(f"Error opening camera: {str(e)}")
            self.status_label.setText(f"打开摄像头错误: {str(e)}")
    
    def close_camera(self):
        try:
            print("\nClosing camera...")
            if not self.is_camera_on:
                print("Camera is already off")
                return
            
            # 安全停止摄像头线程
            if hasattr(self, 'camera_thread') and self.camera_thread is not None:
                print("Stopping camera thread...")
                # 断开信号连接，防止线程仍在运行时发出信号
                try:
                    self.camera_thread.image_update.disconnect(self.update_image)
                except:
                    pass
                
                # 停止线程
                self.camera_thread.stop()
                # 等待线程结束
                if self.camera_thread.isRunning():
                    print("Waiting for thread to finish...")
                    self.camera_thread.wait(2000)  # 最多等待2秒
                    if self.camera_thread.isRunning():
                        print("Warning: Thread did not finish in time")
                
                # 清理线程对象
                self.camera_thread = None
                print("Camera thread cleaned up")
            
            self.is_camera_on = False
            self.open_camera_button.setEnabled(True)
            self.close_camera_button.setEnabled(False)
            self.status_label.setText("摄像头已关闭")
            print("Camera closed successfully")
        except Exception as e:
            print(f"Error closing camera: {str(e)}")
            # 即使发生错误，也要确保状态被重置
            self.is_camera_on = False
            self.open_camera_button.setEnabled(True)
            self.close_camera_button.setEnabled(False)
            self.status_label.setText(f"关闭摄像头错误: {str(e)}")
    
    def update_image(self, qt_image):
        try:
            self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Error updating image: {str(e)}")
    
    def on_thread_finished(self):
        print("Camera thread finished signal received")
        # 可以在这里进行额外的清理
    
    def closeEvent(self, event):
        print("Window closing...")
        self.close_camera()
        event.accept()

if __name__ == "__main__":
    print("Starting application...")
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    print("Application running")
    sys.exit(app.exec_())