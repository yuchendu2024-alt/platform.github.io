import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtCore import Qt

class ShortcutTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("快捷键测试程序")
        self.setGeometry(100, 100, 600, 400)
        
        # 设置暗色主题
        self.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF;")
        
        # 初始化UI
        self.init_ui()
    
    def init_ui(self):
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 添加标题标签
        title_label = QLabel("智能制造平台 - 快捷键测试")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFD700; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # 添加快捷键说明标签
        shortcut_label = QLabel("已实现的快捷键：")
        shortcut_label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        
        # 添加快捷键列表
        shortcuts_text = ""
        shortcuts_text += "• Ctrl+M: 启动步进电机\n"
        shortcuts_text += "• Ctrl+R: 电机复位\n"
        shortcuts_text += "• Ctrl+F: 翻转\n"
        shortcuts_text += "• Ctrl+P: 冲压\n"
        shortcuts_text += "• Ctrl+D: 开始识别\n"
        
        shortcuts_list = QLabel(shortcuts_text)
        shortcuts_list.setStyleSheet("font-size: 14px; margin-bottom: 20px;")
        
        # 添加操作日志
        log_label = QLabel("操作日志：")
        log_label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #444444;")
        self.log_text.append("程序已启动，请测试快捷键功能...")
        
        # 添加到底部状态栏的说明
        status_label = QLabel("请在窗口中按下快捷键进行测试")
        status_label.setStyleSheet("font-size: 12px; color: #CCCCCC; margin-top: 10px;")
        
        # 添加所有组件到布局
        main_layout.addWidget(title_label)
        main_layout.addWidget(shortcut_label)
        main_layout.addWidget(shortcuts_list)
        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_text, 1)
        main_layout.addWidget(status_label)
    
    def keyPressEvent(self, event):
        """键盘事件处理，实现快捷按键功能"""
        # 启动步进电机 - Ctrl+M
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_M:
            self.log_text.append("[快捷键] 启动步进电机")
            event.accept()
        # 电机复位 - Ctrl+R
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_R:
            self.log_text.append("[快捷键] 电机复位")
            event.accept()
        # 翻转气缸 - Ctrl+F
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_F:
            self.log_text.append("[快捷键] 翻转")
            event.accept()
        # 冲压气缸 - Ctrl+P
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_P:
            self.log_text.append("[快捷键] 冲压")
            event.accept()
        # 开始识别 - Ctrl+D
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_D:
            self.log_text.append("[快捷键] 开始识别")
            event.accept()
        else:
            super().keyPressEvent(event)

# 主函数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ShortcutTestWindow()
    window.show()
    sys.exit(app.exec_())