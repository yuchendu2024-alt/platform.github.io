import cv2
import numpy as np

# 定义HSV颜色范围（示例为红色检测）
lower_red1 = np.array([0, 100, 100])  # H最小值 (0°)
upper_red1 = np.array([10, 255, 255])  # H最大值 (10°)
lower_red2 = np.array([160, 100, 100])  # 红色在色环另一端
upper_red2 = np.array([180, 255, 255])

# 形态学核定义
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 开运算核
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 闭运算核

# 初始化摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 颜色分割
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 双区间掩膜合并（红色检测）
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 形态学优化
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)  # 去噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)  # 填充空洞

    # 2. 轮廓分析
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # 面积阈值过滤小噪声
            continue

        # 形状验证
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # 绘制最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # 形状判断逻辑
        if 0.85 < circularity < 1.2:  # 圆形判定
            color = (0, 255, 0)  # 绿色标识有效目标
            cv2.circle(frame, center, radius, color, 2)
            cv2.putText(frame, f"Circle: {circularity:.2f}", (center[0] - 50, center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:  # 非目标形状
            color = (0, 0, 255)  # 红色标识干扰项
            cv2.drawContours(frame, [cnt], -1, color, 2)

    # 显示结果
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()