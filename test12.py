import cv2
import numpy as np

# 读取图像
image = cv2.imread('opencv_logo.jpg')

# 将图像从BGR颜色空间转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色在HSV颜色空间中的两个阈值范围
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
# 定义了红色在 HSV 颜色空间中的第一个阈值范围，较低值和较高值分别表示色相、饱和度和明度的下限和上限
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
# 定义了红色在 HSV 颜色空间中的第二个阈值范围，用于处理色相环上接近 0° 和 180° 的红色部分(因为红色在 HSV 中分布在两端）

# 根据两个阈值范围分别生成掩膜
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
# 使用cv2.inRange函数根据第一个红色阈值范围在hsv_image中生成掩膜mask1，掩膜中符合阈值范围的像素值为 255（白色），不符合的为 0（黑色）
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# 合并两个掩膜
mask = cv2.bitwise_or(mask1, mask2)  # 将两个掩膜合并成一个mask，这样就包含了所有符合红色阈值范围的像素

# 根据掩膜提取红色部分
red_part = cv2.bitwise_and(image, image, mask=mask)
# 将原始图像image与自身进行按位与操作，并使用合并后的掩膜mask。这样就提取出了图像中所有红色部分，存储在red_part变量中

# 显示提取的红色部分
cv2.imshow('original', image)
cv2.imshow('Red Part', red_part)
cv2.waitKey(0)
cv2.destroyAllWindows()