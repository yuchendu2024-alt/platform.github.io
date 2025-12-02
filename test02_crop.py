# 图像裁剪操作
import cv2

image = cv2.imread("opencv_logo.jpg")

crop = image[1:120, 1:120]

cv2.imshow("crop", crop)
cv2.waitKey()