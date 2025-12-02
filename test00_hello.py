import cv2

print(cv2.getVersionString())# 读取版本号

image = cv2.imread("opencv_logo.jpg")# 读取图片

print(image.shape)# 打印图片的形状（高度，宽度，通道数）

cv2.imshow("image", image)#把openCV读取到的图像数据显示到一个窗口上
cv2.waitKey()  # 让窗口暂停

