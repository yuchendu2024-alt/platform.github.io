import cv2
import numpy as np

# 颜色检测参数设置
lower_red1 = np.array([0, 50, 50])
lower_red2 = np.array([170, 50, 50])
lower_blue = np.array([100, 50, 50])
higher_red1 = np.array([10, 255, 255])
higher_red2 = np.array([180, 255, 255])
higher_blue = np.array([125, 255, 255])
kernel = np.ones((5, 5), np.float32)

class ColorDetection:
    def __init__(self):
        self.auto_detect = True  # 自动检测标志
    
    def red_detect(self, contours_red):
        """检测红色区域"""
        for contours_reds in contours_red:
            area = cv2.contourArea(contours_reds)
            if area > 1800:
                return True
        return False
    
    def blue_detect(self, contours_blue):
        """检测蓝色区域"""
        for contours_blues in contours_blue:
            area = cv2.contourArea(contours_blues)
            if area > 1800:
                return True
        return False
    
    def process_image(self, frame):
        """处理图像，进行颜色检测和方框标记"""
        color = "未知"
        angle = ""
        display_frame = frame.copy()
        
        # 自动检测时使用方框搜索
        if self.auto_detect:
            # 定义检测区域（中心区域的方框）
            h, w = frame.shape[:2]
            roi_size = min(h, w) // 2
            x1, y1 = (w - roi_size) // 2, (h - roi_size) // 2
            x2, y2 = x1 + roi_size, y1 + roi_size
            
            # 在图像上绘制检测方框
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 提取ROI区域进行处理
            roi = frame[y1:y2, x1:x2]
            
            # 进行颜色检测
            frame_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
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
            
            # 根据检测结果设置颜色
            if result_red:
                color = "红色"
                # 在方框周围绘制红色边框
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(display_frame, "检测到红色", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif result_blue:
                color = "蓝色"
                # 在方框周围绘制蓝色边框
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(display_frame, "检测到蓝色", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return display_frame, color, angle

# 创建全局实例供导入使用
color_detector = ColorDetection()



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

error = [0]

def ORBMatcher(img1, img2):
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
            error.append(error1)
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
    # 定义最小匹配点数量
    MIN_MATCH_COUNT = 30  #保证鲁棒性，为RANSAC提供足够内点
    if len(points_good) > MIN_MATCH_COUNT:
        # 获取匹配点的坐标
        src_array = np.float32([keypoints1[m.queryIdx].pt for m in points_good]).reshape(-1, 1, 2)
        dst_array = np.float32([keypoints2[m.trainIdx].pt for m in points_good]).reshape(-1, 1, 2)
        # 计算单应性矩阵
        M, mask = cv2.findHomography(src_array, dst_array, cv2.RANSAC, 5.0)
        # 计算旋转角度
        theta = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
        matchesMask = mask.ravel().tolist()
        h = img1.shape[0]
        w = img1.shape[1]
        # 定义四个角点的坐标
        array = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = np.array([[50, 50], [200, 50], [50, 200], [200, 200]], dtype=np.int32)
        # 进行透视变换
        dst = cv2.perspectiveTransform(array, M)
        # 在图像2上绘制多边形
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 6, cv2.LINE_AA)
        # 定义绘制匹配点的参数
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
        # 在图像上显示旋转角度和误差值
        cv2.putText(img3, f"angle:{int(theta)}", (210, 70), cv2.FONT_HERSHEY_PLAIN, 3, (128, 0, 128), 3)
        cv2.putText(img3, str("error:" + str(error[-1] )+ 'mm'), (500, 70), cv2.FONT_HERSHEY_PLAIN, 3, (128, 0, 128), 3)
        # 显示匹配结果图像
        cv2.imshow("display", img3)
        return True
    else:
        cv2.imshow("display", img2)
        return False

bool_result = True

# 创建多个窗口并设置窗口大小和位置
cv2.namedWindow("p1", cv2.WINDOW_NORMAL)
cv2.namedWindow("p2", cv2.WINDOW_NORMAL)
cv2.namedWindow("p3", cv2.WINDOW_NORMAL)
cv2.namedWindow("display",cv2.WINDOW_NORMAL)

cv2.moveWindow("display",110,20)
cv2.resizeWindow("display",600,630)
cv2.moveWindow("p1", 110, 660)
cv2.resizeWindow("p1", 200, 200)
cv2.moveWindow("p2", 310, 660)
cv2.resizeWindow("p2", 200, 200)
cv2.moveWindow("p3", 510, 660)
cv2.resizeWindow("p3", 200, 200)


# 滑动条的回调函数，不做任何操作
def nothing(x):
    pass

# 读取多张图像
img1 = cv2.imread("1.jpg")
img4 = cv2.imread("2.jpg")
img5 = cv2.imread("3.jpg")



cv2.imshow("p1", img1)
cv2.imshow("p2", img4)
cv2.imshow("p3", img5)


# 打开摄像头
capture = cv2.VideoCapture(0)



# 循环读取摄像头数据
while capture.isOpened():
    # 读取摄像头图像
    ret, img_camera = capture.read()
    #print(ret)
    #cv2.imshow("display", img_camera)
    if (ret):
        cv2.waitKey(20)

        current_img = None
        images = [img4, img1, img5]
        if current_img is None:
            for img in images:
                # 进行ORB特征匹配
                result = ORBMatcher(img, img_camera)
                if result:
                    current_img = img
                    break
        if current_img is not None:
            result = ORBMatcher(current_img, img_camera)
