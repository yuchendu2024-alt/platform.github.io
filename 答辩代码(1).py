import cv2
import numpy as np
import time

lower_red1 = np.array([0, 30, 20])
lower_red2 = np.array([160, 30, 20])
lower_blue = np.array([90, 80, 50])
higher_red1 = np.array([10, 255, 150])
higher_red2 = np.array([180, 255, 150])
higher_blue = np.array([130, 255, 255])

kernel = np.ones((3, 3), np.float32)


def red_detect(contours_red):
    # 遍历所有红色轮廓
    for contours_reds in contours_red:
        # 计算轮廓的面积
        area = cv2.contourArea(contours_reds)
        # 如果面积大于3000，则认为检测到红色区域
        if (area > 1000):
            return True
    return False


def blue_detect(contours_blue):
    # 遍历所有蓝色轮廓
    for contours_blues in contours_blue:
        # 计算轮廓的面积
        area = cv2.contourArea(contours_blues)
        # 如果面积大于1，则认为检测到蓝色区域
        if (area > 1000):
            return True
    return False


def ORBMatcher(img1, img2):
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
    result_red = red_detect(contours_red)
    result_blue = blue_detect(contours_blue)

    if result_blue:
        result_red = False

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
    # 定义最小匹配点数量
    MIN_MATCH_COUNT = 30
    if len(points_good) > MIN_MATCH_COUNT:
        # 获取匹配点的坐标
        src_array = np.float32([keypoints1[m.queryIdx].pt for m in points_good]).reshape(-1, 1, 2)
        dst_array = np.float32([keypoints2[m.trainIdx].pt for m in points_good]).reshape(-1, 1, 2)
        # 计算单应性矩阵
        M, mask = cv2.findHomography(src_array, dst_array, cv2.RANSAC, 8.0)
        # 计算旋转角度
        angle_history = []
        SMOOTH_WINDOW = 20
        theta = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
        angle_history.append(theta)
        if len(angle_history) > SMOOTH_WINDOW:
            angle_history.pop(0)
        smooth_theta = np.average(angle_history)
        last_angle=None
        ANGLE_THRESHOLD=2
        if last_angle is None or abs(smooth_theta - last_angle) > ANGLE_THRESHOLD:
            last_angle = smooth_theta

        matchesMask = mask.ravel().tolist()

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
        cv2.putText(img3, f"angle:{int(smooth_theta)}", (210, 70), cv2.FONT_HERSHEY_PLAIN, 3, (128, 0, 128), 3)

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
cv2.namedWindow("display", cv2.WINDOW_NORMAL)

cv2.moveWindow("display", 110, 20)
cv2.resizeWindow("display", 600, 630)
cv2.moveWindow("p1", 110, 660)
cv2.resizeWindow("p1", 200, 200)
cv2.moveWindow("p2", 310, 660)
cv2.resizeWindow("p2", 200, 200)
cv2.moveWindow("p3", 510, 660)
cv2.resizeWindow("p3", 200, 200)

# 读取多张图像
img1 = cv2.imread("1.jpg")
img4 = cv2.imread("2.jpg")
img5 = cv2.imread("3.jpg")

cv2.imshow("p1", img1)
cv2.imshow("p2", img4)
cv2.imshow("p3", img5)

# 打开摄像头
capture = cv2.VideoCapture(1)
if not capture.isOpened():
    print("错误：无法打开摄像头！")
    exit()
current_img = None
images = [img4, img1, img5]
max_retry = 10
retry_count = 0

try:
    while capture.isOpened():
        # 读取摄像头图像
        ret, img_camera = capture.read()
        if not ret:
            retry_count += 1
            print("警告：摄像头帧读取失败，重试中...")
            if retry_count > max_retry:
                capture.release()
                capture = cv2.VideoCapture(1)
                retry_count = 0
                time.sleep(1)
            continue
        retry_count = 0

        if current_img is None:
            for img in images:
                # 进行ORB特征匹配
                result = ORBMatcher(img, img_camera)
                if result:
                    current_img = img
                    break
        if current_img is not None:
            result = ORBMatcher(current_img, img_camera)
            key = cv2.waitKey(1)
            if key != -1:
                print(key)
                break

except KeyboardInterrupt:
    print("用户终止程序")