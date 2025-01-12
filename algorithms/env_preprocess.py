import numpy as np
import cv2

def pong_obs_preprocess(observation):
    """

    :param observation: pong环境observation是280*480*3的ndarray，需要降维。
    :return: 处理好的observation
    """
    image = observation

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []

    # 制作feature
    for cnt in contours:
        # 计算中心点
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            continue
        # 计算面积
        area = cv2.contourArea(cnt)
        # 计算边界框
        x, y, w, h = cv2.boundingRect(cnt)
        # 将特征添加到列表
        features += [cx, cy, w, h, area]

    if len(features) == 0:
        features += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif len(features) == 5:
        features += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif len(features) == 10:
        features += [0, 0, 0, 0, 0]
    else:
        pass

    features = np.array(features)
    min_val = features.min()
    max_val = features.max()
    return (features - min_val) / (max_val - min_val)  # normalized feature.

def pursuit_obs_preprocess(observation):
    min_val = observation.min()
    max_val = observation.max()

    return (observation - min_val) / (max_val - min_val)
