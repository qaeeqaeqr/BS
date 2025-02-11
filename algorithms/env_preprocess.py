import numpy as np
import cv2

def pong_obs_preprocess(observation):
    """

    :param observation: pong环境observation是280*480*3的ndarray，需要降维。
    :return: 处理好的observation
    """
    bin_img = observation[:, :, 0]
    compressed_img = cv2.resize(bin_img, (24, 14))

    features = compressed_img.flatten()
    return (features - np.min(features)) / (np.max(features) - np.min(features))


def pursuit_obs_preprocess(observation):
    mean = observation.mean()
    std = observation.std()
    return (observation - mean) / std 
