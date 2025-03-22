import numpy as np
import cv2

def pong_obs_preprocess(observation, agent_idx):
    """

    :param observation: pong环境observation是280*480*3的ndarray，需要降维。
    :return: 处理好的observation
    """
    bin_img = observation[:, :, 0]
    features = cv2.resize(bin_img, (38, 21))
    # 这里pong环境有问题，每个智能体观测到的都是全局观测，所以手动截取一些。
    if agent_idx == 0:
        features = features[:, :32]
    else:
        features = features[:, -32:]

    features = features.flatten()
    return (features - np.min(features)) / (np.max(features) - np.min(features))


def pursuit_obs_preprocess(observation):
    mean = observation.mean()
    std = observation.std()
    return (observation - mean) / std 


def spread_obs_preprocess(observation):
    observation = observation[:14]  # 智能体没有communication
    mean = observation.mean()
    std = observation.std()
    return (observation - mean) / std
