import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def plane(img: ndarray) -> ndarray:
    assert img.ndim == 2
    planes = []
    mask = 0b1 << 7
    for i in range(9):
        planes.append(np.left_shift(np.bitwise_and(img, mask), i))
        mask = mask >> 1
    layer1 = np.hstack(planes[0:3])
    layer2 = np.hstack(planes[3:6])
    layer3 = np.hstack(planes[6:9])
    res = np.vstack((layer1, layer2, layer3))
    return res


def equalize(img: ndarray) -> ndarray:
    return None


def denoise(img: ndarray) -> ndarray:
    return None


def interpolate(img: ndarray) -> ndarray:
    return None


def dft(img: ndarray) -> ndarray:
    return None


def butterworth(img: ndarray) -> ndarray:
    return None


def canny(img: ndarray) -> ndarray:
    return None


def morphology(img: ndarray) -> ndarray:
    return None
