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

def equalize_channel(image, number_bins=256):
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

def equalize(img: ndarray) -> ndarray:
    if img.ndim == 3:
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        r_out = equalize_channel(r)
        g_out = equalize_channel(g)
        b_out = equalize_channel(b)
        return np.dstack((r_out,g_out,b_out))
    else:
        return equalize_channel(img)


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
