import matplotlib.pyplot as plt
import numpy as np
# import cv2
# import scipy.ndimage
from numpy import ndarray

def merge(img, func_channel, *args):
    assert img.ndim == 3
    if img.ndim == 3:
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        r_out = func_channel(r)
        g_out = func_channel(g)
        b_out = func_channel(b)
    return np.dstack((r_out,g_out,b_out))


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
        return merge(img, equalize_channel)
    else:
        return equalize_channel(img)


def denoise_channel(img):
    pad_num = 1
    image_pad = img.copy()
    image_pad = np.pad(image_pad, (pad_num, pad_num), mode='constant')
    w, h = image_pad.shape
    out = image_pad.copy()

    for i in range(pad_num, w - pad_num):
        for j in range(pad_num, h - pad_num):
            out[i, j] = np.median(image_pad[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1])

    out = out[pad_num:w - pad_num, pad_num:h - pad_num]
    return out

# def denoise_scipy(img):
#     img = scipy.ndimage.median_filter(img, (3, 3))
#     return img

# def denoise_opencv(img):
#     img = np.multiply(img, 255).astype(np.uint8)
#     img = cv2.medianBlur(img, 3)
#     return np.divide(img.astype(np.float32), 255)

def denoise(img: ndarray) -> ndarray:
    if img.ndim == 3:
        return merge(img, denoise_channel)
    else:
        return denoise_channel(img)


def interpolate_channel(img: ndarray):
  nR0 = len(img)     # source number of rows
  nC0 = len(img[0])  # source number of columns
  nR = img.shape[0] * 2
  nC = img.shape[1] * 2
  return np.array([[ img[int(nR0 * r / nR)][int(nC0 * c / nC)]  for c in range(nC)] for r in range(nR)])

def interpolate(img: ndarray) -> ndarray:
    if img.ndim == 3:
        return merge(img, interpolate_channel)
    else:
        return interpolate_channel(img)


def dft_ori(img):
    return np.fft.fftshift(np.fft.fft2(img))

def dft(img: ndarray) -> ndarray:
    return np.log(np.abs(dft_ori(img)))


def butterworth(img: ndarray) -> ndarray:
    h, w = img.shape
    M, N = h // 2, w // 2
    F = dft_ori(img)
    d0 = (h + w) // 10
    H = np.zeros_like(F)
    # t = np.array(np.meshgrid(range(h),range(w))).transpose([2,1,0])
    # f = lambda x: 1 / (1 + (np.linalg.norm((x[0] - M, x[1] - N)) / d0) ** 4)
    # t = f(t)

    for i in range(h):
        for j in range(w):
            d = np.linalg.norm((i - M, j - N))
            H[i, j] = 1 / (1 + (d / d0) ** 4)
    G = np.multiply(F, H)

    # return np.log(np.abs(G))
    return np.abs(np.fft.ifft2(np.fft.ifftshift(G)))

def canny(img: ndarray) -> ndarray:
    return None


def morphology(img: ndarray) -> ndarray:
    return None
