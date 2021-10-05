import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte

path = './images/office.jpg'
img = io.imread(path)


def equalize_channel(img: ndarray) -> ndarray:
    h, w = img.shape
    n = h * w
    res = img.copy()
    sum_p = 0

    for i in range(1, 255):
        idx = np.where(img == i)
        sum_p += len(img[idx])
        s = 255 / n * sum_p
        res[idx] = s

    res.astype(np.int8)
    return res

def image_histogram_equalization(image, number_bins=256):
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized = np.divide(image_equalized, 255)

    return image_equalized.reshape(image.shape)


def equalize(img: ndarray) -> ndarray:
    if img.ndim == 3:
        print('three channels')
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        # r_out = equalize_channel(r)
        # g_out = equalize_channel(g)
        # b_out = equalize_channel(b)
        r_out = image_histogram_equalization(r)
        g_out = image_histogram_equalization(g)
        b_out = image_histogram_equalization(b)
        return np.dstack((r_out,g_out,b_out))
    else:
        return image_histogram_equalization(img)

out = equalize(img)

if img.ndim == 3:
    plt.subplot(3, 2, 1)
    plt.hist(img[:,:,0].ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.subplot(3, 2, 2)
    plt.hist(out[:,:,0].ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.subplot(3, 2, 3)
    plt.hist(img[:,:,1].ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.subplot(3, 2, 4)
    plt.hist(out[:,:,1].ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.subplot(3, 2, 5)
    plt.hist(img[:,:,2].ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.subplot(3, 2, 6)
    plt.hist(out[:,:,2].ravel(), bins=255, rwidth=0.8, range=(0, 255))

plt.show()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(out)
plt.show()