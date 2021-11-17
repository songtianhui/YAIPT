import numpy as np
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

def equalize(img: ndarray) -> ndarray:
    def equalize_channel(image, number_bins=256):
        # get image histogram
        image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = cdf / cdf[-1] # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape)


    if img.ndim == 3:
        return merge(img, equalize_channel)
    else:
        return equalize_channel(img)

def denoise(img: ndarray) -> ndarray:
    def gauss_denoise_channel(img):
        gauss_filter = (np.array([1,2,1,2,4,2,1,2,1]) / 16).reshape((3,3))

        pad_num = 1
        image_pad = img.copy()
        image_pad = np.pad(image_pad, (pad_num, pad_num), mode='constant')
        w, h = image_pad.shape
        out = image_pad.copy()

        for i in range(pad_num, w - pad_num):
            for j in range(pad_num, h - pad_num):
                out[i, j] = np.multiply(image_pad[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1], gauss_filter).sum()

        out = out[pad_num:w - pad_num, pad_num:h - pad_num]
        return out


    if img.ndim == 3:
        return merge(img, gauss_denoise_channel)
    else:
        return gauss_denoise_channel(img)

def interpolate(img: ndarray) -> ndarray:
    def interpolate_channel(img: ndarray):
        nR0 = len(img)     # source number of rows
        nC0 = len(img[0])  # source number of columns
        nR = img.shape[0] * 2
        nC = img.shape[1] * 2
        return np.array([[ img[int(nR0 * r / nR)][int(nC0 * c / nC)]  for c in range(nC)] for r in range(nR)])


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
    def gauss(img):
        return denoise(img)

    def gradient(img):
        sobely = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape((3,3))
        sobelx = sobely.transpose()
        pad_num = 1
        image_pad = img.copy()
        image_pad = np.pad(image_pad, (pad_num, pad_num), mode='constant')
        w, h = image_pad.shape
        grad_x = image_pad.copy()
        grad_y = image_pad.copy()

        for i in range(pad_num, w - pad_num):
            for j in range(pad_num, h - pad_num):
                grad_x[i, j] = np.multiply(sobelx, image_pad[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]).sum()
                grad_y[i, j] = np.multiply(sobely, image_pad[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]).sum()

        grad_x = grad_x[pad_num:w - pad_num, pad_num:h - pad_num] + 0.00000001 # avoid div by zero
        grad_y = grad_y[pad_num:w - pad_num, pad_num:h - pad_num]
        
        # m = np.sqrt(dx**2 + dy**2)
        m = np.abs(grad_x) + np.abs(grad_y)
        theta = np.arctan(grad_y / grad_x)
        return m, theta

    def non_maximum_supress(grad, theta):
        theta = theta + np.pi / 2
        pad_num = 1
        image_pad = grad.copy()
        image_pad = np.pad(image_pad, (pad_num, pad_num), mode='constant')
        w, h = image_pad.shape
        out = image_pad.copy()

        for i in range(pad_num, w - pad_num):
            for j in range(pad_num, h - pad_num):
                if (theta[i-pad_num, j-pad_num] >= 0 and theta[i-pad_num, j-pad_num] < np.pi / 8) or (theta[i-pad_num, j-pad_num] >= 7 * np.pi / 8 and theta[i-pad_num, j-pad_num] < np.pi):
                    out[i, j] = image_pad[i, j] if image_pad[i, j] >= image_pad[i-1, j] and image_pad[i, j] >= image_pad[i+1, j] else 0
                elif (theta[i-pad_num, j-pad_num] >= np.pi / 8 and theta[i-pad_num, j-pad_num] < 3 * np.pi / 8):
                    out[i, j] = image_pad[i, j] if image_pad[i, j] >= image_pad[i+1, j-1] and image_pad[i, j] >= image_pad[i-1, j+1] else 0
                elif (theta[i-pad_num, j-pad_num] >= 3 * np.pi / 8 and theta[i-pad_num, j-pad_num] < 5 * np.pi / 8):
                    out[i, j] = image_pad[i, j] if image_pad[i, j] >= image_pad[i, j-1] and image_pad[i, j] >= image_pad[i, j+1] else 0
                else:
                    out[i, j] = image_pad[i, j] if image_pad[i, j] >= image_pad[i+1, j+1] and image_pad[i, j] >= image_pad[i-1, j-1] else 0

        out = out[pad_num:w - pad_num, pad_num:h - pad_num]
        return out

    def double_threshold(grad):
        ht = 0.3
        lt = 0.1
        pad_num = 1
        image_pad = grad.copy()
        image_pad = np.pad(image_pad, (pad_num, pad_num), mode='constant')
        w, h = image_pad.shape
        out = image_pad.copy()

        for i in range(pad_num, w - pad_num):
            for j in range(pad_num, h - pad_num):
                if image_pad[i, j] >= ht:
                    out[i, j] = 1
                elif image_pad[i, j] < lt:
                    out[i, j] = 0
                else:
                    out[i, j] = 1 if (np.array([image_pad[i-pad_num, j], image_pad[i+pad_num, j], image_pad[i, j+pad_num], image_pad[i, j-pad_num]]) >= ht).any() else 0
        
        out = out[pad_num:w - pad_num, pad_num:h - pad_num]
        return out

    img_denoise = gauss(img)
    grad, theta = gradient(img_denoise)
    grad = non_maximum_supress(grad, theta)
    grad = double_threshold(grad)
    return grad


def morphology(img: ndarray) -> ndarray:
    def dilate(img, kernel):
        a = kernel.shape[0]
        pad_num = (a - 1) // 2
        image_pad = img.copy()
        image_pad = np.pad(image_pad, (pad_num, pad_num), mode='constant')
        w, h = image_pad.shape
        out = image_pad.copy()

        for i in range(pad_num, w - pad_num):
            for j in range(pad_num, h - pad_num):
                if np.multiply(image_pad[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1], kernel).sum() >= 1:
                    out[i, j] = 1

        out = out[pad_num:w - pad_num, pad_num:h - pad_num]
        return out
    
    def erode(img, kernel):
        a = kernel.shape[0]
        pad_num = (a - 1) // 2
        image_pad = img.copy()
        image_pad = np.pad(image_pad, (pad_num, pad_num), mode='constant')
        w, h = image_pad.shape
        out = image_pad.copy()

        for i in range(pad_num, w - pad_num):
            for j in range(pad_num, h - pad_num):
                if np.multiply(image_pad[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1], kernel).sum() < 4:
                    out[i, j] = 0

        out = out[pad_num:w - pad_num, pad_num:h - pad_num]
        return out


    img = (img > 0.5).astype(np.float32)
    kernel = np.array([0,1,0,1,0,1,0,1,0]).reshape((3,3))
    return dilate(erode(img, kernel), kernel)