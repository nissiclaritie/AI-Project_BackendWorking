from PIL import Image
import numpy as np
from urllib.request import urlopen


def refsize_image(data):
    print(data.shape)
    refpos = (50, 50)  # top left corner
    refsize = (32, 32)  # reference sample size
    sub = data[refpos[0]:refpos[0] + refsize[0], refpos[1]:refpos[1] + refsize[1]]
    # Image.fromarray(sub)

    c = list(np.mean(sub[:, :, i]) for i in range(3))

    wb = data.astype(float)
    for i in range(3): wb[:, :, i] /= c[i] / float(min(c))

    Image.fromarray(wb.astype(np.uint8))
    ycbcr = rgb2ycbcr(data)
    ysub = rgb2ycbcr(sub)
    yc = list(np.mean(ysub[:, :, i]) for i in range(3))

    for i in range(1, 3):
        ycbcr[:, :, i] = np.clip(ycbcr[:, :, i] + (128 - yc[i]), 0, 255)

    rgb = ycbcr2rgb(ycbcr)
    # Image.fromarray(rgb)
    return rgb


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr  # np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)