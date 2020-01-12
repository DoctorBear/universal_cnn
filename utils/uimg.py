import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def get_bounds(img, foreground_color='black'):
    """
    Get the minimum rectangle box containing the text. This method assumes that:
        1. there is no noise in the given image
    :param foreground_color:
    :param img: input image, of which the pixel value should either be `0` or `255`
    :return: top, left, bottom, right
    > FYI: image_height = bottom - top, image_width = right - left
    """
    assert foreground_color in ('white', 'black')
    h, w = img.shape
    # print(h, w)
    background_vertical = np.zeros((h,), np.uint8) if foreground_color == 'white' else np.ones((h,), np.uint8) * 255
    background_horizontal = np.zeros((w,), np.uint8) if foreground_color == 'white' else np.ones((w,), np.uint8) * 255
    for left in range(w):
        if not (img[:, left] == background_vertical).all():
            break
    else:
        left = None

    for right in range(w - 1, -1, -1):
        if not (img[:, right] == background_vertical).all():
            right += 1
            break
    else:
        right = None

    for top in range(h):
        if not (img[top, :] == background_horizontal).all():
            break
    else:
        top = None

    for bottom in range(h - 1, -1, -1):
        if not (img[bottom, :] == background_horizontal).all():
            bottom += 1
            break
    else:
        bottom = None

    return top, left, bottom, right


def save(file_path: str, img):
    """
    Do not use `cv2.imwrite` to save image, doesn't work as expected when it comes to Chinese file name
    :param file_path:
    :param img:
    :return:
    """
    cv.imencode('.jpg', img)[1].tofile(file_path)


def read(file_path: str, read_flag=0):
    return cv.imdecode(np.fromfile(file_path, dtype=np.uint8), read_flag)


def show(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def reverse(img):
    """
    :param img: channels = 1, dtype = numpy.uint8
    :return: val = 255 - val, pixel-wise
    """
    img = 255 - img
    return img


def auto_bin(img, otsu: bool = False):
    """
    :param otsu: use OTSU instead of Adaptive
    :param img:
    :return:
    """
    # 读取图像，并转为灰度图
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img

    if otsu:
        return cv.threshold(img_grey, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # 自适应二值化
    img_at_mean = cv.adaptiveThreshold(img_grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 47, 10)
    return img_at_mean


# 固定阈值二值化
def thresh_bin(img, minv=150, maxv=255, type=cv.THRESH_BINARY_INV):
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
    _, threshold = cv.threshold(img_grey, minv, maxv, type)
    return threshold


# 边缘检测
def canny_bin(img, minv=50, maxv=150, apertureSize=3):
    img_canny = cv.Canny(img, minv, maxv, apertureSize)
    return img_canny


# 腐蚀
def erode_img(img, kernal_heght, kernal_width):
    kernel = np.uint8(np.zeros((kernal_heght, kernal_width)))
    img = cv.erode(img, kernel, iterations=2)
    return img


def fit_resize(img, new_height, new_width):
    """
    > ***ALERT `fit_resize` may return `None` value***, this is because sometimes an image can be fit_resize
    into another size( the calculated new height or width can be 0)
    :param img:
    :param new_height:
    :param new_width:
    :return:
    """
    height, width = img.shape[:2]
    if height == 0 or width == 0:
        return None
    h_rate = height / new_height
    w_rate = width / new_width
    rate = max(h_rate, w_rate)
    dsize = int(width / rate), int(height / rate)
    try:
        resized_img = auto_bin(cv.resize(img, dsize), otsu=True)
            # if rate >= 1 else img
        return pad_to(resized_img, new_height, new_width, 255)
    except cv.error:
        return None


def pad_to(img, new_height, new_width, padding_val):
    height, width = img.shape[:2]
    assert height <= new_height and width <= new_width
    top = (new_height - height) // 2
    bottom = new_height - height - top
    left = (new_width - width) // 2
    right = new_width - width - left
    out_img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=padding_val)
    return out_img


if __name__ == '__main__':
    import random as rd

    for i in range(20):
        h, w = rd.randint(30, 80), rd.randint(30, 80)
        origin = np.ones((h, w), dtype=np.uint8) * 255
        fit = fit_resize(origin, 64, 64)
        out = pad_to(fit, 64, 64, 100)
        save("/usr/local/src/data/%d_%dx%d.png" % (i, h, w), out)
