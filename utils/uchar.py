from queue import Queue

import cv2 as cv
import numpy as np

import utils.uimg as uimg


def to_size(img, new_height, new_width):
    """
    text area in the `img` is called `text_img`, the target is to scale `text_img` to `(new_height, new_width)`
    - width > new_width and height > new_height
    :param img:
    :param new_height:
    :param new_width:
    :return:
    """
    top, left, bottom, right = uimg.get_bounds(img, foreground_color='black')
    text_img = img[:, left:right]
    print("top: "+str(top) + "bottom: " + str(img.shape[0] - bottom))
    if 0 in text_img.shape or None in (top, left, bottom, right):
        # 宽或高为0
        return None
    text_img = uimg.pad_around(text_img, int(1.4*img.shape[0]), int(1.4*(right-left)), 255)
    out_img = uimg.fit_resize(text_img, new_height, new_width)
    return out_img


# width就是像素比例，如果是64*64，这里width就输入64
def contains_text(img, width):
    return True
    h, w = img.shape[:2]

    def is_foreground(_y, _x):
        return img[_y][_x] == 0

    def is_in(_y, _x):
        return 0 <= _y < h and 0 <= _x < w

    visited = {}
    components = []
    for y in range(h):
        for x in range(w):
            if (y, x) in visited:
                continue
            elif not is_foreground(y, x):
                visited[y, x] = True
            else:
                # not visited and is foreground (text)
                cnt = 0
                q = Queue()
                q.put((y, x))
                while not q.empty():
                    cnt += 1
                    cur_y, cur_x = q.get()
                    visited[cur_y, cur_x] = True
                    nbrs = [(cur_y - 1, cur_x), (cur_y + 1, cur_x), (cur_y, cur_x - 1), (cur_y, cur_x + 1)]
                    for nbr_y, nbr_x in nbrs:
                        if (nbr_y, nbr_x) in visited:
                            continue
                        if is_in(nbr_y, nbr_x):
                            visited[nbr_y, nbr_x] = True

                        if is_in(nbr_y, nbr_x) and is_foreground(nbr_y, nbr_x):
                            q.put((nbr_y, nbr_x))
                components.append(cnt)
    try:
        min_num = int((width * width) * 36 / (64 * 64))
    except ValueError:
        print("ValueError while computing the min_num of text")
    # todo 根据`components`判断该图片是否包含文字，返回值改为True|False
    return max_num(components) > min_num


def max_num(array):
    max = 0
    for num in array:
        if num > max:
            max = num
    return max


# ascending == 1,ascending;ascending == -1,descending
def top_ten(array, ascending):
    top = []
    for i in array:
        sign = 0
        if ascending == 1:
            for j in top:
                if j > i:
                    sign = 1
                    top.insert(top.index(j), i)
                    break
        elif ascending == -1:
            for j in top:
                if j < i:
                    sign = 1
                    top.insert(top.index(j), i)
                    break
        if sign == 0:
            top.append(i)
        if len(top) == 11:
            top.pop()
    return top


if __name__ == '__main__':
    # _img = uimg.read('/home/stone/PycharmProjects/universal_cnn/4.jpg')
    # # get_bounds(img)
    # to_size(_img, 64, 64)

    maxLinked = []
    import os

    print(os.getcwd())
    count = 0
    count1 = 0
    # for filename in os.listdir('char_split_output/822_1169'):
    #     im = uimg.read(os.path.join('char_split_output/822_1169', filename))
    for filename in os.listdir('J:/labels/labels/positive'):
        im = uimg.read(os.path.join('J:/labels/labels/positive', filename))
        _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        coms = contains_text(im, 64)
        count = count + 1
        if coms:
            count1 = count1 + 1
    print(count1 / count)

    count = 0
    count1 = 0
    maxLinked2 = []
    for filename in os.listdir('J:/labels/labels/negative'):
        im = uimg.read(os.path.join('J:/labels/labels/negative', filename))
        _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        coms = contains_text(im, 64)
        count = count + 1
        if coms:
            count1 = count1 + 1
    print(count1 / count)
