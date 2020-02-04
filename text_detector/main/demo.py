# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import math
import tensorflow as tf

from text_detector.processing.ctpn_processor import CtpnProcessor
from utils.uimg import pad_right_and_below

from text_detector.nets import model_train as model
from text_detector.utils.rpn_msr.proposal_layer import proposal_layer
from text_detector.utils.text_connector.detectors import TextDetector

sys.path.append(os.getcwd())
print(sys.path)

test_data_path = '../data/demo/'
output_path = '../data/res/'
tf.app.flags.DEFINE_string('gpu', '-1', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../ctpn_ckpt/', '')
FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'bmp']
    # os.walk will recursively go through all nodes(directories) in this path
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.lower().endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


# todo Replace resize below with cut to avoid lose of pixels
def pad_image(img, filter_size=600, stride=400):
    '''
    Cut just like filter in cnn, size of filter and stride is needed first!
    The overlap is needed to avoid character cut!
    If one side is not equal to 600 + 400 * n(n>=0), pad it too.
    :param img: picure
    :param filter_size: size of images after cut
    :param stride: just like stride in cnn
    :return images and their offset(height and width)
    '''
    height, width = img.shape[0:2]
    new_h = height if height // 16 == 0 else (height // 16 + 1) * 16
    new_w = width if width // 16 == 0 else (width // 16 + 1) * 16
    re_im = pad_right_and_below(img, new_height=new_h, new_width=new_w)
    return re_im


# todo Replace resize with cut to avoid lose of pixels
def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    # 在较大边不超过1200的情况下，按照较小边到600缩放,较小边=600，600<=较大边<1200
    # 否则按照较大变1200缩放，即较大边=1200,较小边<=600
    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)

    # 当需要缩小时，返回False
    # if im_scale < 1:
    #     return False

    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    # 上取整两边至整除16
    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    print(new_h, end=':')
    print(new_w)

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imencode('.jpg', re_im)[1].tofile('data/res/1.jpg')
    return img, (1, 1)
    # return re_im, (new_h / img_size[0], new_w / img_size[1])


def main(argv=None):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    main = CtpnProcessor().load(FLAGS.checkpoint_path)
    im_fn_list = get_images()
    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        start = time.time()
        try:
            im = cv2.imread(im_fn)[:, :, ::-1]
        except:

            print("Error reading image {}!".format(im_fn))
            continue

        img = pad_image(im)

        # resize_time = time.time()
        # print("Resize cost time: {:.2f}s".format(resize_time - start))

        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        boxes, scores = main.detect(img, im_info)
        h1, w1, c1 = im.shape
        panel = np.ones((h1, w1, c1), np.uint8)*255

        actual_boxes = []

        for i, box in enumerate(boxes):

            #  not sure whether there should be padding, padding may let the noise in,
            #  but without padding, part of the valuable info may be lost
            actual = [box[0] - 5, box[1] - 2, box[2] + 6, box[5] + 2]
            actual[0] = int(actual[0]) if actual[0] > -1 else 0
            actual[1] = int(actual[1]) if actual[1] > -1 else 0
            actual[2] = int(actual[2]) if actual[2] < w1 else w1-1
            actual[3] = int(actual[3]) if actual[3] < h1 else h1-1

            panel[actual[1]:actual[3]+1, actual[0]:actual[2]+1, :] = im[actual[1]:actual[3]+1, actual[0]:actual[2]+1, :]

            actual_boxes.append(actual)
            # cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
            #               thickness=2)
        # img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), panel[:, :, ::-1])

        with open(os.path.join(output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                  "w") as f:
            for i, box in enumerate(actual_boxes):
                line = ",".join(str(box[k]) for k in range(4))
                line += "," + str(scores[i]) + "\r\n"
                f.writelines(line)


if __name__ == '__main__':
    tf.app.run()
