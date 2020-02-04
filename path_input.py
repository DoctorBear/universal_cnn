#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import tkinter.font as font

import json
import os
import shutil
import yaml
from processing.single_char_processing import Processor
import frozen_dir

# for text_detector


# for data.py
import gc
import re
import cv2 as cv
import numpy as np
from progressbar import ProgressBar

# for main.py
import tensorflow as tf

# for utext.py
from collections import deque
from math import sqrt
from typing import List, Tuple

# for utable.py
import time

# for uimg.py
from matplotlib import pyplot as plt

# for uchar.py
from queue import Queue

# for rectification.py:
from typing import Generator

# for test_char.py
import argparse
import json

# for single_char_processing
import threading

from multiprocessing import *

# with open(frozen_dir.app_path()+'configs/infer.yaml', encoding='utf-8') as conf_file:
#     conf_args = yaml.load(conf_file)
from text_detector.processing.ctpn_processor import CtpnProcessor
from utils.uimg import pad_right_and_below

CONF = {**{
    'ctpn_ckpt_dir': frozen_dir.app_path()+"all/ctpn_ckpt/",
    'ctpn_output_path': frozen_dir.app_path()+'ctpn_output/',
    'charmap_path': frozen_dir.app_path()+"all/all_4190/all_4190.json",
    'aliasmap_path': frozen_dir.app_path()+"all/all_4190/aliasmap.json",
    'ckpt_dir': frozen_dir.app_path()+"all/all_4190/ckpts/",
    'input_height': 64,
    'input_width': 64,
    'num_class': 4190,
    'batch_size': 64,
    'p_thresh': 0.8
}}
for key, value in CONF.items():
    CONF[key] = frozen_dir.unify_sep(value)

print(json.dumps(CONF, ensure_ascii=False, indent=2))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
text_detector = CtpnProcessor().load(CONF['ctpn_ckpt_dir'])
proc = Processor(CONF['charmap_path'], CONF['aliasmap_path'], CONF['ckpt_dir'],
                 CONF['input_height'], CONF['input_width'], CONF['num_class'], CONF['batch_size'])


def get_images(input_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif']
    # os.walk will recursively go through all nodes(directories) in this path
    for parent, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            for ext in exts:
                if filename.lower().endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


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


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.entry_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.file_names = None
        self.dir_name = None
        self.create_widgets()

    def create_widgets(self):
        self.title('OCR Transfer')
        self.geometry('500x300')

        lb0 = Label(self, text='文字识别', bg='#3385ff', fg='white', font=('华文新魏', 24), width=10, height=2)
        lb0.pack(pady=5)
        top_frame = tk.Frame(self, height=80)
        next_frame = tk.Frame(self, height=80)
        top_frame.pack(side=tk.TOP, pady=5)
        next_frame.pack(side=tk.TOP, pady=5)

        # TODO use the fileDialog instead of direct input to ensure correct input
        input_label = Label(top_frame, text='输入路径:', font=('幼圆', 14, font.BOLD))
        input_entry = Entry(top_frame, textvariable=self.entry_var, font=('Arial', 14))
        input_button = Button(top_frame, command=self.__openfile, text='选择')
        input_label.grid(row=0, column=0, sticky=tk.W)
        input_entry.grid(row=0, column=1)
        input_button.grid(row=0, column=2)
        # var_usr_name1 = askopenfilenames(title="请选择一个JPG或者PNG图片文件", parent=input_entry,
        #                                  filetypes=[("JPG图片文件", "*.jpg"), ("PNG图片文件", "*.png")])
        # input_entry.bind('<Return>', func=self.__refresh)

        output_label = Label(next_frame, text='输出路径:', font=('幼圆', 14, font.BOLD))
        output_entry = Entry(next_frame, textvariable=self.output_var, font=('Arial', 14))
        output_button = Button(next_frame, command=self.__opendir, text='选择')
        output_label.grid(row=0, column=0, sticky=tk.W)
        output_entry.grid(row=0, column=1)
        output_button.grid(row=0, column=2)

        submit_button = Button(self, command=self.__submit, text='提交运行', font=('幼圆', 14, font.BOLD))
        submit_button.pack()

    def __openfile(self):
        self.file_names = filedialog.askopenfilenames(title="请选择一个JPG或者PNG图片文件",
                                                      filetypes=[("JPG图片文件", "*.jpg"), ("PNG图片文件", "*.png")])
        self.entry_var.set(self.file_names)
        if not self.file_names:
            messagebox.showwarning('警告', message='未选择文件')

    def __opendir(self):
        # 打开文件夹的逻辑
        self.dir_name = filedialog.askdirectory()  # 打开文件夹对话框
        self.output_var.set(self.dir_name)  # 设置变量output_var，等同于设置部件Entry
        if not self.dir_name:
            messagebox.showwarning('警告', message='未选择文件夹！')  # 弹出消息提示框

    def __submit(self):
        if (not self.file_names) or str(self.entry_var.get()) == '':
            messagebox.showwarning('警告', message='未选择文件')
        elif (not self.dir_name) or str(self.output_var.get()) == '':
            messagebox.showwarning('警告', message='未选择文件夹！')  # 弹出消息提示框
        else:
            file_list = self.entry_var.get()
            output_dir = self.output_var.get()
            file_list = file_list[1:-1]
            print(file_list)
            files = file_list.strip().split(",")
            if files[-1] == '':
                files = files[:-1]
            sign = 0
            for index, filename in enumerate(files):
                filename = filename.strip()
                print(filename)
                files[index] = filename[1:-1]
                if not os.path.isfile(files[index]):
                    messagebox.showwarning('警告', message=files[index]+'文件找不到路径！')
                    sign = 1
                    break
            if not os.path.isdir(output_dir):
                sign = 1
                messagebox.showwarning('警告', message=output_dir+'目录找不到路径！')
            if sign == 0:
                self.__recognize(files, output_dir)

    @staticmethod
    def __recognize(files, target_dir):
        print(files)
        for path in files:
            rs = proc.get_text_result(path, p_thresh=CONF['p_thresh'],
                                      auxiliary_img=frozen_dir.app_path()+'static/auxiliary.jpg',
                                      remove_lines=1)
            name = target_dir+'/'+path.split('/')[-1][:-4]+'.txt'
            print(name)
            f = open(name, 'w+')
            f.write(rs)
            f.close()


if __name__ == '__main__':
    # 可以考虑添加在线修改功能，显示对应坐标的图中位置和文本，并实时修正，以及删除相应文本框和添加文本框并识别
    # # 实例化Application
    # app = Application()
    # # 主消息循环:
    # app.mainloop()

    # if os.path.exists(CONF['ctpn_output_path']):
    #     shutil.rmtree(CONF['ctpn_output_path'])
    # os.makedirs(CONF['ctpn_output_path'])

    input_path = 'text_detector/data/demo/'

    im_fn_list = get_images(input_path)

    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        start = time.time()
        try:
            im = cv.imread(im_fn)[:, :, ::-1]
        except:

            print("Error reading image {}!".format(im_fn))
            continue

        img = pad_image(im)

        resize_time = time.time()
        print("Resize cost time: {:.2f}s".format(resize_time - start))

        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        boxes, scores = text_detector.detect(img, im_info)
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
        cv.imwrite(os.path.join(CONF['ctpn_output_path'], os.path.basename(im_fn)), panel[:, :, ::-1])

        with open(os.path.join(CONF['ctpn_output_path'], os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                  "w") as f:
            for i, box in enumerate(actual_boxes):
                line = ",".join(str(box[k]) for k in range(4))
                line += "," + str(scores[i]) + "\r\n"
                f.writelines(line)

    sep = os.path.sep
    test_data_path = CONF['ctpn_output_path']


    files = []
    exts = ['jpg', 'png', 'jpeg']
    # os.walk will recursively go through all nodes(directories) in this path
    for parent, dirnames, filenames in os.walk(test_data_path):
        print(filenames)
        print(parent)
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    name_part = filename[:len(filename)-len(ext)-1]
                    files.append((os.path.join(parent, filename), os.path.join(parent, name_part+'.txt')))
                    break
    print(files)

    output_dir = 'test_data'
    cache_dir = 'static'
    for i in os.listdir(output_dir):
        file_data = output_dir + sep + i
        if os.path.isfile(file_data):
            os.remove(file_data)
    for i in os.listdir(cache_dir):
        file_data = cache_dir + sep + i
        if os.path.isfile(file_data):
            os.remove(file_data)

    for file in files:
        print(file[0])
        name_part = str(file[1].split(sep)[-1][:-4])
        print(name_part)
        output_path = os.path.join(output_dir, name_part+'.txt')
        with open(output_path, "w") as f:
            boxes = []
            for line in open(file[1], 'r'):
                box = list(map(lambda x: int(x), line.split(',')[:-1]))
                if box:
                    boxes.append(box)
            # todo 暂时是以y为排序，但是实际上不大合理，应该专门写一个方法对box进行排序
            # box需要以下信息：left, right, height*std_width
            boxes.sort(key=lambda l: l[1])
            for box in boxes:
                if box:
                    rs = proc.get_text_result(file[0], p_thresh=CONF['p_thresh'],
                                              auxiliary_img=frozen_dir.app_path()+'static'+sep+str(box)+'.jpg',
                                              box=box,
                                              remove_lines=1)
                    print(box)
                    f.writelines(str(box)+rs+'\n')









