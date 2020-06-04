#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import tkinter.font as font
import datetime
import time

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTImage, LTTextBox
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFPasswordIncorrect
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdocument import PDFPasswordIncorrect
import fitz
from binascii import b2a_hex
from PIL import Image

import json
import os
import shutil
import yaml
from processing.ocr_manager import OcrManager
from processing.processing_record import TaskRecord
import frozen_dir

# from processing_record
import wmi
import uuid

# from diff_leven
from bs4 import BeautifulSoup
import difflib

# for image preprocessing
from utils.imgs_transformer import process_gif
from utils.pdf_transformer import PDFTransformer

# for text_detector
from text_detector.processing import img_prepro, bbox_postpro

# for outputing files
import utils.files_manager as fm
import utils.output_generator as og

from pages.view import *  # 菜单栏对应的各个子页面

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
from typing import List, Tuple, Generator

# for utable.py
import time

# for uimg.py
from matplotlib import pyplot as plt

# for uchar.py
from queue import Queue


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

# for ctpn
from tensorflow.contrib import slim
from text_detector.nets import vgg
from text_detector.utils.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
import math

from distutils.core import setup
from Cython.Build import cythonize
import matplotlib.pyplot as plt

from tqdm import tqdm


SAVE_DIRS = {**{
    'ctpn_output_dir': frozen_dir.app_path()+'ctpn_output'+os.path.sep,
    # 输出目录需要由用户指定
    'temp_dir': frozen_dir.app_path()+'temp'+os.path.sep,
    'output_dir': frozen_dir.app_path()+'temp_output'+os.path.sep,
    'cache_dir': frozen_dir.app_path()+'static'+os.path.sep

}}
CONF = {**{
    'ctpn_ckpt_dir': frozen_dir.app_path()+"all/ctpn_ckpt/",
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

# print(json.dumps(CONF, ensure_ascii=False, indent=2))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# get input for ctpn
def get_images(input_dir):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'gif', 'pdf']
    # os.walk will recursively go through all nodes(directories) in this path
    for parent, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            for ext in exts:
                if filename.lower().endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def init_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


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

    re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    # cv.imencode('.jpg', re_im)[1].tofile('data/res/1.jpg')
    # return img, (1, 1)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


# def ctpn(im):
#     h0, w0, c0 = im.shape
#
#     filter_size = 800
#     stride = 600
#     img1, parts = img_prepro.split_images(im, filter_size=filter_size, stride=stride)
#     all_boxes = []
#
#     for part in parts:
#         start = (part[0], part[1])
#         print(start)
#         tmp_im = img1[part[1]:part[3], part[0]:part[2], :].copy()
#         h, w, c = tmp_im.shape
#         im_info = np.array([h, w, c]).reshape([1, 3])
#         tmp_boxes, scores = ctpn_processor.detect(tmp_im, im_info)
#
#         # for tmp_box in tmp_boxes:
#         #     cv.rectangle(tmp_im, pt1=(tmp_box[0], tmp_box[1]), pt2=(tmp_box[2], tmp_box[5]), color=(0, 255, 0),
#         #                   thickness=2)
#         # cv.imwrite(os.path.join(output_path, str(start) + '.jpg'), tmp_im[:, :, ::-1])
#
#         for tmp_box in tmp_boxes:
#             tmp2 = [0 for i in range(4)]
#             tmp2[0] = tmp_box[0] - 5 + start[0] if tmp_box[0] - 5 + start[0] > start[0] else start[0]
#             tmp2[1] = tmp_box[1] - 2 + start[1] if tmp_box[1] - 2 + start[1] > start[1] else start[1]
#             tmp2[2] = tmp_box[2] + 5 + start[0] if tmp_box[2] + 5 + start[0] < start[0] + filter_size else start[
#                                                                                                                0] + filter_size
#             tmp2[3] = tmp_box[5] + 2 + start[1] if tmp_box[3] + 2 + start[1] < start[1] + filter_size else start[
#                                                                                                                1] + filter_size
#             all_boxes.append(tmp2)
#
#     # resize还是得要，不然一个大字冒出来，人都没了，而且resize可以有效减少all_boxes的数量
#     img2, (rh, rw) = img_prepro.resize_image(im)
#
#     h, w, c = img2.shape
#     im_info = np.array([h, w, c]).reshape([1, 3])
#     boxes, scores = ctpn_processor.detect(img2, im_info)
#     h1, w1, c1 = im.shape
#
#     resize_boxes = []
#
#     for i, box in enumerate(boxes):
#         #  not sure whether there should be padding, padding may let the noise in,
#         #  but without padding, part of the valuable info may be lost
#         actual = [box[0] - 5, box[1] - 2, box[2] + 6, box[5] + 2]
#         actual[0] = int(actual[0] / rw) if actual[0] > -1 else 0
#         actual[1] = int(actual[1] / rh) if actual[1] > -1 else 0
#         actual[2] = int(actual[2] / rw) if actual[2] < w1 else w1 - 1
#         actual[3] = int(actual[3] / rh) if actual[3] < h1 else h1 - 1
#
#         resize_boxes.append(actual)
#
#     result_boxes = bbox_postpro.merge_boxes(resize_boxes, all_boxes)
#     img3 = img1.copy()
#     for box in result_boxes:
#         cv.rectangle(img3, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(0, 255, 0), thickness=2)
#     cv.imwrite(os.path.join(SAVE_DIRS['output_dir'], 'resize.jpg'), img3[:, :, ::-1])
#     return result_boxes
#
#
# def get_now():
#     return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
#
#
# class Application(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.entry_var = tk.StringVar(value='')
#         self.record_var = tk.StringVar(value='')
#         self.output_var = tk.StringVar(value='')
#         self.show_var = tk.StringVar()
#         self.show_var.set('欢迎使用文本审查！')
#         self.input_dir = None
#         self.record_path = None
#         self.output_dir = None
#         self.create_widgets()
#
#     def create_widgets(self):
#         self.title('OCR Investigation')
#         self.geometry('500x300')
#
#         lb0 = Label(self, text='文本审查', bg='#3385ff', fg='white', font=('华文新魏', 24), width=10, height=2)
#         lb0.pack(pady=5)
#         top_frame = tk.Frame(self, height=80)
#         record_frame = tk.Frame(self, height=80)
#         next_frame = tk.Frame(self, height=80)
#         top_frame.pack(side=tk.TOP, pady=5)
#         record_frame.pack(side=tk.TOP, pady=5)
#         next_frame.pack(side=tk.TOP, pady=5)
#
#         input_label = Label(top_frame, text='输入路径:', font=('幼圆', 14, font.BOLD))
#         input_entry = Entry(top_frame, textvariable=self.entry_var, font=('Arial', 14))
#         input_button = Button(top_frame, command=self.__openfile, text='选择')
#         input_label.grid(row=0, column=0, sticky=tk.W)
#         input_entry.grid(row=0, column=1)
#         input_button.grid(row=0, column=2)
#
#         record_label = Label(record_frame, text='任务记录:', font=('幼圆', 14, font.BOLD))
#         record_entry = Entry(record_frame, textvariable=self.record_var, font=('Arial', 14))
#         record_button = Button(record_frame, command=self.__openrecord, text='选择')
#         record_label.grid(row=0, column=0, sticky=tk.W)
#         record_entry.grid(row=0, column=1)
#         record_button.grid(row=0, column=2)
#
#         output_label = Label(next_frame, text='输出路径:', font=('幼圆', 14, font.BOLD))
#         output_entry = Entry(next_frame, textvariable=self.output_var, font=('Arial', 14))
#         output_button = Button(next_frame, command=self.__opendir, text='选择')
#         output_label.grid(row=0, column=0, sticky=tk.W)
#         output_entry.grid(row=0, column=1)
#         output_button.grid(row=0, column=2)
#
#         submit_button = Button(self, command=self.__submit, text='开始审查', font=('幼圆', 14, font.BOLD))
#         submit_button.pack()
#
#         lb0 = Label(self, textvariable=self.show_var, font=('华文新魏', 12), height=2)
#         lb0.pack(pady=5)
#
#     def change_text(self, status):
#         self.show_var.set(status)
#
#     def __openfile(self):
#         self.input_dir = filedialog.askdirectory(title="请选择需要被审查的文件夹")
#         self.entry_var.set(self.input_dir)
#         if not self.input_dir:
#             messagebox.showwarning('警告', message='未选择文件夹！')
#
#     def __openrecord(self):
#         self.record_path = filedialog.askopenfilename(title="请选择任务记录", filetypes=[("JSON", "*.json")])
#         self.record_var.set(self.record_path)
#         if not self.record_var:
#             messagebox.showwarning('警告', message='请选择任务记录！')
#
#     def __opendir(self):
#         # 打开文件夹的逻辑
#         self.output_dir = filedialog.askdirectory(title='请选择输出路径')  # 打开文件夹对话框
#         self.output_var.set(self.output_dir)  # 设置变量output_var，等同于设置部件Entry
#         if not self.output_dir:
#             messagebox.showwarning('警告', message='未选择文件夹！')  # 弹出消息提示框
#
#     def __submit(self):
#         self.input_dir = str(self.entry_var.get())
#         self.record_path = str(self.record_var.get())
#         self.output_dir = str(self.output_var.get())
#         if self.input_dir == '' and self.record_path == '':
#             messagebox.showwarning('警告', message='未选择输入！')
#         elif self.output_dir == '':
#             messagebox.showwarning('警告', message='未选择输出文件夹！')  # 弹出消息提示框
#         elif self.record_path != '' and self.entry_var.get() != '':
#             messagebox.showwarning('警告', message='待审查目录和任务记录只需要其中一个！')
#         elif self.input_dir != ''and (not os.listdir(self.input_dir)):
#             messagebox.showwarning('警告', message='未检测到输入目录中存在需要审查的文件！')
#         elif self.entry_var.get() != '' and os.listdir(self.output_dir):
#             messagebox.showwarning('警告', message='输出目录不为空！')
#         else:
#             if self.record_path != '':
#                 record = TaskRecord(record_path=self.record_path)
#                 if not record.get_record():
#                     messagebox.showwarning('警告', message='任务记录不存在或已失效!')
#                 else:
#                     self.__recognize(self.record_path, self.output_dir)
#             elif not os.path.isdir(self.input_dir):
#                 messagebox.showwarning('警告', message=self.input_dir+'待审查目录路径不存在！')
#             elif not os.path.isdir(self.output_dir):
#                 messagebox.showwarning('警告', message=self.output_dir+'输出目录路径不存在！')
#             else:
#                 self.__recognize(self.input_dir, self.output_dir)
#
#     def __recognize(self, input, output_dir):
#         start = time.time()
#         init_dir(SAVE_DIRS['cache_dir'])
#
#         file_list = []
#         sep = os.path.sep
#         if input.endswith('.json'):
#             task_record = TaskRecord(record_path=input)
#             input_dir = task_record.get_record()['input_path']
#             output_dir = task_record.get_record()['output_path']
#             files = task_record.get_record()['files']
#             for file in get_images(input_dir):
#                 if os.path.basename(file) not in files:
#                     file_list.append(file)
#         else:
#             input_dir = input + sep
#             input_dir.replace('/', sep)
#             output_dir += sep
#             output_dir.replace('/', sep)
#             task_record = TaskRecord(record_path=output_dir+'record.json')
#             task_record.create_record(input_dir, output_dir)
#             file_list = get_images(input_dir)
#         print(file_list)
#         for file in file_list:
#             print('===============')
#             print(file)
#
#             basename = os.path.basename(file)
#             print(basename)
#             ext = '.' + str(basename.split('.')[-1])
#             name_part = basename[:len(basename) - len(ext)]
#
#             init_dir(SAVE_DIRS['temp_dir'])
#             all_text_record = []
#             if file.lower().endswith('gif'):
#                 process_gif(file, SAVE_DIRS['temp_dir'])
#             elif file.lower().endswith('pdf'):
#                 test = PDFTransformer(file, SAVE_DIRS['temp_dir'])
#                 all_text_record = test.extract_text_and_image()
#             else:
#                 shutil.copy(file, SAVE_DIRS['temp_dir'])
#
#             image_list = get_images(SAVE_DIRS['temp_dir'])
#             print(image_list)
#             self.show_var.set('初始化输出目录中！')
#             meta_path, text_path, absolute_path, record = \
#                 fm.init_file(input_dir, basename, pages_num=len(image_list), output_dir=output_dir)
#             print(text_path)
#             print(absolute_path)
#
#             if len(all_text_record) != 0:
#                 for i, page_record in enumerate(all_text_record):
#                     tmp = {'pagination': str(i), 'content': []}
#                     for text_record in page_record:
#                         bbox = text_record[0]
#                         text_in = text_record[1]
#                         line_height = bbox[3] - bbox[1]
#                         tmp['content'].append({'box': str(bbox),
#                                                'text': text_in,
#                                                'line_height': str(line_height)})
#                     record['pages'].append(tmp)
#
#             for page, im_fn in enumerate(image_list):
#                 if len(record['pages']) < page+1:
#                     tmp = {'pagination': str(page),
#                            'content': []}
#                     record['pages'].append(tmp)
#
#                 self.show_var.set(os.path.basename(file)+'检测文本中')
#                 try:
#                     print(im_fn)
#                     im = cv.imdecode(np.fromfile(im_fn, dtype=np.uint8), 1)[:, :, ::-1]
#                     # im = cv.imread(im_fn)[:, :, ::-1]
#                 except IOError:
#                     print("Error reading image {}!".format(im_fn))
#                     continue
#                 h0, w0, c0 = im.shape
#                 record['pages'][page]['height'] = str(h0)
#                 record['pages'][page]['width'] = str(w0)
#                 actual_boxes = ctpn(im)
#
#                 print(im_fn)
#                 # basename = os.path.basename(file)
#                 # ext = '.' + str(basename.split('.')[-1])
#                 # name_part = '%s(%d)' % (basename[:len(basename)-len(ext)], page)
#                 # print(name_part)
#                 # output_path = os.path.join(output_dir, name_part+'.txt')
#
#                 # 排版什么的放在参考文本就好吧
#                 # box需要以下信息：left, right, height*std_width
#                 actual_boxes.sort(key=lambda l: l[1])
#
#                 # with open(output_path, 'w') as f:
#                 self.show_var.set(os.path.basename(file) + '识别文本中')
#                 for num, actual_box in enumerate(actual_boxes):
#                     if actual_box:
#                         rs_text, lines_height = proc.get_text_result(im_fn, p_thresh=CONF['p_thresh'],
#                                                   auxiliary_img=frozen_dir.app_path()+'static'+sep+str(actual_box)+'.jpg',
#                                                   box=actual_box,
#                                                   remove_lines=1)
#                         if len(lines_height) == 0:
#                             print(actual_box)
#                             continue
#                         tmp = record['pages'][page]['content']
#                         tmp.append({'box': str(actual_box),
#                                     'text': rs_text,
#                                     'line_height': str(max(lines_height)),
#                                     'num': num})
#                         print(actual_box)
#                         # f.writelines(str(actual_box)+rs_text+'\n')
#             record['last_modified_time'] = get_now()
#             with open(meta_path, 'w', encoding='utf-8') as f:
#                 file_json = json.dumps(record, ensure_ascii=False)
#                 f.write(file_json)
#             og.transfer_for_read(record, text_path, absolute_path)
#             task_record.save_record(basename)
#         end = time.time()
#         print('总共的时间为:', round(end - start, 2), 'secs')
#         self.show_var.set('运行完毕！')

class MainPage(object):
    def __init__(self, master=None):
        self.root = master  # 定义内部变量root
        self.root.geometry('%dx%d' % (500, 300))  # 设置窗口大小
        self.create_page()

    def create_page(self):
        self.inputPage = InputFrame(self.root)  # 创建不同Frame
        self.queryPage = TransferFrame(self.root)
        self.countPage = CountFrame(self.root)
        self.aboutPage = AboutFrame(self.root)
        self.inputPage.pack()  # 默认显示数据录入界面
        menubar = Menu(self.root)
        menubar.add_command(label='文本审查', command=self.inputData)
        menubar.add_command(label='文本纠正', command=self.queryData)
        menubar.add_command(label='统计', command=self.countData)
        menubar.add_command(label='关于', command=self.aboutDisp)
        self.root['menu'] = menubar  # 设置菜单栏

    def inputData(self):
        self.inputPage.pack()
        self.queryPage.pack_forget()
        self.countPage.pack_forget()
        self.aboutPage.pack_forget()

    def queryData(self):
        self.inputPage.pack_forget()
        self.queryPage.pack()
        self.countPage.pack_forget()
        self.aboutPage.pack_forget()

    def countData(self):
        self.inputPage.pack_forget()
        self.queryPage.pack_forget()
        self.countPage.pack()
        self.aboutPage.pack_forget()

    def aboutDisp(self):
        self.inputPage.pack_forget()
        self.queryPage.pack_forget()
        self.countPage.pack_forget()
        self.aboutPage.pack()


if __name__ == '__main__':
    # 可以考虑添加在线修改功能，显示对应坐标的图中位置和文本，并实时修正，以及删除相应文本框和添加文本框并识别
    # 实例化Application
    # app = Application()
    # # 主消息循环:
    # app.mainloop()
    root = Tk()
    root.title('OCR Investigation')
    MainPage(root)
    root.mainloop()
















