#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import threading
import tensorflow as tf
import numpy as np
from text_detector.nets import model_train as model
from text_detector.utils.rpn_msr.proposal_layer import proposal_layer
from text_detector.utils.text_connector.detectors import TextDetector


class CtpnProcessor:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if CtpnProcessor._instance is None:
            with CtpnProcessor._instance_lock:
                if CtpnProcessor._instance is None:
                    CtpnProcessor._instance = object.__new__(cls)
        return CtpnProcessor._instance

    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        self.input_image = None
        self.input_im_info = None
        self.saver = None
        self.bbox_pred = None
        self.cls_pred = None
        self.cls_prob = None
        self.global_step = None

    def load(self, ckpt_dir=None):
        with self.sess.as_default():
            with self.graph.as_default():
                self.input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
                self.input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

                self.bbox_pred, self.cls_pred, self.cls_prob = model.model(self.input_image)

                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
                variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
                self.saver = tf.train.Saver(variable_averages.variables_to_restore())
                self.saver = tf.train.import_meta_graph(self.get_meta_path(ckpt_dir), clear_devices=True)
                self.restore(ckpt_dir)
        return self

    def restore(self, ckpt_dir=None):
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        print('Restore from {}'.format(ckpt))
        self.saver.restore(self.sess, ckpt)

    def detect(self, img, im_info):
        with self.sess.as_default():
            with self.graph.as_default():
                bbox_pred_val, cls_prob_val = self.sess.run([self.bbox_pred, self.cls_prob],
                                                       feed_dict={self.input_image: [img],
                                                                  self.input_im_info: im_info})
                # net_time = time.time()
                # print("Net cost time: {:.2f}s".format(net_time - resize_time))

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                # proposal_time = time.time()
                # print("Proposal cost time: {:.2f}s".format(proposal_time - net_time))

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)
                return boxes, scores

    @staticmethod
    def get_meta_path(ckpt_dir):
        for parent, dirnames, filenames in os.walk(ckpt_dir):
            for filename in filenames:
                if filename.endswith('meta'):
                    print(os.path.join(parent, filename))
                    return os.path.join(parent, filename)
