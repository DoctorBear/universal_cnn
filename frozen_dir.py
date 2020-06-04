#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
import sys
import os


def app_path():
    """Returns the base application path."""
    if hasattr(sys, 'frozen'):
        # Handles PyInstaller
        return os.path.dirname(sys.executable)+os.path.sep
    return os.path.dirname(__file__)+os.path.sep


def unify_sep(path):
    if isinstance(path, str):
        wrong = '\\' if os.path.sep == '/' else '/'
        return path.replace(wrong, os.path.sep)
    else:
        return path


# 修正pdf密码中作为索引的路径中的路径分隔符
def unify_dict_sep(conf: dict):
    result = {}
    for k, v in conf.items():
        result[unify_sep(k)] = v
    return result


if __name__ == '__main__':
    print(app_path())
    a = app_path()+'\\/\\/'
    print(a)
    print(unify_sep(a))
