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


if __name__ == '__main__':
    a = app_path()+'\\/\\/'
    print(a)
    print(unify_sep(a))
