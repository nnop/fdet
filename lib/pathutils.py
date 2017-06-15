import os
import os.path as osp

def make_if_not_exists(folder):
    if not osp.isdir(folder):
        os.makedirs(folder)

