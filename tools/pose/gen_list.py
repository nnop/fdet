#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import fnmatch

list_path = "data/stuprt/list/train.txt"

image_root = 'data/stuprt/ori_data/'
json_root = 'data/stuprt/json/'
lines = []
for cur, _, fs in os.walk(image_root):
    images = fnmatch.filter(fs, '*.jpg')
    for f in images:
        image_path = osp.join(cur, f)
        json_path = osp.join(json_root, osp.splitext(f)[0]+'.json')
        assert osp.isfile(json_path), json_path
        lines.append('{} {}\n'.format(image_path, json_path))

with open(list_path, 'w') as f:
    f.writelines(lines)
