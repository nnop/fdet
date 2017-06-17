#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import cv2
import os.path as osp
import sys

import _init_paths
import pathutils
from vis import draw_bbox

if __name__ == "__main__":
    samp_name = sys.argv[1]
    image_path = 'data/pose/images/{}.jpg'.format(samp_name)
    json_path = 'data/pose/json/{}.json'.format(samp_name)
    pathutils.check_f_exists(image_path)
    pathutils.check_f_exists(json_path)

    print json_path
    im = cv2.imread(image_path)[:, :, [2, 1, 0]]
    plt.imshow(im)
    with open(json_path) as f:
        ann = json.load(f)
    ax = plt.gca()
    for body in ann['annotation']:
        bbox = body['bbox']
        bbox[2] = bbox[0] + bbox[2] -1
        bbox[3] = bbox[1] + bbox[3] -1
        draw_bbox(ax, bbox, text=str(body['aziLabel']), color='r', linewidth=1)
    plt.savefig('out.png')
