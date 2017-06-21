#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import cv2
import os.path as osp
import sys
import argparse

import _init_paths
import pathutils
from vis import draw_bbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--json-path', required=True)
    parser.add_argument('--class-file', required=True)

    image_path = args.image_path
    json_path = args.json_path
    class_file = args.class_file

    pathutils.check_f_exists(image_path)
    pathutils.check_f_exists(json_path)

    with open(class_file) as f:
        classes = [l.strip() for l in f] + ['Other']

    im = cv2.imread(image_path)[:, :, [2, 1, 0]]
    plt.imshow(im)
    with open(json_path) as f:
        ann = json.load(f)
    ax = plt.gca()
    for body in ann['annotation']:
        bbox = body['bbox']
        bbox[2] = bbox[0] + bbox[2] -1
        bbox[3] = bbox[1] + bbox[3] -1
        i_label = int(body['aziLabel'])
        draw_bbox(ax, bbox, text=classes[i_label], color='r', linewidth=1)
    plt.savefig('out.png')
    print 'result saved to out.png'
