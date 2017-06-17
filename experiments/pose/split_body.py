#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split train/val/test datasets
"""
import sys
import json
import os.path as osp
from glob import glob
import random

import _init_paths
from pathutils import main_name, make_if_not_exists

def parse_ann(ann_p, classes):
    """
    parser origin ann
    """
    info = {}
    with open(ann_p) as f:
        ori_info = json.load(f)
    info['image'] = {
            'height': ori_info['image']['height'],
            'width' : ori_info['image']['width'],
        }
    info['annotation'] = []
    info['filename'] = ori_info['image']['filename']
    for psn in ori_info['persons']:
        bbox = psn['body']['bbox']
        body = {}
        body['category_id'] = 'person'
        body['aziLabel'] = classes.index(psn['body']['label'])
        body['aziLabelFlip'] = body['aziLabel']
        body['difficult'] = False
        body['iscrowd'] = False
        body['bbox'] = [bbox[0], bbox[1], bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
        info['annotation'].append(body)
    return info

def dump_list(ann_files, classes, image_dir, root_dir, dump_ann_dir, dump_list_path):
    lines = []
    # split all list
    for ann_p in ann_files:
        samp_name = main_name(ann_p)
        image_p = osp.join(image_dir, samp_name+'.jpg')
        dst_json_p = osp.join(dump_ann_dir, samp_name+'.json')
        info = parse_ann(ann_p, classes)
        with open(dst_json_p, 'w') as f:
            json.dump(info, f)
        l = '{} {}\n'.format(osp.relpath(image_p, root_dir),
                osp.relpath(dst_json_p, root_dir))
        lines.append(l)
    with open(dump_list_path, 'w') as f:
        f.writelines(lines)
    print 'dump list to ' + dump_list_path

if __name__ == "__main__":
    ori_ann_dir = 'data/pose/images'
    image_dir = ori_ann_dir
    dst_ann_dir = 'data/pose/json'
    dump_list_path = 'data/pose/trainlist.txt'
    class_file = 'data/pose/bodylabels.txt'

    make_if_not_exists(dst_ann_dir)
    with open(class_file) as f:
        classes = [l.strip() for l in f.readlines()]

    ann_files = glob(osp.join(ori_ann_dir, '*.json'))
    random.shuffle(ann_files)

    n_total = len(ann_files)
    n_tr = int(n_total * 0.7)
    train_ann_files = ann_files[:n_tr]
    val_ann_files = ann_files[n_tr:]

    dump_list(train_ann_files, classes, image_dir, 'data/pose', dst_ann_dir,
            dump_list_path)
