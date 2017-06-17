#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import numpy as np
import os.path as osp
from glob import glob
import random

import _init_paths
from pathutils import make_if_not_exists, check_f_exists

this_dir = osp.split(__file__)[0]
data_dir = osp.normpath(osp.join(this_dir, '../../data/pascal3d'))
all_anns_path = osp.join(data_dir, 'all_anns.json')

def bin_angles(ann, bins, rotate=False):
    offset = 0
    if rotate:
        offset = 360 / (bins * 2)
    for obj in ann['annotation']:
        if 'viewpoint' in obj:
            azi = obj['viewpoint']['azimuth_coarse']
            
            if 'azimuth' in obj['viewpoint'] and obj['viewpoint']['distance'] != 0.0:
                azi = obj['viewpoint']['azimuth']

            # bin = int(azi / (360/bins))
            bin = int(((azi + offset) % 360) / (360/bins))
            #print "Azi: %f  bin: %d" %(azi, bin) 
            obj['aziLabel'] = bin
            flipAzi = 360 - azi
            # obj['aziLabelFlip'] = int(flipAzi / (360/bins))
            obj['aziLabelFlip'] = int(((flipAzi + offset) % 360) / (360/bins))
        else:
            print 'woah where yo angle'

def filter_difficult(data):
    keysToRemove = []
    count = 0
    for key, ann in data.iteritems():
        newList = []
        for obj in ann['annotation']:
            if not obj['difficult']:
                newList.append(obj)
            else:
                count += 1
        if len(newList) == 0: keysToRemove.append(key)
        ann['annotation'] = newList
    for rem in keysToRemove:
        del data[rem]

if __name__ == "__main__":
    with open(all_anns_path) as f:
        data = json.load(f)
    filter_difficult(data)

    json_dir = osp.join(data_dir, 'json')
    train_dir = osp.join(json_dir, 'train')
    val_dir = osp.join(json_dir, 'val')
    test_dir = osp.join(json_dir, 'test')
    split_dirs = { 'train': train_dir, 'val': val_dir, 'test': test_dir }

    make_if_not_exists(train_dir)
    make_if_not_exists(val_dir)
    make_if_not_exists(test_dir)

    split_lines = { 'train': [], 'val': [], 'test': [] }
    for samp_name, ann in data.iteritems():
        if ann['database'] == 'ImageNet':
            continue
        bin_angles(ann, bins=8, rotate=True)
        ann_p = osp.join(split_dirs[ann['split']], samp_name+'.json')
        img_p = osp.join(data_dir, 'JPEGImages', ann['filename'])
        with open(ann_p, 'w') as f:
            json.dump(ann, f)
        check_f_exists(img_p)
        out_line = '{} {}\n'.format(
                osp.relpath(img_p, data_dir),
                osp.relpath(ann_p, data_dir))
        print out_line.strip()
        split_lines[ann['split']].append(out_line)

    for split in split_lines:
        split_p = osp.join(data_dir, split+'_list.txt')
        random.shuffle(split_lines[split])
        with open(split_p, 'w') as f:
            f.writelines(split_lines[split])
            print 'dump {}'.format(split_p)
