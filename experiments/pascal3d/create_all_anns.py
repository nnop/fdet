#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import numpy as np
import os.path as osp
from glob import glob
import random
import scipy.io as sio

from pprint import pprint

this_dir = osp.split(osp.realpath(__file__))[0]
data_dir = osp.normpath(osp.join(this_dir, '../../data/pascal3d'))
all_anns_path = osp.join(data_dir, 'all_anns.json')

def remove_bottle(data):
    keysToRemove = []
    count = 0
    for key, ann in data.iteritems():
        newList = []
        for obj in ann['annotation']: 
            if obj['category_id'] != 'bottle':
                newList.append(obj)
            else:
                count += 1
        if len(newList) == 0: keysToRemove.append(key)
        ann['annotation'] = newList

    for rem in keysToRemove:
        del data[rem]
    return data

def parse_mat(matfile):
    img_annotation = {}
    attrs = ['filename', 'database']
    for attr in attrs:
        img_annotation[attr] = getattr(matfile['record'], attr)

    # get size attributes about the image
    size_obj = getattr(matfile['record'], 'size')
    sz = []
    for field in size_obj._fieldnames:
        sz.append(getattr(size_obj, field))
    img_annotation['image'] = {}
    img_annotation['image']['width'] = int(sz[0])
    img_annotation['image']['height'] = int(sz[1])

    list_obj = []
    attrs = ['bbox', 'class', 'viewpoint', 'difficult']
    
    # get objects in the image
    tempAttr = getattr(matfile['record'], 'objects')
    objects_obj = np.asarray([tempAttr]).flatten()

    for ind in range(objects_obj.shape[0]):
        the_obj = objects_obj[ind]
        objects = {}
        #print 'broke to a'
        for field in attrs:
            #print 'broke to b'
            if field == 'viewpoint':    
                view_obj = getattr(the_obj, 'viewpoint')
                if isinstance(view_obj, np.asarray([]).__class__):
                    continue
                views = {}
                #print view_obj._fieldnames
                for view_field in view_obj._fieldnames:
                    views[view_field] = getattr(view_obj, view_field)
                objects[field] = views
            elif field == 'bbox':
                objects[field] = getattr(the_obj, field).tolist()
            elif field == 'difficult':
                diff_int = getattr(the_obj, field)
                if diff_int == 0:
                    objects[field] = False
                else:
                    objects[field] = True
            else:
                cat_id = getattr(the_obj, field)
                objects['category_id'] = cat_id
        if 'viewpoint' in objects:
            objects['iscrowd'] = 0
            list_obj.append(objects)

    if list_obj == []:
        print 'whoops'
    img_annotation['annotation'] = list_obj
    return img_annotation

def split_data(data):
    tr_list_p = osp.join(data_dir, 'ImageSets/Main/train.txt')
    with open(tr_list_p) as f:
        tr_list = [l.strip() for l in f.readlines()]
    count = 0
    for e in tr_list:
        if e in data:
            count += 1
            data[e]['split'] = 'train'
    print '{} train items'.format(count)

    val_list_p = osp.join(data_dir, 'ImageSets/Main/val.txt')
    with open(val_list_p) as f:
        val_list = [l.strip() for l in f.readlines()]
    count = 0
    for e in val_list:
        if e in data:
            count += 1
            data[e]['split'] = 'test'
    print '{} test items'.format(count)

    for key, obj in data.iteritems():
        if obj['database'] == 'ImageNet':
            obj['split'] = 'train'

    keys = data.keys()
    random.shuffle(keys)
    count = 0
    for k in keys:
        if count == 500: break
        if data[k]['split'] == 'train':
            data[k]['split'] = 'val'
            count += 1
    return data

def check_counts(data):
    pascal_tr_data = {}
    for k in data:
        if data[k]['database'] != 'ImageNet' and data[k]['split'] == 'test':
            pascal_tr_data[k] = data[k]
    counts = {}
    for im in pascal_tr_data.itervalues():
        for obj in im['annotation']:
            cat = obj['category_id']
            if cat in counts:
                counts[cat] += 1
            else:
                counts[cat] = 1
    for k in sorted(counts):
        print '{} : {}'.format(k, counts[k])

def convert_bbox(data):
    for ann in data.itervalues():
        for obj in ann['annotation']:

            if obj['bbox'][2] < obj['bbox'][0]:
                temp = obj['bbox'][0]
                obj['bbox'][0] = obj['bbox'][2]
                obj['bbox'][2] = temp 
                print 'a' 
                print ann['filename']

            if obj['bbox'][3] < obj['bbox'][1]:
                temp = obj['bbox'][1]
                obj['bbox'][1] = obj['bbox'][3]
                obj['bbox'][3] = temp
                print b
                print ann['filename']

            if obj['bbox'][0] < 0:
                obj['bbox'][0] = 0

            if obj['bbox'][1] < 0:
                obj['bbox'][1] = 0

            if obj['bbox'][2] > ann['image']['width']:
                obj['bbox'][2] = ann['image']['width']
            if obj['bbox'][3] > ann['image']['height']:
                obj['bbox'][3] = ann['image']['height']

            obj['bbox'][2] = obj['bbox'][2] - obj['bbox'][0]
            obj['bbox'][3] = obj['bbox'][3] - obj['bbox'][1]

def make_all_anns(all_anns_path):
    all_anns = {}

    anns_dir = osp.join(data_dir, 'Annotations')
    assert osp.isdir(anns_dir), '{} not exists.'.format(anns_dir)
    all_ann_dirs = glob(osp.join(anns_dir, '*/'))
    for ann_dir in all_ann_dirs:
        files = glob(osp.join(ann_dir, '*.mat'))
        for idx, mat_p in enumerate(files):
            samp_name = osp.split(mat_p)[1][:-4]
            h_mat = sio.loadmat(mat_p, squeeze_me=True, struct_as_record=False)
            temp = parse_mat(h_mat)
            if samp_name in all_anns:
                all_anns[samp_name]['annotation'].extend(temp['annotation'])
            else:
                all_anns[samp_name] = temp
            if idx % 100 == 0:
                print 'processed {} files'.format(idx)
    all_anns = remove_bottle(all_anns)
    all_anns = split_data(all_anns)
    check_counts(all_anns)
    convert_bbox(all_anns)

    # hack hard coded 
    try:
        del all_anns['n03790512_11192']
        del all_anns['n02690373_190'] 
    except KeyError as e:
        pass
    with open(all_anns_path, 'w') as f:
        json.dump(all_anns, f)
    print 'dump anns to {}'.format(all_anns_path)

if __name__ == "__main__":
    make_all_anns(all_anns_path)
