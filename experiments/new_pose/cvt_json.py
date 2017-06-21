#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split train/val/test datasets
"""
import sys
import os
import json
import os.path as osp
from glob import glob
import random
import fnmatch

import _init_paths
from pathutils import main_name, make_if_not_exists
from vocparser import VOCParser

def parse_xml(xml_p, classes):
    info = {}
    # load
    ori_info = VOCParser(xml_p)
    # parse
    hei, wid, dep = ori_info.size()
    info['image'] = {
            'height': hei,
            'width' : wid,
        }
    info['annotation'] = []
    info['filename'] = ori_info.image_name()
    objects = ori_info.objects()
    for obj in objects:
        bbox = obj['bbox'] # x1, x2, y1, y2
        body = {}
        body['category_id'] = 'person'
        body['aziLabel'] = classes.index(obj['name'])
        body['aziLabelFlip'] = body['aziLabel']
        body['difficult'] = False
        body['iscrowd'] = False
        # note: [x1, y1, wid, hei]
        body['bbox'] = [bbox[0], bbox[1], bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
        info['annotation'].append(body)
    return info

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
        try:
            body['aziLabel'] = classes.index(psn['body']['label'])
        except ValueError as e:
            body['aziLabel'] = len(classes)
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

def get_all_xml_files(data_root):
    xml_files = []
    for cur_dir, dirs, fns in os.walk(data_root):
        for fn in fnmatch.filter(fns, '*.xml'):
            xml_p = osp.join(cur_dir, fn)
            xml_files.append(xml_p)
            assert osp.isfile(xml_p), xml_p
    return xml_files

def process_xml_files(xml_files, classes, save_root_dir, dump_list_path):
    """
    each xml path: save_root/data_path/task_name/xml/xxx.xml
    """
    json_root_dir = osp.join(save_root_dir, 'json')
    make_if_not_exists(json_root_dir)
    lines = []
    for xml_p in xml_files:
        # path blocks
        p_blks = xml_p.split('/')
        samp_name = osp.splitext(p_blks[-1])[0]
        # json paths
        i_xml = p_blks.index('xml')
        task_name = p_blks[i_xml - 1]
        json_dir = osp.join(json_root_dir, task_name)
        make_if_not_exists(json_dir)
        json_p = osp.join(json_dir, samp_name+'.json')
        json_rel_p = osp.relpath(json_p, save_root_dir)
        # image path
        tmp = p_blks[:i_xml]+['Frame']
        image_d = osp.join(*tmp)
        image_p = osp.join(image_d, samp_name+'.jpg')
        image_rel_p = osp.relpath(image_p, save_root_dir)
        # dump json
        info = parse_xml(xml_p, classes)
        with open(json_p, 'w') as f:
            json.dump(info, f)
        # add to lines
        lines.append('{} {}\n'.format(image_rel_p, json_rel_p))
        with open(dump_list_path, 'w') as f:
            f.writelines(lines)
    print 'dump to ' + dump_list_path

if __name__ == "__main__":
    save_root_dir = 'data/new_pose'
    dump_list_path = 'data/new_pose/all_xml_list.txt'
    class_file = 'data/new_pose/body_labels.txt'

    with open(class_file) as f:
        classes = [l.strip() for l in f]
    print 'classes: {}'.format(classes)

    xml_files = get_all_xml_files(save_root_dir)
    process_xml_files(xml_files, classes, save_root_dir, dump_list_path)
