#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert xml annotation to pose format and dump to json file
"""
import argparse
import os.path as osp
import json

import _init_paths
from pathutils import make_if_not_exists
import poseutils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-list', required=True, \
            help='file store class name list')
    parser.add_argument('--data-root', required=True, \
            help='data root')
    parser.add_argument('image_path')
    args = parser.parse_args()
    with open(args.class_list) as f:
        class_names = f.read().split()
    image_path = osp.normpath(args.image_path)
    data_root = osp.normpath(args.data_root)
    xml_path = poseutils.image_to_xml_path(image_path)
    # read xml
    image_info = poseutils.load_image_info(xml_path, class_names)
    # dump json
    json_path = poseutils.image_to_json_path(image_path, data_root)
    json_dir = osp.split(json_path)[0]
    make_if_not_exists(json_dir)
    with open(json_path, 'w') as f:
        json.dump(image_info, f)
    print '{} -> {}'.format(image_path, json_path)
