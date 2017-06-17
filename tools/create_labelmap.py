#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from caffe.proto import caffe_pb2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', help='class names')
    parser.add_argument('--class-file', help='file store class names')
    parser.add_argument('--save-path', help='folder to store the labelmap file')
    args = parser.parse_args()
    if args.classes:
        class_names = args.classes
    elif args.class_file:
        with open(args.class_file) as f:
            class_names = f.read().split()
    else:
        raise ValueError('class file {} not exists.'.format(args.class_file))
    labelmap = caffe_pb2.LabelMap()
    idx = 0
    item = labelmap.item.add()
    item.name = 'null'
    item.label = idx
    item.display_name = 'null'
    for cls_name in class_names:
        idx += 1
        item = labelmap.item.add()
        item.name = cls_name
        item.label = idx
        item.display_name = cls_name
    if args.save_path:
        with open(args.save_path, 'w') as f:
            f.write(str(labelmap))
        print 'save to', args.save_path
    else:
        print labelmap
