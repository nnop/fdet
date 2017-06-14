#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from caffe.proto import caffe_pb2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', help='folder to store the labelmap file')
    parser.add_argument('--class-names', nargs='+', help='class names')
    parser.add_argument('--class-file', help='file store class names')
    args = parser.parse_args()
    if args.class_names:
        class_names = args.class_names
    elif args.class_file:
        with open(args.class_file) as f:
            class_names = f.read().split()
    else:
        raise ValueError('wrong argument')
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
