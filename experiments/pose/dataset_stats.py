#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import argparse
import os.path as osp
from glob import glob

class PoseDataset(object):
    def __init__(self):
        self.head_labels = {}
        self.body_labels = {}
        self.n_body = 0
        self.n_head = 0

    def add_head(self, label):
        self._add_label(self.head_labels, label)
        self.n_head += 1

    def add_body(self, label):
        self._add_label(self.body_labels, label)
        self.n_body += 1
    
    def _add_label(self, d, label):
        if label in d:
            d[label] += 1
        else:
            d[label] = 0

    def _show_labels(self, d):
        for i, k in enumerate(d):
            print '[{}] {}: {}'.format(i, k, d[k])

    def _dump_labels(self, label_d, info_s, dump_path):
        labels = label_d.keys()
        labels = [e+'\n' for e in sorted(labels)]
        with open(dump_path, 'w') as f:
            f.writelines(labels)
        print 'dump {} labels to: {}'.format(info_s, dump_path)

    def show_head(self):
        print '---- head labels ----'
        self._show_labels(self.head_labels)

    def show_body(self):
        print '---- body labels ----'
        self._show_labels(self.body_labels)

    def dump_body(self, dump_path):
        self._dump_labels(self.body_labels, 'body', dump_path)

    def dump_head(self, dump_path):
        self._dump_labels(self.head_labels, 'head', dump_path)

    def reset(self):
        self.head_labels.clear()
        self.body_labels.clear()
        self.n_body = 0
        self.n_head = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-head', help='file to save all head labels')
    parser.add_argument('--dump-body', help='file to save all body labels')
    parser.add_argument('jsondir')
    args = parser.parse_args()

    stat = PoseDataset()
    n_body_miss = 0
    n_head_miss = 0
    n_student = 0
    n_teacher = 0
    json_files = glob(osp.join(args.jsondir, '*.json'))
    for json_p in json_files:
        with open(json_p) as f:
            image_info = json.load(f)
        for psn in image_info['persons']:
            if psn['student']:
                n_student += 1
            else:
                n_teacher += 1
            if 'body' in psn:
                stat.add_body(psn['body']['label'])
            else:
                n_body_miss += 1
            if 'head' in psn:
                stat.add_head(psn['head']['label'])
            else:
                n_head_miss += 1

    print '==== results ===='
    print 'label files: {}'.format(len(json_files))
    print 'body miss: {}/{}'.format(n_body_miss, stat.n_body)
    print 'head miss: {}/{}'.format(n_head_miss, stat.n_head)
    print 'student: {}'.format(n_student)
    print 'teacher: {}'.format(n_teacher)
    stat.show_head()
    stat.show_body()

    if args.dump_head:
        stat.dump_head(args.dump_head)
    if args.dump_body:
        stat.dump_body(args.dump_body)
