#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path as osp
from glob import glob
import scipy.io as sio

from pprint import pprint

this_dir = osp.split(osp.realpath(__file__))[0]
data_dir = osp.normpath(osp.join(this_dir, '../../data/pascal3d'))
all_anns_paths = osp.join(data_dir, 'all_anns.json')

def make_all_anns(all_anns_path):
    anns_dir = osp.join(data_dir, 'Annotations')
    assert osp.isdir(anns_dir), '{} not exists.'.format(anns_dir)
    all_ann_dirs = glob(osp.join(anns_dir, '*/'))
    for ann_dir in all_ann_dirs:
        files = glob(osp.join(ann_dir, '*.mat'))
        for idx, mat_p in enumerate(files):
            h_mat = sio.loadmat(mat_p, squeeze_me=True, struct_as_record=False)
            temp = parseMat(h_mat)
            if idx % 100 == 0:
                print 'processed {} files'.format(idx)

if __name__ == "__main__":
    make_all_anns(all_anns_paths)
