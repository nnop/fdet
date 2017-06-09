#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import random
import os
from os import path
import fnmatch
import shutil

def make_if_not_exists(folder):
    if not path.isdir(folder):
        os.makedirs(folder)

def get_sub_samples(data_dir, num):
    image_dir = path.join(data_dir, 'Frame')
    samples = fnmatch.filter(os.listdir(image_dir), '*.jpg')
    samples = [path.splitext(fn)[0] for fn in samples]
    sub_samples = random.sample(samples, num)
    return sub_samples

def copy_samples(src_root, dst_root, samples, save_dir):
    src_image_dir = path.join(src_root, 'Frame')
    src_xml_dir = path.join(src_root, 'xml')
    assert path.isdir(src_image_dir)
    assert path.isdir(src_xml_dir)
    dst_image_dir = path.join(dst_root, 'Frame')
    dst_xml_dir = path.join(dst_root, 'xml')
    make_if_not_exists(dst_image_dir)
    make_if_not_exists(dst_xml_dir)
    for name in samples:
        src_image_p = path.join(src_image_dir, name+'.jpg')
        dst_image_p = path.join(dst_image_dir, name+'.jpg')
        src_xml_p = path.join(src_xml_dir, name+'.xml')
        dst_xml_p = path.join(dst_xml_dir, name+'.xml')
        shutil.copyfile(src_image_p, dst_image_p)
        shutil.copyfile(src_xml_p, dst_xml_p)
        print name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--save-dir')
    parser.add_argument('--num', default=10, type=int)
    args = parser.parse_args()

    # get all samples
    sub_samples = get_sub_samples(args.data_dir, args.num)

    # copy samples
    copy_samples(args.data_dir, args.save_dir, sub_samples, args.num)
