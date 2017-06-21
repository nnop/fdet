#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
from os import path
import random
import cv2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
import dataset
from vocparser import VOCParser
from vis import draw_bbox

def get_image_xml_paths(args):
    if args.image_path:
        image_path = args.image_path
        xml_path = dataset.image_to_xml_path(image_path)
    elif args.xml_path:
        xml__path = args.xml_path
        image_path = dataset.xml_to_image_path(xml_path)
    elif args.data_dir and args.sample_id:
        image_path, xml_path = dataset.sample_id_to_path(args.data_dir,
                args.sample_id)
    else:
        raise ValueError('wrong arguments.')
    assert path.isfile(image_path), image_path
    assert path.isfile(xml_path), xml_path
    return image_path, xml_path

if __name__ == "__main__":
    """
    use image/xml path or sample-id as input
    """
    parser = argparse.ArgumentParser()
    samp_g = parser.add_mutually_exclusive_group()
    samp_g.add_argument('--image-path')
    samp_g.add_argument('--xml-path')
    parser.add_argument('--data-dir')
    parser.add_argument('--sample-id')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    image_path, xml_path = get_image_xml_paths(args)
    tmp = path.split(image_path)[1]
    sample_id = path.splitext(tmp)[0]

    # check xml info
    voc_xml = VOCParser(xml_path)
    hei_xml, wid_xml, ch_xml = voc_xml.size()
    objects = voc_xml.objects()

    # verify image size
    im = cv2.imread(image_path)[:, :, [2, 1, 0]]
    hei, wid, ch = im.shape
    assert hei_xml == hei and wid_xml == wid and ch_xml == ch, 'size wrong in xml.'

    # show each object
    colors = plt.cm.hsv
    plt.imshow(im)
    ax = plt.gca()
    cls_colors = {}
    for obj in objects:
        name = obj['name']
        bbox = obj['bbox']
        if name not in cls_colors:
            # show name on first detection
            cls_colors[name] = colors(random.random())
            c = cls_colors[name]
            draw_bbox(ax, bbox, text=name, color=c)
        else:
            c = cls_colors[name]
            draw_bbox(ax, bbox, color=c)

    if args.save:
        out_fn = sample_id+'.png'
        plt.savefig(out_fn)
        print 'save {} bbox to {}'.format(len(objects), out_fn)
    else:
        plt.show()
    
