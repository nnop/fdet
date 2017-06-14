#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from xml.etree import ElementTree
import json

def load_image_info(xml_path, class_names):
    xml_obj = ElementTree.parse(xml_path)
    width = int(xml_obj.find('size').find('width').text)
    height = int(xml_obj.find('size').find('height').text)
    annotations = []
    objects = xml_obj.findall('object')
    for obj in objects:
        anno = {}
        name = obj.find('name').text
        cls_label = class_names.index(name)
        anno['aziLabel'] = cls_label
        anno['aziLabelFlip'] = cls_label
        bbobj = obj.find('bndbox')
        xmin = int(float(bbobj.find('xmin').text))
        xmax = int(float(bbobj.find('xmax').text))
        ymin = int(float(bbobj.find('ymin').text))
        ymax = int(float(bbobj.find('ymax').text))
        anno['bbox'] = [xmin, ymin, xmax-xmin+1, ymax-ymin+1]
        anno['category_id'] = 'person'
        anno['difficult'] = False
        anno['iscrowd'] = 0
        annotations.append(anno)
    # convert json
    image_info = {
            'image': { 'height': height, 'width': width },
            'annotation': annotations
        }
    return image_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-list', required=True, help='file store class name list')
    parser.add_argument('xml_path', help='xml path')
    parser.add_argument('save_json_path', help='json path')
    args = parser.parse_args()
    with open(args.class_list) as f:
        class_names = f.read().split()
    # read xml
    image_info = load_image_info(args.xml_path, class_names)
    # dump json
    with open(args.save_json_path, 'w') as f:
        json.dump(image_info, f)
    print '{} -> {}'.format(args.xml_path, args.save_json_path)
