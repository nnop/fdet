#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
show json annotation
"""
import matplotlib.pyplot as plt
import json

import _init_paths
from vis import draw_bbox

json_path = 'data/stuprt/json/1164.json'
image_path = 'data/stuprt/ori_data/Frame/1164.jpg'

im = plt.imread(image_path)
with open(json_path) as f:
    image_info = json.load(f)
assert image_info['image']['height'] == im.shape[0]
assert image_info['image']['width'] == im.shape[1]
objects = image_info['annotation']
plt.imshow(im)
ax = plt.gca()
colors = plt.cm.hsv([0, 0.5])
for obj in objects:
    label =  obj['aziLabel']
    bbox = obj['bbox']
    bbox[2] = bbox[0] + bbox[2] - 1
    bbox[3] = bbox[1] + bbox[3] - 1
    draw_bbox(ax, bbox, color=colors[label])

plt.show()
