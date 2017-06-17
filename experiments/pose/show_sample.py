#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import json

import _init_paths
from vis import draw_bbox

if __name__ == "__main__":
    image_path = '~/datasets/facethink/gen/1.jpg'
    json_path = '~/datasets/facethink/gen/1.json' 

    im = plt.imread(image_path)
    with open(json_path) as f:
        ann = json.load(f)

