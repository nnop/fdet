#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import cv2
import matplotlib.pyplot as plt

import _init_paths
from detector import Detector
from vis import show_detection
from utils import load_labelmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('imagepath', help='image path')
    parser.add_argument('--model-file')
    parser.add_argument('--weights-file')
    parser.add_argument('--labelmap-file')
    parser.add_argument('--threshold', default=0.1, help='confidence threshold')
    parser.add_argument('--save-result', help='save result to disk.')
    args = parser.parse_args()

    # load labelmap file
    if args.labelmap_file:
        labelmap = load_labelmap(args.labelmap_file)

    detor = Detector(args.model_file, args.weights_file)
    det_res = detor.detect(args.imagepath, args.threshold)
    im = cv2.imread(args.imagepath)[:, :, [2, 1, 0]]
    show_detection(im, det_res, labelmap)
    if args.save_result:
        plt.savefig(args.save_result)
    else:
        plt.show()
