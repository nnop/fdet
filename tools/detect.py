#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2

def show_detection_results(image, bboxes, labelmap=[], threshold=0.0):
    """
    bboxes shape: (N, 7)
    """
    num_class = len(labelmap)
    if num_class:
        colors = plt.cm.hsv(np.linspace(0, 1, num_class))
    plt.imshow(image)
    ax = plt.gca()
    if threshold:
        bboxes = bboxes[bboxes[:, 2] >= threshold]
    im_hei, im_wid = image.shape[:2]
    bboxes[:, 3] = bboxes[:, 3] * im_wid
    bboxes[:, 4] = bboxes[:, 4] * im_hei
    bboxes[:, 5] = bboxes[:, 3] * im_wid
    bboxes[:, 6] = bboxes[:, 6] * im_hei
    for _, label, conf, xmin, ymin, xmax, ymax in bboxes:
        coords = (xmin, ymin), (xmax - xmin + 1), (ymax - ymin + 1)
        if num_class:
            c = colors[int(label)]
        else:
            c = np.random.random(3)
        ax.add_patch(plt.Rectangle(*coords, fill=False, linewidth=2, edgecolor=c))

class Detector(object):
    def __init__(self, model_file, weights_file):
        # load net
        self.net = caffe.Net(model_file, weights_file, caffe.TEST)
        # create transformer
        self.transformer = caffe.io.Transformer(
                inputs={'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', np.array([104, 117, 123]))
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_channel_swap('data', (2, 1, 0))

    def _load_data(self, imagepath):
        """
        load image and preprocess
        """
        im = cv2.imread(imagepath)
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)

    def detect(self, imagepath, threshold=0.0):
        self._load_data(imagepath)
        det_res = self.net.forward()['detection_out'][0, 0]
        if threshold:
            det_res = det_res[det_res[:, 2] >= threshold]
        return det_res

def load_labelmap(labelmap_file):
    labelmap = caffe_pb2.LabelMap()
    with open(labelmap_file) as f:
        text_format.Merge(str(f.read()), labelmap)
    labelmap = dict([(item.label, item.name) for item in labelmap.item])
    return labelmap

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
    show_detection_results(im, det_res, labelmap)
    if args.save_result:
        plt.savefig(args.save_result)
    else:
        plt.show()
