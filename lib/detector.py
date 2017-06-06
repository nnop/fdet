import caffe
import cv2
import numpy as np

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
