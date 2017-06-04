#!/bin/sh
python tools/detect.py --model-file models/SSD_300x300_ft/deploy.prototxt \
  --weights-file models/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel \
  --labelmap-file data/labelmap_voc.prototxt \
  data/VOC2007/JPEGImages/000001.jpg
