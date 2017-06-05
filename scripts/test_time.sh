#!/bin/sh
$CAFFEROOT/build/tools/caffe time -gpu 0 -iterations 10 \
  -model models/SSD_300x300_ft/deploy.prototxt \
  -weights models/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel \
  2>&1 | tee experiments/time_eval/test_time.log
