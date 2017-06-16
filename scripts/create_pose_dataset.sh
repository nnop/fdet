#!/bin/sh
# NOTE: the labelmap file is for detection, the pose cls info is in json file
max_dim=512
data_root_dir="data/stuprt"
label_map_file="data/stuprt/labelmap_det.prototxt"
subset=train
python $CAFFEROOT/scripts/create_annoset.py --shuffle --check-label \
  --anno-type=detection --label-type=json \
  --max-dim=512 \
  --encoded --encode-type=jpg \
  --label-map-file=$label_map_file \
  --root=./ \
  --listfile=$data_root_dir/list/$subset.txt \
  --outdir=$data_root_dir/lmdb/$subset
