#!/bin/bash
lmdb_dir="data/pascal3d/lmdb"
if [[ ! -d $lmdb_dir ]]; then
  mkdir $lmdb_dir
fi

for split in train val test; do
  python $CAFFEROOT/scripts/create_annoset.py --anno-type=detection \
    --label-type=json --encode-type=jpg \
    --label-map-file="data/pascal3d/labelmap_3D.prototxt" \
    --root="data/pascal3d/" \
    --listfile="data/pascal3d/$split""_list.txt" \
    --outdir="$lmdb_dir/$split""_lmdb"
done
