#!/bin/bash
lmdb_dir="data/pose/lmdb"
if [[ ! -d $lmdb_dir ]]; then
  mkdir $lmdb_dir
fi

for split in train val test; do
  python $CAFFEROOT/scripts/create_annoset.py --anno-type=detection \
    --label-type=json --encode-type=jpg --min-dim=512 \
    --label-map-file="data/pose/labelmap_person.prototxt" \
    --root="data/pose/" \
    --listfile="data/pose/$split""_list.txt" \
    --outdir="$lmdb_dir/$split""_lmdb"
done
