#!/bin/sh
find -L data/stuprt -name '*.jpg' | \
  xargs -n1 -P8 -I{} \
  tools/pose/xml_to_json.py \
    --class-list data/stuprt/classes.txt \
    --data-root data/stuprt {}
