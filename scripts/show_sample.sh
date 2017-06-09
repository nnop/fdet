#!/bin/sh
if [[ ! $# -eq 1 ]]; then
  echo "Usage: $0 sample_id"
  exit 1
fi
tools/show_sample.py --data-dir ~/datasets/facethink/headloc/ \
  --sample-id $1
