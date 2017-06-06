from google.protobuf import text_format
from caffe.proto import caffe_pb2

def load_labelmap(labelmap_file):
    labelmap = caffe_pb2.LabelMap()
    with open(labelmap_file) as f:
        text_format.Merge(str(f.read()), labelmap)
    labelmap = dict([(item.label, item.name) for item in labelmap.item])
    return labelmap

