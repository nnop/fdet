import os.path as osp
import sys

def add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)

# add lib path
this_dir = osp.split(osp.realpath(__file__))[0]
lib_dir = osp.normpath(osp.join(this_dir, '..', 'lib'))
assert osp.isdir(lib_dir), '{} not exists.'.format(lib_dir)
add_path(lib_dir)
