from os import path
import sys

def add_path(p):
    if path not in sys.path:
        sys.path.insert(0, p)

# add lib path
this_dir = path.split(path.realpath(__file__))[0]
lib_dir = path.normpath(path.join(this_dir, '..', 'lib'))
add_path(lib_dir)
