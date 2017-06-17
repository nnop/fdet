import os
import os.path as osp

def make_if_not_exists(folder):
    if not osp.isdir(folder):
        os.makedirs(folder)
        print 'makedirs {}'.format(folder)

def check_f_exists(f):
    assert osp.isfile(f), '{} not exists'.format(f)

def check_d_exists(d):
    assert osp.isdir(d), '{} not exists'.format(d)

def main_name(p):
    fn = osp.split(p)[1]
    return osp.splitext(fn)[0]
