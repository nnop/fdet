#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

def plot_time(time_info):
    n, t = zip(*time_info[:50])
    plt.bar(np.arange(len(t)), t)
    plt.xticks(np.arange(len(t)), n)
    plt.xlabel('layer names')
    plt.ylabel('time (ms)')
    plt.gcf().autofmt_xdate()
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.label.set_fontsize(6)
    for t in ax.yaxis.get_major_ticks():
        t.label.set_fontsize(8)

if __name__ == "__main__":
    out_dir = 'experiments/time_eval/'
    with open(path.join(out_dir, 'time_stats.txt')) as f:
        lines = [l.strip() for l in f if l[0] != '#']

    fwd_time = []
    bwd_time = []
    for l in lines:
        name, tp = l.split(':')[0].split()
        t, _ = l.split(':')[1].split()
        if tp == 'forward':
            fwd_time.append((name, float(t)))
        elif tp == 'backward':
            bwd_time.append((name, float(t)))
    fwd_time.sort(key=lambda x: x[1], reverse=True)
    bwd_time.sort(key=lambda x: x[1], reverse=True)

    fig = plt.figure()
    plot_time(fwd_time)
    plt.savefig(path.join(out_dir, 'fwd.pdf'))

    fig = plt.figure()
    plot_time(bwd_time)
    plt.savefig(path.join(out_dir, 'fwd.pdf'))
