import numpy as np
import matplotlib.pyplot as plt

def show_detection(image, bboxes, labelmap=[], threshold=0.0):
    """
    bboxes shape: (N, 7)
    """
    num_class = len(labelmap)
    if num_class:
        colors = plt.cm.hsv(np.linspace(0, 1, num_class))
    plt.imshow(image)
    ax = plt.gca()
    if threshold:
        bboxes = bboxes[bboxes[:, 2] >= threshold]
    im_hei, im_wid = image.shape[:2]
    bboxes[:, 3] = bboxes[:, 3] * im_wid
    bboxes[:, 4] = bboxes[:, 4] * im_hei
    bboxes[:, 5] = bboxes[:, 3] * im_wid
    bboxes[:, 6] = bboxes[:, 6] * im_hei
    for _, label, conf, xmin, ymin, xmax, ymax in bboxes:
        coords = (xmin, ymin), (xmax - xmin + 1), (ymax - ymin + 1)
        if num_class:
            c = colors[int(label)]
        else:
            c = np.random.random(3)
        ax.add_patch(plt.Rectangle(*coords, fill=False, linewidth=2, edgecolor=c))
