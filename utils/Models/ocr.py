import os
import sys
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils.Models.charnet.charnet.modeling.model import CharNet

def preprocess_img(im: np.ndarray | str, size: int = 2280):
    if isinstance(im, str):
        im = cv.imread(im)
    
    h, w, _ = im.shape
    scale = max(h, w) / float(size)
    image_resize_height = int(round(h / scale / 128) * 128)
    image_resize_width = int(round(w / scale / 128) * 128)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv.resize(im, (image_resize_width, image_resize_height), interpolation=cv.INTER_LINEAR)
    
    return {
        'im': im,
        'im_scale_w': scale_w,
        'im_scale_h': scale_h,
        'original_im_w': w,
        'original_im_h': h
    }

def visualize_boxes(img, word_instances):
    img_word_ins = img.copy()
    for word_ins in word_instances:
        word_bbox = word_ins.word_bbox
        cv.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)], True, (255, 0, 0), 4)
        cv.putText(
            img_word_ins,
            '{}'.format(word_ins.text),
            (int(word_bbox[0]), int(word_bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5
        )
    plt.imshow(img_word_ins)
    return img_word_ins