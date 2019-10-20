from __future__ import print_function
import numpy as np
import random
import math
import sys
import cv2
import os


def brightness_aug(src, x):
    alpha = 1.0 + random.uniform(-x, x)
    src  *= alpha
    return src


def contrast_aug(src, x):
    alpha = 1.0 + random.uniform(-x, x)
    coef  = np.array([[[0.299, 0.587, 0.114]]])
    gray  = src * coef
    gray  = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    src  *= alpha
    src  += gray
    return src


def saturation_aug(src, x):
    alpha = 1.0 + random.uniform(-x, x)
    coef  = np.array([[[0.299, 0.587, 0.114]]])
    gray  = src * coef
    gray  = np.sum(gray, axis=2, keepdims=True)
    gray *= (1.0 - alpha)
    src  *= alpha
    src  += gray
    return src


def color_aug(img, x, color_mode):
    if color_mode > 1:
        augs = [brightness_aug, contrast_aug, saturation_aug]
        random.shuffle(augs)
    else:
        augs = [brightness_aug]
    for aug in augs:
        img = aug(img, x)
    return img


# def get_image(roidb, scale=False):
