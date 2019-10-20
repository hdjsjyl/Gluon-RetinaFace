from dataset import *

import numpy as np
import os


def load_gt_roidb(dataset, image_set, root_path='', dataset_path='', flip=False):
    """load ground truth roidb"""
    imdb  = eval(dataset)(image_set, root_path, dataset_path)
    roidb = imdb.gt_roidb()
    print('image size: ', len(roidb))
    if flip:
        roidb = imdb.append_flipped_images(roidb)
        print('flipped image size: ', len(roidb))
    return roidb


def merge_roidb(roidbs):
    """roidbs are list, concat them together"""
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb, fg_thr, bg_hi_thr, bg_lo_thr):
    """remove roidb entries without usable rois"""

    def is_valid(entry):
        """valid images have at least 1 fg or bg roi"""
        overlaps = entry['max_overlaps']
        fg_inds  = np.where(overlaps >= fg_thr)[0]
        bg_inds  = np.where((overlaps >= bg_lo_thr) & (overlaps < bg_hi_thr))[0]
        valid    = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num              = len(roidb)
    filter_roidb     = [entry for entry in roidb if is_valid(entry)]
    num_after_filter = len(filter_roidb)
    print('load data: filtered %d roidb entries: %d -> %d' % (num - num_after_filter, num, num_after_filter))
    return filter_roidb