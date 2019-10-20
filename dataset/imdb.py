"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path. As to training, it is necessary that ground
truth and proposals are mixed together for training.
roidb
basic format [image_index]
['image', 'height', 'width', 'flipped',
'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

from ..bbox_process.bbox_transform import bbox_overlaps

import os
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

class IMDB(object):
    def __init__(self, dataset, image_set, root_path, dataset_path):
        """
        basic information about an image database
        :param dataset: dataset name
        :param image_set: image set name
        :param root_path: root path of storing cache and proposal data
        :param dataset_path: dataset path of storing images and images lists
        """
        self.name         = dataset + '_' + image_set
        self.image_set    = image_set
        self.root_path    = root_path
        self.dataset_path = dataset_path

        self.classes         = []
        self.num_classes     = 0
        self.image_set_index = []
        self.num_images      = 0

        self.config = {}

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.root_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def append_flipped_images(self, roidb):
        assert self.num_images == len(roidb), 'self.num_images is equal to length of roidb'
        for i in range(self.num_images):
            roi_rec = roidb[i]
            entry = {'image': roi_rec['image'],
                     'stream': roi_rec['stream'],
                     'height': roi_rec['height'],
                     'width': roi_rec['width'],
                     'gt_classes': roidb[i]['gt_classes'],
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'flipped': True}

            for j in roi_rec:
                if not j.startswith('boxes'):
                    continue
                boxes = roi_rec['boxes'].copy()
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = roi_rec['width'] - oldx2 - 1
                boxes[:, 2] = roi_rec['width'] - oldx1 - 1
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                entry[j] = boxes

            ## add landmarks
            # if 'landmarks' in roi_rec:
            #     landmarks = roi_rec['landmarks'].copy()
            #     landmarks[:, :, 0] *= -1
            #     landmarks[:, :, 0] += (roi_rec['width'] - 1)
            #     order = [1, 0, 2, 4, 3]
            #     flipped_landmarks = landmarks.copy()
            #     for idx, a in enumerate(order):
            #         flipped_landmarks[:, idx, :] = landmarks[:, a, :]
            #     entry['landmarks'] = flipped_landmarks

            ## add blur
            # if 'blur' in roi_rec:
            #     entry['blur'] = roi_rec['blur']
            # roidb.append(entry)

        self.image_set_index *= 2
        return roidb


    @staticmethod
    def merge_roidbs(a, b):
        """
        merge roidbs into one
        :param a: roidb to be merged into
        :param b: roidb to be merged
        :return: merged roidb
        """
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.vstack((a[i]['gt_classes'], b[i]['gt_classes']))
            a[i]['gt_overlaps'] = np.vstack((a[i]['gt_overlaps'], b[i]['gt_overlaps']))
            a[i]['max_classes'] = np.vstack((a[i]['max_classes'], b[i]['max_classes']))
            a[i]['max_overlaps'] = np.vstack((a[i]['max_overlaps'], b[i]['max_overlaps']))
        return a

