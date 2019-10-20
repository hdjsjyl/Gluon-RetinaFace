from __future__ import print_function
from .imdb import IMDB
from PIL import Image
## from .ds_utils import unique_boxes, filter_small_boxes
## from .config import config

import os
import cv2
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle


class widerface(IMDB):
    def __init__(self, dataset, image_set, root_path='', dataset_path=''):
        super(widerface, self).__init__('widerface', image_set, root_path, dataset_path)

        self._image_set    = image_set
        self._dataset      = dataset
        self._root_path    = root_path
        self._dataset_path = dataset_path
        self._imgs_path    = os.path.join(self._dataset_path, image_set, 'images')

        self._fp_bbox_map  = {}
        self._label_file   = os.path.join(self._dataset_path, image_set, 'label.txt')
        temp_name = None
        for line in open(self._label_file, 'r'):
            line = line.strip()
            if line.startswith('#'):
                temp_name = line[1:].strip()
                self._fp_bbox_map[temp_name] = []
                continue
            assert temp_name is not None
            assert temp_name in self._fp_bbox_map
            self._fp_bbox_map[temp_name].append(line)
        print('origin image size: ', len(self._fp_bbox_map))

        self.classes = ['bg', 'face']
        self.num_classes = len(self.classes)

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, '{}_{}_gt_roidb.pkl'.format(self._dataset, self._image_set))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                roidb = pickle.load(file)
            print('{} {} gt roidb loaded from {}'.format(self._dataset, self._image_set, cache_file))
            self.num_images = len(roidb)
            return roidb

        roidb           = []
        max_num_boxes   = 0
        nonattr_box_num = 0
        landmark_num    = 0

        for fp in self._fp_bbox_map:
            if self._image_set == 'test':
                image_path = os.path.join(self._imgs_path, fp)
                roi = {'image': image_path}
                roidb.append(roi)
                continue
            boxes      = np.zeros([len(self._fp_bbox_map[fp]), 4], np.float)
            landmarks  = np.zeros([len(self._fp_bbox_map[fp]), 5, 3], np.float)
            blur       = np.zeros((len(self._fp_bbox_map[fp]), ), np.float)
            boxes_mask = []

            gt_classes = np.ones([len(self._fp_bbox_map[fp])], np.int32)
            overlaps   = np.zeros([len(self._fp_bbox_map[fp]), 2], np.float)

            ix = 0
            for aline in self._fp_bbox_map[fp]:
                imsize = Image.open(os.path.join(self._imgs_path, fp)).size
                values = [float(x) for x in aline.strip().split()]
                bbox   = [values[0], values[1], values[0] + values[2], values[1] + values[3]]
                x1     = bbox[0]
                y1     = bbox[1]
                x2     = min(imsize[0], bbox[2])
                y2     = min(imsize[1], bbox[3])
                if x1 >= x2 or y1 >= y2:
                    continue

                ## bbox mask setting
                # if bbox_mask_thr > 0:
                #     if x2 - x1 < bbox_mask_thr or y2 - y1 < bbox_mask_thr:
                #         boxes_mask.append(np.array([x1, y1, x2, y2], np.float))
                #         continue
                ## min bbox setting
                # if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
                #     continue

                boxes[ix, :] = np.array([x1, y1, x2, y2], np.float)
                if self._image_set == 'train':
                    landmark = np.array(values[4:19], dtype=np.float32).reshape((5, 3))
                    for li in range(5):
                        if landmark[li][0] == -1. and landmark[li][1] == -1.: ## missing landmark
                            assert landmark[li][2] == -1.
                        else:
                            assert landmark[li][2] >= 0
                            if li == 0:
                                landmark_num += 1
                            if landmark[li][2] == 0.0: ## visible
                                landmark[li][2] = 1.0
                            else:
                                landmark[li][2] = 0.0
                    landmarks[ix] = landmark

                    blur[ix] = values[19]
                    if blur[ix] < 0:
                        blur[ix] = 0.3
                        nonattr_box_num += 1

                cls = int(1)
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                ix += 1
            max_num_boxes = max(max_num_boxes, ix)

            if self._image_set == 'train' and ix == 0:
                continue
            boxes      = boxes[:ix, :]
            landmarks  = landmarks[:ix, :, :]
            blur       = blur[:ix]
            gt_classes = gt_classes[:ix]
            overlaps   = overlaps[:ix, :]
            image_path = os.path.join(self._imgs_path, fp)
            with open(image_path, 'rb') as fin:
                stream = fin.read()
            stream = np.fromstring(stream, dtype=np.uint8)
            roi = {
                'image': image_path,
                'stream': stream,
                'height': imsize[1],
                'width': imsize[0],
                'boxes': boxes,
                'landmarks': landmarks,
                'blur': blur,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'max_classes': overlaps.argmax(axis=1),
                'max_overlaps': overlaps.max(axis=1),
                'flipped': False}
            if len(boxes_mask) > 0:
                boxes_mask = np.array(boxes_mask)
                roi['boxes_mask'] = boxes_mask
            roidb.append(roi)
        for roi in roidb:
            roi['max_num_boxes'] = max_num_boxes
        self.num_images = len(roidb)
        print('roidb size', len(roidb))
        print('non attr box num', nonattr_box_num)
        print('landmark num', landmark_num)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return roidb