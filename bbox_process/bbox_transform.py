from ..cython.bbox import bbox_overlaps_cython

import numpy as np


def bbox_overlaps(boxes, query_boxes):
    return bbox_overlaps_cython(boxes, query_boxes)


def bbox_overlaps_py(boxes, query_boxes):
    """
    :param boxes: n*4 ground truth boxes
    :param query_boxes: k*4 query boxes
    :return: n*k overlaps
    """
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    overlaps = np.zeros(shape=(n, k), dtype=np.float)
    for i in range(n):
        box_area = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
        for j in range(k):
            iw = min(boxes[i, 2], query_boxes[j, 2]) - max(boxes[i, 0], query_boxes[j, 0]) + 1
            if iw > 0:
                ih = min(boxes[i, 3], query_boxes[j, 3]) - max(boxes[i, 1], query_boxes[j, 1]) + 1
                if ih > 0:
                    query_box_area = (query_boxes[j, 2] - query_boxes[j, 0] + 1) * (query_boxes[j, 3] - query_boxes[j, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[i, j] = iw * ih / all_area
    return overlaps


def clip_boxes(boxes, im_shape):
    """
    clip boxes to image boundaries.
    :param boxes: [N, 4*num_classes]
    :param im_shape: [tuple of 2]
    :return: [N, 4*num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 <= im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 <= im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def nonlinear_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths  = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x   = ex_rois[:, 0] + (ex_widths - 1.0) * 0.5
    ex_ctr_y   = ex_rois[:, 1] + (ex_heights - 1.0) * 0.5

    gt_widths  = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x   = gt_rois[:, 0] + (gt_widths - 1.0) * 0.5
    gt_ctr_y   = gt_rois[:, 1] + (gt_heights - 1.0) * 0.5

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    if gt_rois.shape[1] <= 4:
        targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
        return targets
    else:
        targets = [targets_dx, targets_dy, targets_dw, targets_dh]
        targets = np.vstack(targets).transpose()
        return targets


def landmark_transform(ex_rois, gt_rois):
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths  = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x   = ex_rois[:, 0] + (ex_widths - 1.0) * 0.5
    ex_ctr_y   = ex_rois[:, 1] + (ex_heights - 1.0) * 0.5

    targets = []
    for i in range(gt_rois.shape[1]):
        for j in range(gt_rois.shape[2]):
            if j == 2:
                continue
            if j == 0: ## w
                target = (gt_rois[:, i, j] - ex_ctr_x) / (ex_widths + 1e-14)
            elif j == 1: ## h
                target = (gt_rois[:, i, j] - ex_ctr_y) / (ex_heights + 1e-14)
            else: ## visible
                target = gt_rois[:, i, j]
            targets.append(target)

    targets = np.vstack(targets).transpose()
    return targets


def nonlinear_pred(bboxes, bbox_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if bboxes.shape[0] == 0:
        return np.zeros((0, bbox_deltas.shape[1]))

    bboxes  = bboxes.astype(np.float, copy=False)
    widths  = bboxes[:, 2] - bboxes[:, 0] + 1.0
    heights = bboxes[:, 3] - bboxes[:, 1] + 1.0
    ctr_x   = bboxes[:, 0] + (widths - 1.0) * 0.5
    ctr_y   = bboxes[:, 1] + (heights - 1.0) * 0.5

    dx = bbox_deltas[:, 0::4]
    dy = bbox_deltas[:, 1::4]
    dw = bbox_deltas[:, 2::4]
    dh = bbox_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w     = np.exp(dw) * widths[:, np.newaxis]
    pred_h     = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(shape=bbox_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - (pred_w - 1.0) * 0.5
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - (pred_h - 1.0) * 0.5
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + (pred_w - 1.0) * 0.5
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + (pred_h - 1.0) * 0.5

    return pred_boxes


def landmark_pred(boxes, lanrmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros(shape=(0, lanrmark_deltas.shape[1]))
    boxes   = boxes.astype(np.float, copy=False)
    widths  = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x   = boxes[:, 0] + (widths - 1.0) * 0.5
    ctr_y   = boxes[:, 1] + (heights - 1.0) * 0.5
    preds = []
    for i in range(lanrmark_deltas.shape[1]):
        if i % 2 == 0:
            pred = lanrmark_deltas[:, i] * widths + ctr_x
        else:
            pred = lanrmark_deltas[:, i] * heights + ctr_y
        preds.append(pred)
    preds = np.vstack(preds).transpose()
    return preds


def iou_tranform(ex_rois, gt_rois):
    """return bbox targets, IoU loss used gt_rois as gt"""
    assert ex_rois.shape[0] == gt_rois.shape[0]
    return gt_rois


