import torch
import numpy as np
from ensemble_boxes import *

def get_NonMaxSup_boxes(pred_dict):
        scores = pred_dict['scores']
        boxes = pred_dict['boxes']/1024
        labels = pred_dict['labels']
        lambda_nms = 0.3

        out_scores = []
        out_boxes = []
        out_labels = []
        for ix, (score, box, label) in enumerate(zip(scores,boxes,labels)):
            discard = False
            # for other_box in out_boxes:
            #     if intersection_over_union(box, other_box) > lambda_nms:
            #         discard = True
            #         break
            if not discard:
                out_scores.append(score.cpu().numpy().tolist())
                out_boxes.append((box.cpu().numpy()).tolist())
                out_labels.append(label.cpu().numpy().tolist())
        return {'scores':out_scores, 'boxes':out_boxes, 'labels':out_labels}

def get_weighted_boxes_fusion_boxes_4(pred_dict1, pred_dict2, pred_dict3, pred_dict4, device):
    labels_list = [pred_dict1['labels'], pred_dict2['labels'], pred_dict3['labels'], pred_dict4['labels']]
    scores_list = [pred_dict1['scores'], pred_dict2['scores'], pred_dict3['scores'], pred_dict4['scores']]
    boxes_list = [pred_dict1['boxes'], pred_dict2['boxes'], pred_dict3['boxes'], pred_dict4['boxes']]

    weights = [1, 1, 1, 1]
    iou_thr = 0.5
    skip_box_thr = 0.0001

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    scores = torch.tensor(np.array(scores)).to(device)
    boxes = torch.tensor(np.array(boxes)*1024).to(device)
    return {'scores': scores, 'boxes': boxes}

def get_weighted_boxes_fusion_boxes(pred_dict1, pred_dict2, device):
    if len(pred_dict1['boxes']) * len(pred_dict2['boxes']) == 0:
        scores = torch.tensor(np.array(pred_dict1['scores'] + pred_dict2['scores'])).to(device)
        boxes = torch.tensor(np.array(pred_dict1['boxes'] + pred_dict2['boxes']) * 1024).to(device)
        return {'scores': scores, 'boxes': boxes}

    else:
        labels_list = [pred_dict1['labels'], pred_dict2['labels']]
        scores_list = [pred_dict1['scores'], pred_dict2['scores']]
        boxes_list = [pred_dict1['boxes'], pred_dict2['boxes']]

        weights = [1, 1]
        iou_thr = 0.5
        skip_box_thr = 0.0001

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        scores = torch.tensor(np.array(scores)).to(device)
        boxes = torch.tensor(np.array(boxes)*1024).to(device)
        return {'scores': scores, 'boxes': boxes}

# Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou