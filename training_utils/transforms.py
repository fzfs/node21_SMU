import random
import torch
import numpy as np

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomCrop(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if len(target["boxes"]) > 0:
            if random.random() < self.prob:
                height, width = image.shape[-2:]
                bboxes = target["boxes"].cpu().numpy()

                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
                max_l_trans = max_bbox[0] - 10
                max_u_trans = max_bbox[1] - 10
                max_r_trans = width - max_bbox[2] - 10
                max_d_trans = height - max_bbox[3] - 10

                crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
                crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
                crop_xmax = min(width, int(max_bbox[2] + random.uniform(0, max_r_trans)))
                crop_ymax = min(height, int(max_bbox[3] + random.uniform(0, max_d_trans)))

                image = image[:, crop_ymin: crop_ymax, crop_xmin: crop_xmax]

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

                target["boxes"] = torch.tensor(bboxes)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
