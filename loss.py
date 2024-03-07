# References:
    # https://herbwood.tistory.com/3
    # https://github.com/KimRass/train_easyocr/blob/main/evaluate.py

import torch
from torchmetrics.detection import IntersectionOverUnion
from torchvision.ops import box_iou


torch.set_printoptions(linewidth=70)


def get_area(bbox):
    return torch.clip(
        bbox[:, 2] - bbox[:, 0], min=0
    ) * torch.clip(bbox[:, 3] - bbox[:, 1], min=0)


def get_intersection_area(bbox1, bbox2):
    l = torch.maximum(bbox1[:, 0][:, None], bbox2[:, 0][None, :])
    t = torch.maximum(bbox1[:, 1][:, None], bbox2[:, 1][None, :])
    r = torch.minimum(bbox1[:, 2][:, None], bbox2[:, 2][None, :])
    b = torch.minimum(bbox1[:, 3][:, None], bbox2[:, 3][None, :])
    return torch.clip(r - l, min=0) * torch.clip(b - t, min=0)


def get_iou(bbox1, bbox2):
    bbox1_area = get_area(bbox1)
    bbox2_area = get_area(bbox2)
    intersec_area = get_intersection_area(bbox1, bbox2)
    union_area = bbox1_area[:, None] + bbox2_area[None, :] - intersec_area
    return torch.where(union_area == 0, 0, intersec_area / union_area)


def get_smallest_enclosing_area(bbox1, bbox2):
    l = torch.minimum(bbox1[:, 0][:, None], bbox2[:, 0][None, :])
    t = torch.minimum(bbox1[:, 1][:, None], bbox2[:, 1][None, :])
    r = torch.maximum(bbox1[:, 2][:, None], bbox2[:, 2][None, :])
    b = torch.maximum(bbox1[:, 3][:, None], bbox2[:, 3][None, :])
    return torch.clip(r - l, min=0) * torch.clip(b - t, min=0)


def get_giou(bbox1, bbox2):
    bbox1_area = get_area(bbox1)
    bbox2_area = get_area(bbox2)
    intersec_area = get_intersection_area(bbox1, bbox2)
    union_area = bbox1_area[:, None] + bbox2_area[None, :] - intersec_area
    c = get_smallest_enclosing_area(bbox1, bbox2)
    iou = torch.where(union_area == 0, 0, intersec_area / union_area)
    return torch.where(c == 0, -1, iou - ((c - union_area) / c))


img_size = 64
n_classes = 10
n = 16
m = 512
iou_thresh = 0.5
gt = torch.cat(
    [
        torch.randint(0, img_size, size=(n, 4)),
        torch.randint(0, n_classes, size=(n, 1))
    ],
    dim=1,
)
pred = torch.cat(
    [
        torch.randint(0, img_size, size=(m, 4)),
        torch.rand(size=(m, 1)),
        torch.randn(size=(m, n_classes)),
    ],
    dim=1,
)
pred[:, -n_classes:] = torch.softmax(pred[:, -n_classes:], dim=1)
sorted_pred = pred[torch.sort(pred[:, 4], dim=0, descending=True)[1]]

gt_bbox = gt[gt[:, 4] == c][:, : 4]
pred_bbox = sorted_pred[torch.argmax(sorted_pred[:, -n_classes:], dim=1) == c][:, : 4]
iou = get_iou(gt[:, : 4], sorted_pred[:, : 4])
iou *= 10

for pred_idx in range(pred_bbox.size(0)):
    # pred_bbox[pred_idx]
    pred_idx = 3
    gt[iou[:, pred_idx] >= iou_thresh]




# gt
# iou_thresh = 0.1
# for c in range(n_classes):
#     # c = 0
#     gt_bbox = gt[gt[:, 4] == c][:, : 4]
#     pred_bbox = sorted_pred[torch.argmax(sorted_pred[:, -n_classes:], dim=1) == c][:, : 4]

#     iou = get_iou(gt_bbox, pred_bbox)
#     iou *= 10
#     if iou.sum().item() != 0 and c not in [2, 4]:
#         break
#     iou.shape, pred_bbox.shape
#     (iou >= iou_thresh).shape
#     iou[:, 5: 10]


#     iou[[iou >= iou_thresh]].shape



# iou = get_iou(bbox1, bbox2)
# giou = get_giou(bbox1, bbox2)

# (giou >= iou_thresh)
# giou.max(dim=1)[0]
