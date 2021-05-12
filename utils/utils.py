import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

# rpn
nms_thresh=0.7
n_train_pre_nms=12000
n_train_post_nms=600
n_test_pre_nms=3000
n_test_post_nms=300
min_size=16

# VOC dataset
PASCAL_VOC_CLASSES = ['__background__',
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']
PASCAL_VOC_NUM_CLASSES = 20 + 1  # 20 object classes and 1 background



def new_img_size(height, width, min_side=600):
    """
    [abandoned]
    calculate the new width and height of the input image. minimize=600
    Inputs:
        width, height: size of the image
        min_side: min side length of the resized image
    Outputs:
        the width and height of the resized image.
    """

    if width > height:
        ratio = height/600
        height = 600
        width = int(width/ratio)
    else:
        ratio = width/600
        width = 600
        height = int(height/ratio)

    return height, width

def reg2bbox(anchors, reg):
    """
    transfer the output of regressor head into bounding boxes
    Inputs:
        anchors: the anchors of a image, [xmin, ymin, xmax, ymax]
        reg: output of the rgressor head [dx, dy, dh, dw]
    Outputs:
        bboxes from the regressor, [xmin, ymin, xmax, ymax]
    """

    anchor_h = anchors[:,2]-anchors[:,0]
    anchor_w = anchors[:,3]-anchors[:,1]
    anchor_cx = (anchors[:,2]+anchors[:,0]) / 2
    anchor_cy = (anchors[:,1]+anchors[:,3]) / 2
    # transfer reg to xywh
    x = reg[:, 0]*anchor_h+anchor_cx # dx = (x-cx)/ha
    y = reg[:, 1]*anchor_w+anchor_cy # dy = (y-cy)/wa
    h = torch.exp(reg[:, 2])*anchor_h # dh = log(h/ha)
    w = torch.exp(reg[:, 3])*anchor_w # dw = log(w/wa)
    # transfer xywh to x1y1x2y2
    bbox = torch.zeros(reg.shape) 
    bbox[:, 0] = x - h*.5
    bbox[:, 1] = y - w*.5
    bbox[:, 2] = x + h*.5
    bbox[:, 3] = y + w*.5

    return bbox

def bbox2reg(anchors, bbox):
    """
    transfer the bbox back to output of regressor
    Inputs:
        anchors: the anchors of a image, [xmin, ymin, xmax, ymax]
        bbox: output of the rgressor head, [xmin, ymin, xmax, ymax]
    Outputs:
        bboxes from the regressor, [dx, dy, dh, dw]
    """
    anchor_h = anchors[:,2]-anchors[:,0]
    anchor_w = anchors[:,3]-anchors[:,1]
    anchor_cx = (anchors[:,2]+anchors[:,0]) / 2
    anchor_cy = (anchors[:,1]+anchors[:,3]) / 2

    bbox_h = bbox[:,2]-bbox[:,0]
    bbox_w = bbox[:,3]-bbox[:,1]
    bbox_cx = (bbox[:,2]+bbox[:,0]) / 2
    bbox_cy = (bbox[:,1]+bbox[:,3]) / 2

    reg = np.zeros(bbox.shape)
    reg[:, 0] = (bbox_cx - anchor_cx) / anchor_h
    reg[:, 1] = (bbox_cy - anchor_cy) / anchor_w
    reg[:, 2] = np.log(bbox_h/anchor_h)
    reg[:, 3] = np.log(bbox_w/anchor_w)

    return reg

def bbox_iou(bbox_a, bbox_b):
    """
    calculate the iou between two list of boxes
    Inputs:
        bbox: [N_i, 4] 
    Outputs:
        [N_a, N_b]
    """

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod((br - tl), axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        """
        Inputs:
            n_sample: the number of samples in a image
            pos_iou_thresh: positive samples should be larger than it
            neg_iou_thresh: negative samples should be smaller than it
            pos_ratio: the ratio of positive samples
        """

        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor):
        """
        reg_targets are the closest true boxes for all the anchors
        But only part of the labels are positive
        # bbox: the generated boxes
        # anchor: the anchors on the RPN
        """

        argmax_ious, label = self._create_label(anchor, bbox)
        if (label>0).any():
            reg = bbox2reg(anchor, bbox[argmax_ious])
            return reg, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        """
        Outputs:
            argmax_ious: best bbox positions for all anchors
            max_ious: roi of the best bbox with all anchors
            gt_argmax_ious: the position of the closest anchor for all bboxes
        """

        # calculate the iou between anchors and the bboxes, [KHW, max_n_bbox]
        ious = bbox_iou(anchor, bbox)
        if len(bbox)==0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        # the position of best bbox for all anchors, [n_anchor]
        argmax_ious = ious.argmax(axis=1)
        # the roi of best bbox for all anchors, [n_anchor]
        max_ious = np.max(ious, axis=1)
        # the position of the closest anchor for all bboxes, [n_bbox]
        gt_argmax_ious = ious.argmax(axis=0)
        # all the bboxes should match at least an anchor
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious
        
    def _create_label(self, anchor, bbox):
        # first make all labels ignorable (-1)
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        # negative smaples
        label[max_ious < self.neg_iou_thresh] = 0
        # positive samples
        label[max_ious >= self.pos_iou_thresh] = 1
        # all anchors selected by the bboxes are positive
        if len(gt_argmax_ious)>0:
            label[gt_argmax_ious] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        # randomly negatify some positive samples if too many
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # if too many negative, randomly select some to be -1
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, reg_normalize_mean=(0., 0., 0., 0.), reg_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """
        select rois feed to the roi pooling. true boxes are also used to train the heads
        Inputs:
            roi: the generated boxes
            bbox: the true boxes
            label: true labels of the boxes
        Outputs:
            sample_roi: rois to train the heads [n_sample, 4]
            gt_roi_reg: true regs of the heads [n_sample, 4]
            gt_roi_label: true labels of the heads [n_sample, ]
        """
        
        # also add the true boxes for better training of the head
        roi = np.concatenate((roi.cpu().numpy(), bbox), axis=0)
        # [n_roi+n_bbox, n_bbox]
        iou = bbox_iou(roi, bbox)
        
        if len(bbox)==0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # get the position of best bboxes for all rois
            gt_assignment = iou.argmax(axis=1)
            # get the best iou with the bboxes for all rois
            max_iou = iou.max(axis=1)

            # the labels for rois. don't forget background class
            gt_roi_label = label[gt_assignment]

        #positive samples
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # negative samples
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        #---------------------------------------------------------#
        #   sample_roi      [n_sample, 4]
        #   gt_roi_reg      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        #---------------------------------------------------------#
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox)==0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_reg = bbox2reg(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_reg = ((gt_roi_reg - np.array(reg_normalize_mean, np.float32)) / np.array(reg_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_reg, gt_roi_label



if __name__ == "__main__":
    # test iou
    anchors = np.array([[1,2,3,4],[3,5,7,8],[-1,-1,-1,-1],[3,2,4,5]])
    bboxes = np.array([[2,3,4,5],[5,6,7,8],[1,2,3,4]])
    print(bbox_iou(anchors, bboxes))