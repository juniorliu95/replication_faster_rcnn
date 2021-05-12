import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from utils.anchors import generate_anchor_base, generate_anchors
from utils.utils import reg2bbox
from utils.utils import nms_thresh, n_train_pre_nms, n_train_post_nms,n_test_pre_nms, n_test_post_nms


def normal_init(m, mean, stddev, truncated=False):
    """for initialization of the RPN layers"""
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class region_proposal(nn.Module):
    "generate roi from selected region proposal"
    def __init__(self, mode, nms_thresh=nms_thresh,
                 n_train_pre_nms=n_train_pre_nms,
                 n_train_post_nms=n_train_post_nms,
                 n_test_pre_nms=n_test_pre_nms,
                 n_test_post_nms=n_test_post_nms,
                 min_size=16):
        """
        mode: training, test
        nms_thresh: the threshold of IoU for nms
        pre_nms: # of boxes before nms
        post_nms: # of boxes after nms
        min_size: min_size of the boxes
        """

        super(region_proposal, self).__init__()
        self.mode = mode # ['train', 'test']

        self.pre_nms = n_train_pre_nms
        self.post_nms = n_train_post_nms
        if mode == 'test':
            self.pre_nms = n_test_pre_nms
            self.post_nms = n_test_post_nms
        self.nms_threshold = nms_thresh
        self.min_size = min_size

    def __call__(self, anchors, cls_fg_softmax, reg, img_w, img_h):
        """
        Inputs:
            anchors: the anchors on the image
            cls_fg_softmax: the foreground output of the cls head of RPN, softmax
            reg: the output of the reg head of RPN
            img_w, img_h: the size of the images
        Outputs:
            selected rois [post_nms, [x1, y1, x2, y2]]
        """
        
        anchors = torch.from_numpy(anchors)
        # transfer the regression outputs to bounding boxes
        bbox = reg2bbox(anchors, reg) # [h*w, 4]
        # avoid out of the image
        bbox[:,[0,2]] = torch.clamp(bbox[:,[0,2]], min=0, max=img_h)
        bbox[:,[1,3]] = torch.clamp(bbox[:,[1,3]], min=0, max=img_w)
        # select boxes larger than min_size
        select_scale = torch.where((bbox[:,2]-bbox[:,0]>=self.min_size) & (bbox[:,3]-bbox[:,1]>=self.min_size))[0]

        bbox = bbox[select_scale, :]
        cls_fg_softmax = cls_fg_softmax[select_scale]

        # select the boxes by nms
        rank = torch.argsort(cls_fg_softmax, descending=True)
        rank = rank[:self.pre_nms]
        cls_fg_softmax = cls_fg_softmax[rank]
        bbox = bbox[rank, :]  
        nms_ind = nms(bbox, cls_fg_softmax, iou_threshold=self.nms_threshold)
        roi = bbox[nms_ind]
        roi = roi[:self.post_nms]

        return roi


class RPN(nn.Module):
    """region proposal network"""
    def __init__(self, in_channels=256, mid_channels=256, ratios=[0.5, 1., 2.], anchor_scales=[8, 16, 32], feat_stride=16, mode = "training"):
        super(RPN, self).__init__()
        self.base_anchor = generate_anchor_base(ratios=ratios, anchor_scales=anchor_scales)
        self.K = self.base_anchor.shape[0] # number of anchors in a anchor base
        self.feat_stride = feat_stride
        
        self.proposal_layer = region_proposal(mode)

        # layers after conv5
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.cls = nn.Conv2d(in_channels=mid_channels, out_channels=self.K*2, kernel_size=1, stride=1, padding=0)
        self.reg = nn.Conv2d(in_channels=mid_channels, out_channels=self.K*4, kernel_size=1, stride=1, padding=0)

        # channels inistalization
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.cls, 0, 0.01)
        normal_init(self.reg, 0, 0.01)

    def forward(self, x, img_width, img_height):
        """
        get the cls and reg head output of the rpn, and select the rois
        Inputs:
            img_widths, img_heights: the image size
        Outputs:
            cls: output of the cls head. [N,2, KHW]
            reg: output of the cls head. [N,KHW, 4]
            rois: selected rois. [N*post_thre, 4]
            roi_inds: batch index of the selected rois. [N*post_thre]
        """
        n_img, n_channel, conv_h, conv_w = x.shape
        x = self.conv1(x)
        x = F.relu(x)

        cls = self.cls(x)
        cls = cls.permute(0,2,3,1).contiguous().view(n_img, -1, 2)
        cls_fg_softmax = F.softmax(cls, dim=-1)[:,:,1].contiguous() # [N,w*h,1]
        cls_fg_softmax = cls_fg_softmax.view(n_img, -1)
        cls = cls.permute(0, 2, 1) # ![N, C, KWH] for loss calculation

        reg = self.reg(x) # [N,K*4,H,W]
        reg = reg.permute(0,2,3,1).contiguous().view(n_img, -1, 4) # [N,KHW,4]

        # generate the base anchors from the conv5 layer
        self.anchors = generate_anchors(self.base_anchor, self.feat_stride, width=conv_w, height=conv_h)

        rois = [] # the selected rois
        roi_inds = [] # mark each roi belong to which image
        for img_ind in range(n_img):
            roi = self.proposal_layer(self.anchors, cls_fg_softmax[img_ind], reg[img_ind], img_width, img_height) # [post_thre, 4]
            rois.append(roi)
            roi_inds.append(img_ind*torch.ones(len(roi)))
        rois = torch.cat(rois, dim=0) # [N*post_thre, 4]
        roi_inds = torch.cat(roi_inds, dim=0) # [N*post_thre]

        return cls, reg, rois, roi_inds, self.anchors



