import torch
import torch.nn as nn
from torchvision.ops.roi_pool import roi_pool



class ResnetHead(nn.Module):
    def __init__(self, classifier, roi_size=7, spatial_scale=1, n_classes=21):
        """
        Head of the faster rcnn with resnet backbone
        Inputs:
            classifier: classifier from the resnet
            roi_size: size of the roi pooled features
            spatial_scales
        """
        super(ResnetHead, self).__init__()

        # the classifier of backbone. for feature extraction
        self.classifier = classifier
        # reg and cls of the second stage
        self.reg = nn.Linear(in_features=512, out_features=n_classes*4)
        self.cls = nn.Linear(in_features=512, out_features=n_classes)
        # roi pooling
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def forward(self, x, rois, roi_inds, img_h, img_w):
        """
        roi pooling and 2nd cls/reg
        Inputs:
            x: the backbone features
            rois: rois on the image [N*n_sample, 4]
            roi_inds: the batch indexs [N*n_sample]
            image_hs, img_ws: the size of the image
        Outputs:
            cls: [N, n_classes, n_sample]
            reg: [N, n_sample, n_classes*4]
        """
        N = x.shape[0]

        # transfer the boxes from image size to on the features
        feature_rois = torch.zeros(rois.shape)
        feature_rois[:, [0, 2]] = rois[:, [0, 2]] / img_h * x.shape[2]
        feature_rois[:, [1, 3]] = rois[:, [1, 3]] / img_w * x.shape[3]

        # roi pooling, the first column should contain the batch index.
        boxes = torch.cat([roi_inds[:, None], feature_rois], dim=-1)
        cropped_features = roi_pool(x, boxes=boxes, output_size=(self.roi_size, self.roi_size), spatial_scale=self.spatial_scale) # [N*post_thre, C, roi_size, roi_size]

        # cls and reg
        fc6 = self.classifier(cropped_features) # [N*128, 512, 1, 1]
        fc6 = fc6.view(fc6.shape[0], -1) # [N*128, 512]
        reg = self.reg(fc6) # [N*128, 21*4]
        cls = self.cls(fc6) # [N*128, 21]
        reg = reg.view(N, -1, reg.shape[-1])
        cls = cls.view(N, -1, cls.shape[-1])
        cls = cls.permute(0,2,1)

        return cls, reg

