import torch
import torch.nn as nn

ratios = [0.5, 1., 2.] # h/m
anchor_scales = [8, 16, 32] # 128, 256, 512

class FasterRCNN(nn.Module):
    """
    combining all the parts of the FRCNN
    In the whole project, we define height is x and width is y
    """
    def __init__(self, backbone, rpn, head, mode):
        super(FasterRCNN, self).__init__()

        # network parts
        self.backbone, self.classifier = backbone()
        self.rpn = rpn(ratios=ratios, anchor_scales=anchor_scales, mode=mode)
        self.head = head(self.classifier)

        # other parameters
        self.mode = mode


    def forward(self, x, width, height):
        """
        Inputs:
            x: the input features
            width, height: the size of the image
        """
        features = self.backbone(x)
        cls, reg, rois, roi_inds, anchors = self.rpn(features, width, height)
        cls_output, reg_output = self.head(features, rois, roi_inds)

        return cls, reg, rois, roi_inds, cls_output, reg_output, anchors



