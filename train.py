import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision.ops as ops

from frcnn import FRCNN
from utils.utils import AnchorTargetCreator, ProposalTargetCreator, n_train_post_nms, PASCAL_VOC_CLASSES, PASCAL_VOC_NUM_CLASSES


class trainer(nn.Module):
    def __init__(self):
        super(trainer, self).__init__()
        self.total_loss = 0.
        self.rpn_reg_loss = 0.
        self.rpn_cls_loss = 0.
        self.reg_loss = 0.
        self.cls_loss = 0.
        self.model=FRCNN('train')
        self.model.get_data_loader(shuffule=False)
        self.model.get_network()
        self.n_sample = [256, 128] # number of samples for two stage targets
        self.at = AnchorTargetCreator(self.n_sample[0]) # generate labels for rpn
        self.pt = ProposalTargetCreator(self.n_sample[1]) # generate labels for classifier
        self.post_thre = n_train_post_nms # number of rois kept for each image

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma=1):
        """
        loss function of the regressors of rpn and heads
        Inputs:
            pred_loc: [n_sample, 4]
            gt_loc: [n_sample, 4]
            gt_label: [n_sample]
        Outputs:
            loss
        """
        # only train the positive samples
        pred_loc = pred_loc[gt_label>0]
        gt_loc = gt_loc[gt_label>0]

        def _smooth_l1_loss(x, t, sigma):
            sigma_squared = sigma ** 2
            regression_diff = (x - t)
            regression_diff = regression_diff.abs()
            regression_loss = torch.where(
                    regression_diff < (1. / sigma_squared),
                    0.5 * sigma_squared * regression_diff ** 2,
                    regression_diff - 0.5 / sigma_squared
                )
            return regression_loss.sum()

        loc_loss = _smooth_l1_loss(pred_loc, gt_loc, sigma)
        num_pos = (gt_label > 0).sum().float()
        loc_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return loc_loss

    def train_step(self, image, boxes, labels):
        self.optimizer.zero_grad()

        # extract the backbone features
        features = self.model.net.backbone(image.float())
        # cls: [N,2, KHW], reg: [N,KHW, 4], rois: [N*post_thre, 4], roi_ind: [N*post_thre], anchors: [KHW, 4]
        cls, reg, rois, roi_inds, anchors = self.model.net.rpn(features, image.shape[3], image.shape[2])
        
        # labels for rpn
        reg_targets_rpn = torch.zeros([len(image), anchors.shape[0], anchors.shape[1]]) # [N, KHW, 4]
        cls_labels_rpn = torch.zeros([len(image), anchors.shape[0]]) #[N,KHW]
        
        for i in range(len(image)):
            # for each image
            # make the rpn ratgets
            label = labels[i] # [32]
            true_ind = label!=-1
            box = boxes[i, true_ind, :] # [32, 4]
            reg_target_rpn, label_rpn = self.at(box, anchors)
            reg_targets_rpn[i] = torch.from_numpy(reg_target_rpn)
            cls_labels_rpn[i] = torch.from_numpy(label_rpn)

        rpn_reg_loss = self._fast_rcnn_loc_loss(reg, reg_targets_rpn, cls_labels_rpn)

        rpn_cls_loss = F.cross_entropy(cls, cls_labels_rpn.type(torch.LongTensor), ignore_index=-1)

        # labels for classifier
        sample_rois = torch.zeros([image.shape[0], self.n_sample[1], 4]) # [N,128,4]
        sample_rois_ind = torch.zeros([image.shape[0], self.n_sample[1]]) # [N,128]
        
        reg_targets_classifier = torch.zeros([len(image), self.n_sample[1], 4]) # [N, n_sample, 4]
        cls_labels_classifier = torch.zeros([len(image), self.n_sample[1]]) #[N, n_sample]
        for i in range(len(image)):
            sample_rois_ind[i,:] = i
            # make the classifier tartgets
            roi = rois.detach()[roi_inds==i, :] # get the rois for one image. [600, 4]
            label = labels[i] # [32]
            true_ind = label!=-1
            box = boxes[i, true_ind, :] # [32, 4]
            
            # sample_roi:[n_sample, 4], reg_target_classifier:[n_sample, 4], cls_label_classifier:[n_sample,]
            sample_roi, reg_target_classifier, cls_label_classifier = self.pt(roi, box, label)
            sample_rois[i,:,:] = torch.from_numpy(sample_roi)
            reg_targets_classifier[i,:,:] = torch.from_numpy(reg_target_classifier)
            cls_labels_classifier[i,:] = torch.from_numpy(cls_label_classifier)

        # flatten the rois and the inds
        sample_rois = sample_rois.contiguous().view(-1, 4)
        sample_rois_ind = torch.flatten(sample_rois_ind)
        #cls_outputs: [N, 21, n_sample], reg_outputs: [N, n_sample, 21*4],
        cls_output, reg_output = self.model.net.head(features, sample_rois, sample_rois_ind, image.shape[2], image.shape[3])

        # gather selected boxes
        # first make the indexes from cls labels
        reg_ind = cls_labels_classifier.detach().unsqueeze(-1).type(torch.LongTensor)*4 # [N,n_sample,1]
        reg_ind = torch.cat([reg_ind, reg_ind+1, reg_ind+2, reg_ind+3], dim=-1)
        # select the boxes. [N, n_sample, 4]
        reg_output = torch.gather(reg_output, dim=-1, index=reg_ind)

        # losses
        reg_loss = self._fast_rcnn_loc_loss(reg_output, reg_targets_classifier, cls_labels_classifier)
        cls_loss = F.cross_entropy(cls_output, cls_labels_classifier.type(torch.LongTensor), ignore_index=-1)

        total_loss = rpn_cls_loss+rpn_reg_loss+cls_loss+reg_loss
        print("Total loss:{}, \nrpn cls loss:{}, rpn reg loss:{}, \ncls loss:{},reg loss:{}".format(total_loss,rpn_cls_loss,rpn_reg_loss,cls_loss, reg_loss))

        total_loss.backward()
        self.optimizer.step()

    def train(self, lr, n_epoch, save_folder='data/voc/model', load_path=None):
        """The code for training a faster rcnn"""

        if load_path:
            self.model.load_param(load_path)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        self.model.net.train()
        self.optimizer = opt.Adam(self.model.net.parameters(), lr=lr, weight_decay=5e-6)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, n_epoch)

        for iter in range(n_epoch):
            print('Training epoch:{}'.format(iter))
            for i, data in enumerate(self.model.data_loader):
                image, box, label = data['image'], data['box'].numpy(), data['label'].numpy()
                self.train_step(image, box, label)

            self.scheduler.step()
            if iter % 10 == 0:
                torch.save(self.model.net.state_dict(), os.path.join(save_folder, '{}.pth'.format(iter)))


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    lr = 0.01
    n_epoch = 50
    batch_size = 2
    trainer_frcnn = trainer()
    trainer_frcnn.train(lr=lr, n_epoch=n_epoch)