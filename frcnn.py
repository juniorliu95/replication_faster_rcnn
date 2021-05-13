import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nets.resnet_torch import resnet_backbone
from nets.rpn import RPN
from nets.heads import ResnetHead
from nets.faster_rcnn import FasterRCNN
from utils.utils import *
from utils.data_loader import voc_data

class FRCNN(nn.Module):
    def __init__(self, mode):
        super(FRCNN, self).__init__()
        self.mode = mode

    def get_data_loader(self, root_dir='data/voc/VOCdevkit/VOC2012', batch_size=2, shuffule=True):
        vocdata= voc_data(root_dir, self.mode)
        self.data_loader = DataLoader(vocdata, batch_size, shuffule)
        
        return self.data_loader

    def get_network(self):
        self.net = FasterRCNN(resnet_backbone, RPN, ResnetHead, self.mode)
        return self.net

    def load_param(self, path):
        param = torch.load(path)
        self.net.load_state_dict(param)

    def save_param(self, iter, path):
        self.net.save(os.path.join(path, str(iter+1)+'.pth'))
        print('iteration {} saved...'.format(iter+1))


    

if __name__ == "__main__":
    pass


