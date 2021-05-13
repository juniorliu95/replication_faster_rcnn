import os

import torch
from skimage import io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import xmltodict

try:
    from utils.utils import PASCAL_VOC_CLASSES, PASCAL_VOC_NUM_CLASSES
except:
    from utils import PASCAL_VOC_CLASSES, PASCAL_VOC_NUM_CLASSES

class voc_data(Dataset):
    """
    Make the data loader for voc dataset
    """
    def __init__(self, root_dir, mode, difficult=False, new_size=(600, 600)):
        """
        Inputs:
            root_dir: the dir of the dataset
            mode: train, val
            difficult: whether use the difficult samples
            new_size: the new size of the images
        """
        super(voc_data, self).__init__()

        self.PASCAL_VOC_CLASSES = PASCAL_VOC_CLASSES
        self.PASCAL_VOC_NUM_CLASSES = PASCAL_VOC_NUM_CLASSES

        self.class2num = dict(zip(self.PASCAL_VOC_CLASSES, list(range(self.PASCAL_VOC_NUM_CLASSES))))
        self.num2class = dict(zip(list(range(self.PASCAL_VOC_NUM_CLASSES)), self.PASCAL_VOC_CLASSES))
        

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.root_dir = root_dir

        assert mode in ['train', 'val'], 'mode should be train or val'
        self.mode = mode
        self.difficult = difficult
        self.new_size = new_size

        self.img_list = []
        with open(os.path.join(root_dir, 'ImageSets/Main/aeroplane_{}.txt'.format(mode)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.img_list.append(line.split()[0])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, 'JPEGImages', self.img_list[idx]+'.jpg')
        image = io.imread(img_name) # [H,W,C]
        label_name = os.path.join(self.root_dir, 'Annotations', self.img_list[idx]+'.xml')
        label, box = self._get_labels(label_name)

        new_h, new_w = self.new_size
        try:
            box[:, [0, 2]] = box[:, [0, 2]] / image.shape[0] * new_h
            box[:, [1, 3]] = box[:, [1, 3]] / image.shape[1] * new_w
            box[box<0] = -1
        except:
            pass
        image = resize(image, (new_h, new_w))

        if self.transform:
            image = self.transform(image) # [C,H,W]

        sample = {'image': image, 'box': box, 'label': label}

        return sample

    def _get_labels(self, label_name, n_obj=32):
        """
        get the labels from the xml file
        Inputs:
            label_name: the name of the label file
            n_obj: max # of objects in a images
        """
        labels = -1 * np.ones(n_obj) # at most 32 objects
        boxes = -1 * np.ones([n_obj, 4])

        with open(label_name) as fd:
            doc = xmltodict.parse(fd.read())
            print(doc['annotation']['filename'])
            objects = doc['annotation']['object']
            for obj_ind, object in enumerate(objects):
                # break if more than n_obj objects
                if obj_ind >=n_obj:
                    break

                try:
                    labels[obj_ind] = self.class2num[object['name']]
                    # add box labels
                    bndbox = object['bndbox']
                    # the width of the box is defined as x, but we use height as x
                    box = np.array([float(bndbox['ymin']), float(bndbox['xmin']), float(bndbox['ymax']), float(bndbox['xmax'])])
                    boxes[obj_ind, :] = box

                    if not self.difficult and object['difficult']=='1':
                        labels[obj_ind] = -1

                except: # if no labels
                    labels[obj_ind] = -1
                    boxes[obj_ind, :] = [-1,-1,-1,-1]
        labels = np.array(labels)
        boxes = np.around(boxes)

        return labels, boxes

if __name__ == '__main__':
    vocdata = voc_data('./data/voc/VOCdevkit/VOC2012', 'train')
    data_loader = DataLoader(vocdata, batch_size=2, shuffle=False)
    for data in data_loader:
        image, boxes, labels = data['image'], data['box'], data['label']
        image = image.numpy().squeeze().transpose(1,2,0)
        box = boxes[0, :].numpy()

        if labels[0] != 0: # not background
            plt.axhline(box[0])
            plt.axvline(box[1])
            plt.axhline(box[2])
            plt.axvline(box[3])
            plt.imshow(image)

            plt.show()

