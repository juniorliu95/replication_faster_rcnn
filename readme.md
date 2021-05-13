# replication of Faster RCNN
This is a replication excercise of [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497).

I took the work of [bubbliiiing](https://github.com/bubbliiiing/faster-rcnn-pytorch) as a reference to consider the whole structure of the project, but the implementations are different in many ways. 

## Progress
Writing the test file.

## Usage
1. put the the  voc dataset under `data/voc folder`.
2. put [the resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth) file under the `data/resnet folder`.
3. run `python train.py`

## backbone:
I wrote and trained a resnet18 by myself on CIFAR10 dataset. I got an classification accuracy of about 0.93. But I used the pytorch official pre-trained resnet18 as the backbone.

## reminders
### Backbone

1. optimizer.step() in each iteration, but scheduler.step() each epoch.
2. be careful about the naming of the layers in the *forward* method of the model. In-place naming (x=y;y=x) may push errors. 
3. The image loaders load images as: [N,C,H,W]

### anchors

1. First make the K anchors for each place (relative position to the points)
2. Then generate all the center points of the anchors
3. np.meshgrid(x1, x2), x1 increase in the horizontal direction(y). x2 in the vertical direction(x)
4. put the K anchors on the center points

### RPN

1. Need function to select rois from region proposal
2. Contiguous tensors: for efficent visiting before functions like view()
3. torch.clamp() to crop the bounding boxes

### Head

1. roi pooling need to adjust the boxes' scales from image to features
2. the classifier is trained separately from the backbone and the rpn.
3. add the true boxes to the rois for roi pooling, for better training of the heads

### data_loader

1. need to make a custom data loader
2. the label of the boxes in the xml file are based at the up-left
3. When making true boxes and labels, the length of the boxes must be the same for all images in a batch. I made 32 true boxes for each image, empty boxes are labeled -1

### loss function

1. need to map the true labels and boxes to the rpn and heads.
2. reg uses smooth l1 loss. less sensitive to large biases
3. Use torch.gather for selection of the reg boxes because the reg head output is of shape [N,n_sample, n_class*4].