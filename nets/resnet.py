import torch
import torch.nn as nn
import torch.optim as opt

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Compose


class Basicblock(nn.Module):
    """
    kernel size 3, 3 for two conv in residual path.
    kernel size 1 in main path
    """
    expansion = 1
    def __init__(self, in_channels, channels, identical=True, stride=1):
        super(Basicblock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.sideway = None
        if not identical:
            self.sideway = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=stride, bias=False), 
            nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        shortcut = x

        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)

        res = self.conv2(res)
        res = self.bn2(res)

        if self.sideway is not None:
            shortcut = self.sideway(shortcut)
        # make a new name instead of shortcut += res. or backward will meet question
        # result = shortcut+res
        # result = self.relu(result)
        # return result
        res += shortcut
        res = self.relu(res)
        return res


class Bottleneck(nn.Module):
    """
    kernel size 1, 3, 1 for two conv in residual path.
    kernel size 1 in main path
    """
    expansion = 4
    def __init__(self, in_channels, channels, identical=True, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels*4)

        self.sideway = None
        if not identical:
            self.sideway = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels*4, kernel_size=1, stride=stride, bias=False), 
            nn.BatchNorm2d(channels*4),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)

        res = self.conv2(res)
        res = self.bn2(res)
        res = self.relu(res)

        res = self.conv3(res)
        res = self.bn3(res)

        if self.sideway is not None:
            shortcut = self.sideway(shortcut)
        # make a new name instead of shortcut += res. or backward will meet question
        result = shortcut+res
        result = self.relu(result)
        return result


class ResNet(nn.Module):
    def __init__(self, block, n_blocks, n_class=1000):
        super(ResNet, self).__init__()

        # number of in channels for each bottle. iterative
        self.bottle_in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.bottle_in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.conv2 = self._make_layer(block, 64, n_blocks[0])
        self.conv3 = self._make_layer(block, 128, n_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, n_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, n_blocks[3], stride=2)
        self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.bottle_in_channels*block.expansion, n_class)

    def _make_layer(self, block, in_channels, n_blocks, stride=1):
        # output is 4 times input, but input is half size the output of the last blocks

        identical = False
        # the first block of resnet18 don't have downsample
        if in_channels == 64:
            identical=True
        layer1 = block(self.bottle_in_channels, in_channels, identical=identical, stride=stride)
        self.bottle_in_channels = in_channels * block.expansion

        layers = [layer1, ]
        for i in range(n_blocks-1):
            layers.append(block(in_channels*block.expansion, in_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ave_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(pth_path='data/resnet/resnet18-200.pth'):
    model = ResNet(block=Basicblock, n_blocks=[2,2,2,2], n_class=10)
    state_dict = torch.load(pth_path)
    model.load_state_dict(state_dict)
    # backbone
    features = list([model.conv1, model.conv2, model.conv3, model.conv4, ])
    classifier = list([model.conv5, model.ave_pool])
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    
    return features, classifier


if __name__ == "__main__":
    """train a model with CIFAR10"""

    train_transform = Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    test_transform = Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # # train process
    # cifar10_data = CIFAR10(root='../data/samples/cifar10', train=True, download=True, transform=train_transform)
    # data_loader = torch.utils.data.DataLoader(cifar10_data,
    #                                       batch_size=128,
    #                                       shuffle=True)
    # model = ResNet(block=Basicblock, n_blocks=[2,2,2,2], n_class=10)
    # model.train()
    # model.cuda()
    # print(model)

    # # for debug
    # # torch.autograd.set_detect_anomaly(True)
    # n_epoch = 200
    # criterion = nn.CrossEntropyLoss()
    # optimizer = opt.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch)
    
    # for epoch in range(n_epoch):
    #     print("epoch: {}".format(epoch))
    #     epoch_loss = 0.
    #     for i, data in enumerate(data_loader):
    #         input, label = data
    #         input = input.cuda()
    #         label = label.cuda()

    #         optimizer.zero_grad()
    #         output = model(input)
    #         loss = criterion(output, label)
    #         loss.backward()

    #         optimizer.step()
            
    #         epoch_loss += loss.item()
    #         print("iter:{}/{}, loss:{}".format(i, len(data_loader), loss.item()))
    #     print("epoch {}/{} loss: {}".format(epoch+1, n_epoch, epoch_loss/len(data_loader)))
    #     scheduler.step()
    #     if (epoch+1) % 20 == 0:
    #         torch.save(model.state_dict(), '../data/resnet/resnet18-{}.pth'.format(epoch+1))

    # # eval process
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import seaborn as sb

    # testset = CIFAR10(root='../data/samples/cifar10', train=False,download=True, transform=test_transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                      shuffle=False, num_workers=2)
    # model = ResNet(block=Basicblock, n_blocks=[2,2,2,2], n_class=10)
    # state_dict = torch.load('../data/resnet/resnet18-200.pth')
    # model.load_state_dict(state_dict)
    # model.cuda()
    # model.eval()

    # # make the lists of names and scores
    # acc_names = np.append(classes, 'total')
    # acc_scores = np.zeros(len(acc_names))
    # acc_nums = np.ones(len(acc_names))

    # for i, data in enumerate(testloader):
    #     input, label = data
    #     input = input.cuda()
    #     cpu_label = int(label.detach().numpy())
    #     label = label.cuda()

    #     output = model(input)
    #     _, predicted = torch.max(output, 1)
        
    #     acc_nums[-1] += 1
    #     acc_nums[cpu_label] += 1

    #     if output.argmax() == label[0]:
    #         acc_scores[-1] += 1
    #         acc_scores[cpu_label] += 1

    # # get the acc for all classes and total acc
    # acc_scores /= acc_nums
    # acc_scores = np.around(acc_scores, 2)
    # # plot the results
    # plt.figure()
    # df = pd.DataFrame(data=np.vstack([acc_names, acc_scores]).T, columns=['Classes', 'Accuracy'])
    # df['Accuracy'] = df['Accuracy'].astype(float)
    # sb_plot = sb.barplot(x='Classes', y='Accuracy', data=df)
    # plt.savefig('../data/resnet/acc.png')

    # print("Total acc: ", acc_scores[-1])



    # # test backbone
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sb

    testset = CIFAR10(root='../data/samples/cifar10', train=False,download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                        shuffle=False, num_workers=2)
    backbone, classifier = resnet18('../data/resnet/resnet18-200.pth')
    backbone.cuda()
    backbone.eval()
    classifier.cuda()
    classifier.eval()

    import pdb;pdb.set_trace()

    for i,data in enumerate(testloader):
        input, label = data
        input = input.cuda()
        label = label.cuda()

        output = backbone(input)
        results = classifier(output)
        print(results)