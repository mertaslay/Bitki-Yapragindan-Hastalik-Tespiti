import torch
import torch.nn as nn
from torchsummary import summary

# https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downSample):
        super(BottleneckBlock, self).__init__()

        self.downSample = downSample
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels= out_channels//4, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.conv2 = nn.Conv2d(in_channels=out_channels//4, out_channels=out_channels//4, kernel_size=3, stride=2 if downSample else 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(in_channels=out_channels//4, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if self.downSample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2 if self.downSample else 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, input):
        shortcut = self.shortcut(input)

        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))

        input = input + shortcut

        return nn.ReLU()(input)

class ResNet(nn.Module):
    def __init__(self, in_channels, ResBlock, repeat, useBottleneck = False, outputs = 1000):
        super(ResNet, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', ResBlock(filters[0], filters[1], downSample=False))
        for i in range(1, repeat[0]):
            self.layer1.add_module(f'conv2_{i + 1}', ResBlock(filters[1], filters[1], downSample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', ResBlock(filters[1], filters[2], downSample=True))
        for i in range(1, repeat[1]):
            self.layer2.add_module(f'conv3_{i + 1}', ResBlock(filters[2], filters[2], downSample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', ResBlock(filters[2], filters[3], downSample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module(f'conv4_{i + 1}', ResBlock(filters[3], filters[3], downSample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', ResBlock(filters[3], filters[4], downSample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module(f'conv5_{i + 1}', ResBlock(filters[4], filters[4], downSample=False))

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(filters[4], outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)

        input = self.avg(input)
        input = torch.flatten(input, start_dim = 1)
        input = self.fc(input)

        return input

def ResNetModel(resNetModel:int, numberOfClasses:int):
    assert resNetModel in [50 , 101, 152], "ResNet18 and ResNet34 are not supported or you might enter wrong number"

    if resNetModel == 50:
        return ResNet(3, BottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs = numberOfClasses)
    elif resNetModel == 101:
        return ResNet(3, BottleneckBlock, [3, 4, 23, 3], useBottleneck=True, outputs = numberOfClasses)
    elif resNetModel == 152:
        return ResNet(3, BottleneckBlock, [3, 8, 36, 3], useBottleneck=True, outputs = numberOfClasses)




if __name__ == '__main__':
    resNet50 = ResNetModel(50, numberOfClasses=20)
    resNet50 = resNet50.cuda()
    summary(resNet50, (3, 224, 224), depth=3)
