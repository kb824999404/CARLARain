import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# From https://github.com/hezhangsprinter/DID-MDN/blob/master/models/derain_dense.py
class vgg19ca(nn.Module):
    def __init__(self):
        super(vgg19ca, self).__init__()

        haze_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(haze_class.features[0])

        for i in range(1,3):
            self.feature.add_module(str(i),haze_class.features[i])

        self.conv16=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier=nn.Linear(127896, 512)
        self.dense_classifier1=nn.Linear(512, 4)


    def forward(self, x):
        feature=self.feature(x)
        feature=self.conv16(feature)

        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)

        return out