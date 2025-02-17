import torch
import torch.nn as nn
import torchvision

class DenseNet121(nn.Module):
    
    def __init__(self, n_output=2):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet.densenet121()
        self.model.classifier = nn.Linear(1024, n_output)

    def forward(self, x):
        x = self.model(x)
        return [x]
