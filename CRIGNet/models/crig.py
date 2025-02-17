import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from models.vrgnet import reparametrize,Encoder_64, Decoder_64,Encoder_128,Decoder_128,Encoder_256,Encoder_512
from models.vqgan import ResnetDecoder_64,ResnetDecoder_128,TransformerDecoder
from models.resnet import resnet50, resnet101
from models.stylegan2 import MappingNetwork
from models.fastgan import Generator as FastGAN

# CRIGNet 64: With Rendered Rain Image Labels
class CRIG_EDNet_64(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0):
        super(CRIG_EDNet_64,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_64(self.nc,self.nef,self.nz)
        self.decoder = Decoder_64(self.nz+self.nlabel,self.nef,self.nc)
    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        z_label = torch.cat([z,label],dim=1)
        R = self.decoder(z_label)
        return  R, mu,logvar, z

# CRIGNet 128: With Rendered Rain Image Labels
class CRIG_EDNet_128(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0,w_dim=0,use_mapping=False):
        super(CRIG_EDNet_128,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_128(self.nc,self.nef,self.nz)
        self.use_mapping = use_mapping
        if use_mapping:
            self.decoder = Decoder_128(self.nz+w_dim,self.nef,self.nc)
            self.mapping = MappingNetwork(z_dim=nlabel,c_dim=0,w_dim=w_dim,num_ws=None)
        else:
            self.decoder = Decoder_128(self.nz+self.nlabel,self.nef,self.nc)

    def sample(self, input):
        return  self.decoder(input)

    def sample_mapping(self, z, label):
        label_mapping = self.mapping(label)
        z_label = torch.cat([z,label_mapping],dim=1)
        return self.decoder(z_label)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        if self.nlabel > 0:
            if self.use_mapping:
                label_mapping = self.mapping(label)
                z_label = torch.cat([z,label_mapping],dim=1)
            else:
                z_label = torch.cat([z,label],dim=1)
        else:
            z_label = z
        R = self.decoder(z_label)
        return  R, mu,logvar, z

# CRIGNet 64: With Rendered Rain Image Labels With Resnet Backbone
class CRIG_EDNet_Resnet_64(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0):
        super(CRIG_EDNet_Resnet_64,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_64(self.nc,self.nef,self.nz)
        self.decoder = ResnetDecoder_64(self.nz+self.nlabel,self.nef,self.nc)
    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        z_label = torch.cat([z,label],dim=1)
        R = self.decoder(z_label)
        return  R, mu,logvar, z
    
# CRIGNet 128: With Rendered Rain Image Labels With Resnet Backbone
class CRIG_EDNet_Resnet_128(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0,w_dim=0,use_mapping=False):
        super(CRIG_EDNet_Resnet_128,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_128(self.nc,self.nef,self.nz)
        self.use_mapping = use_mapping
        if use_mapping:
            self.decoder = ResnetDecoder_128(self.nz+w_dim,self.nef,self.nc,use_relu=True)
            self.mapping = MappingNetwork(z_dim=nlabel,c_dim=0,w_dim=w_dim,num_ws=None)
        else:
            self.decoder = ResnetDecoder_128(self.nz+self.nlabel,self.nef,self.nc)
    def sample (self, input):
        return  self.decoder(input)

    def sample_mapping(self, z, label):
        label_mapping = self.mapping(label)
        z_label = torch.cat([z,label_mapping],dim=1)
        return self.decoder(z_label)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        if self.nlabel > 0:
            if self.use_mapping:
                label_mapping = self.mapping(label)
                z_label = torch.cat([z,label_mapping],dim=1)
            else:
                z_label = torch.cat([z,label],dim=1)
        else:
            z_label = z
        R = self.decoder(z_label)
        return  R, mu,logvar, z

# CRIGNet 64: With Rendered Rain Image Labels With Transformer Backbone
class CRIG_EDNet_Transformer_64(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0):
        super(CRIG_EDNet_Transformer_64,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_64(self.nc,self.nef,self.nz)
        self.decoder = TransformerDecoder(nz=self.nz+self.nlabel,in_channels=self.nef,out_channels=self.nc,
                                        attn_resolutions=[16,32])
    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        z_label = torch.cat([z,label],dim=1)
        R = self.decoder(z_label)
        return  R, mu,logvar, z

# CRIGNet 128: With Rendered Rain Image Labels With Transformer Backbone
class CRIG_EDNet_Transformer_128(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0,w_dim=0,use_mapping=False):
        super(CRIG_EDNet_Transformer_128,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_128(self.nc,self.nef,self.nz)
        self.use_mapping = use_mapping
        if use_mapping:
            self.decoder = TransformerDecoder(nz=self.nz+w_dim,in_channels=self.nef,out_channels=self.nc,
                                            ch_mult=(1,2,2,4,4,4,8),
                                            attn_resolutions=[16,32,64],resolution=128)
            self.mapping = MappingNetwork(z_dim=nlabel,c_dim=0,w_dim=w_dim,num_ws=None)
        else:
            self.decoder = TransformerDecoder(nz=self.nz+self.nlabel,in_channels=self.nef,out_channels=self.nc,
                                            ch_mult=(1,2,2,4,4,4,8),
                                            attn_resolutions=[16,32,64],resolution=128)
    def sample (self, input):
        return  self.decoder(input)

    def sample_mapping(self, z, label):
        label_mapping = self.mapping(label)
        z_label = torch.cat([z,label_mapping],dim=1)
        return self.decoder(z_label)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        if self.use_mapping:
            label_mapping = self.mapping(label)
            z_label = torch.cat([z,label_mapping],dim=1)
        else:
            z_label = torch.cat([z,label],dim=1)
        R = self.decoder(z_label)
        return  R, mu,logvar, z

# CRIGNet 64: With Rendered Rain Image Labels With FastGAN Backbone
class CRIG_FastGAN_64(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0):
        super(CRIG_FastGAN_64,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_64(self.nc,self.nef,self.nz)
        self.decoder = FastGAN(nz=self.nz+self.nlabel,im_size=64)
    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        z_label = torch.cat([z,label],dim=1)
        R = self.decoder(z_label)
        return  R, mu,logvar, z

# CRIGNet 128: With Rendered Rain Image Labels With FastGAN Backbone
class CRIG_FastGAN_128(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0,w_dim=0,use_mapping=False):
        super(CRIG_FastGAN_128,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_128(self.nc,self.nef,self.nz)
        self.use_mapping = use_mapping
        if use_mapping:
            self.decoder = FastGAN(nz=self.nz+w_dim,im_size=128)
            self.mapping = MappingNetwork(z_dim=nlabel,c_dim=0,w_dim=w_dim,num_ws=None)
        else:
            self.decoder = FastGAN(nz=self.nz+self.nlabel,im_size=128)

    def sample(self, input):
        return  self.decoder(input)

    def sample_mapping(self, z, label):
        label_mapping = self.mapping(label)
        z_label = torch.cat([z,label_mapping],dim=1)
        return self.decoder(z_label)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        if self.nlabel > 0:
            if self.use_mapping:
                label_mapping = self.mapping(label)
                z_label = torch.cat([z,label_mapping],dim=1)
            else:
                z_label = torch.cat([z,label],dim=1)
        else:
            z_label = z
        R = self.decoder(z_label)
        return  R, mu,logvar, z
    
# CRIGNet 256: With Rendered Rain Image Labels With FastGAN Backbone
class CRIG_FastGAN_256(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0,w_dim=0,use_mapping=False):
        super(CRIG_FastGAN_256,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_256(self.nc,self.nef,self.nz)
        self.use_mapping = use_mapping
        if use_mapping:
            self.decoder = FastGAN(nz=self.nz+w_dim,im_size=256)
            self.mapping = MappingNetwork(z_dim=nlabel,c_dim=0,w_dim=w_dim,num_ws=None)
        else:
            self.decoder = FastGAN(nz=self.nz+self.nlabel,im_size=256)

    def sample(self, input):
        return  self.decoder(input)

    def sample_mapping(self, z, label):
        label_mapping = self.mapping(label)
        z_label = torch.cat([z,label_mapping],dim=1)
        return self.decoder(z_label)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        if self.nlabel > 0:
            if self.use_mapping:
                label_mapping = self.mapping(label)
                z_label = torch.cat([z,label_mapping],dim=1)
            else:
                z_label = torch.cat([z,label],dim=1)
        else:
            z_label = z
        R = self.decoder(z_label)
        return  R, mu,logvar, z


# CRIGNet 512: With Rendered Rain Image Labels With FastGAN Backbone
class CRIG_FastGAN_512(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef,nlabel=0,w_dim=0,use_mapping=False):
        super(CRIG_FastGAN_512,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.nlabel = nlabel
        self.encoder = Encoder_512(self.nc,self.nef,self.nz)
        self.use_mapping = use_mapping
        if use_mapping:
            self.decoder = FastGAN(nz=self.nz+w_dim,im_size=512)
            self.mapping = MappingNetwork(z_dim=nlabel,c_dim=0,w_dim=w_dim,num_ws=None)
        else:
            self.decoder = FastGAN(nz=self.nz+self.nlabel,im_size=512)

    def sample(self, input):
        return  self.decoder(input)

    def sample_mapping(self, z, label):
        label_mapping = self.mapping(label)
        z_label = torch.cat([z,label_mapping],dim=1)
        return self.decoder(z_label)

    def forward(self, input, label):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        if self.nlabel > 0:
            if self.use_mapping:
                label_mapping = self.mapping(label)
                z_label = torch.cat([z,label_mapping],dim=1)
            else:
                z_label = torch.cat([z,label],dim=1)
        else:
            z_label = z
        R = self.decoder(z_label)
        return  R, mu,logvar, z

class RainClassifier(nn.Module):
    def __init__(self, nc=32, loss_type = 'l2') -> None:
        super().__init__()
        self.model= resnet50(pretrained=True)

        num_features = self.model.fc.in_features
        self.model.fc=nn.Linear(num_features,nc)

        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()

    def forward(self, x1, x2):
        if x1.ndim == 3:
            x1 = x1[None,...]
        if x2.ndim == 3:
            x2 = x2[None,...]
        feature1 = self.model(x1)
        feature2 = self.model(x2)
        loss = self.criterion(feature1,feature2)
        return loss
    
class RainHopenet_256(nn.Module):
    def __init__(self, block=torchvision.models.resnet.Bottleneck, 
                 layers=[4, 6, 8, 4], n_feature=256, n_output=2):
        self.inplanes = 64
        super(RainHopenet_256, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(512* 256**2, n_feature)
        self.fc_output = nn.Linear(n_feature, n_output)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = layer4
        x = x.view(x.size(0), -1)

        feature = self.fc(x)
        output = self.fc_output(feature)
        output = F.sigmoid(output)

        out = [layer1, layer2, layer3, layer4, feature, output]

        return out

class RainHopenet_128(nn.Module):
    def __init__(self, block=torchvision.models.resnet.Bottleneck, 
                 layers=[3, 4, 6, 3], n_feature=128, n_output=2):
        self.inplanes = 64
        super(RainHopenet_128, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(2048 * 4**2, n_feature)
        self.fc_output = nn.Linear(n_feature, n_output)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = layer4
        x = x.view(x.size(0), -1)

        feature = self.fc(x)
        output = self.fc_output(feature)
        output = F.sigmoid(output)

        out = [layer1, layer2, layer3, layer4, feature, output]

        return out
    
class RainHopenet_64(nn.Module):
    def __init__(self, block=torchvision.models.resnet.Bottleneck, 
                 layers=[3, 4, 6, 3], n_feature=128, n_output=2):
        self.inplanes = 64
        super(RainHopenet_64, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(2048 * 2**2, n_feature)
        self.fc_output = nn.Linear(n_feature, n_output)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = layer4
        x = x.view(x.size(0), -1)

        feature = self.fc(x)
        output = self.fc_output(feature)
        output = F.sigmoid(output)

        out = [layer1, layer2, layer3, layer4, feature, output]

        return out