import torch.nn as nn
from torch.autograd import Variable
from models.prenet import PReNet

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)

class Derain(nn.Module):
    def __init__(self, recurrent_iter=6, channel=32):
        super(Derain, self).__init__()
        self.derainet = PReNet(recurrent_iter, channel)
    def forward(self, input):
        mu_b, logvar_b = self.derainet(input)
        return mu_b, logvar_b

class Encoder_64(nn.Module):  # RNet
    def __init__(self, nc, nef, nz):
        super(Encoder_64, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*8, nef*16, 4, 1),
            nn.ReLU(True),
            View((-1, nef*16 * 1 * 1)),
            nn.Linear(nef*16, nz* 2),
        )
    def forward(self, input):
        distributions = self.main(input)
        return distributions

class Decoder_64(nn.Module):
    def __init__(self, nz, nef, nc):
        super(Decoder_64, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, nef*16),
            View((-1, nef*16, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef*16, nef * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),
            nn.ReLU(True)
        )
    def forward(self, input):
        R = self.main(input)
        return R


# VRGNet 64
class EDNet_64(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef):
        super(EDNet_64,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.encoder = Encoder_64(self.nc,self.nef,self.nz)
        self.decoder = Decoder_64(self.nz,self.nef,self.nc)
    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        R = self.decoder(z)
        return  R, mu,logvar, z
    

class Encoder_128(nn.Module):  # RNet
    def __init__(self, nc, nef, nz):
        super(Encoder_128, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),     # 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),   # 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),   #  16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),   #  8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*8, nef*16, 4, 2, 1, bias=False),       #  4 x 4
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*16, nef*32, 4, 1),       #  1 x 1
            nn.ReLU(True),
            View((-1, nef*32 * 1 * 1)),
            nn.Linear(nef*32, nz* 2),
        )
    def forward(self, input):
        distributions = self.main(input)
        return distributions

class Decoder_128(nn.Module):
    def __init__(self, nz, nef, nc):
        super(Decoder_128, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, nef*32),
            View((-1, nef*32, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef*32, nef * 16, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef*16, nef * 8, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),
            nn.ReLU(True)
        )
    def forward(self, input):
        R = self.main(input)
        return R

# VRGNet 128
class EDNet_128(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef):
        super(EDNet_128,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.encoder = Encoder_128(self.nc,self.nef,self.nz)
        self.decoder = Decoder_128(self.nz,self.nef,self.nc)
    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        R = self.decoder(z)
        return  R, mu,logvar, z
    


class Encoder_256(nn.Module):  # RNet
    def __init__(self, nc, nef, nz):
        super(Encoder_256, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),  # 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),  # 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),  # 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),  #  16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 8, nef * 16, 4, 2, 1, bias=False),  #  8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False),  #  4 x 4
            # nn.BatchNorm2d(nef * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 32, nef * 64, 4, 1),  #  1 x 1
            nn.ReLU(True),
            View((-1, nef * 64 * 1 * 1)),  #
            nn.Linear(nef * 64, nz * 2),  #
        )
    def forward(self, input):
        distributions = self.main(input)
        return distributions

class Decoder_256(nn.Module):
    def __init__(self, nz, nef, nc):
        super(Decoder_256, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, nef * 64),  #
            View((-1, nef * 64, 1, 1)),  # 1*1
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 64, nef * 32, 4, 1, 0, bias=False),  #  4 x 4
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 32, nef * 16, 4, 2, 1, bias=False),  #  8 x 8
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 16, nef * 8, 4, 2, 1, bias=False),  # 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),  # 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),  # 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),  #  128 x 128
            nn.ReLU(True),
            nn.ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),  # 256 x 256
            nn.ReLU(True)
        )
    def forward(self, input):
        R = self.main(input)
        return R

class Encoder_512(nn.Module):  # RNet
    def __init__(self, nc, nef, nz):
        super(Encoder_512, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),  # 256 x 256
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),  # 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),  # 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),  #  32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 8, nef * 16, 4, 2, 1, bias=False),  #  16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=False),  #  8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 32, nef * 64, 4, 2, 1, bias=False),  #  4 x 4
            # nn.BatchNorm2d(nef * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 64, nef * 128, 4, 1),  #  1 x 1
            nn.ReLU(True),
            View((-1, nef * 128 * 1 * 1)),  #
            nn.Linear(nef * 128, nz * 2),  #
        )
    def forward(self, input):
        distributions = self.main(input)
        return distributions

# VRGNet 256
class EDNet_256(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef):
        super(EDNet_256,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.encoder = Encoder_256(self.nc,self.nef,self.nz)
        self.decoder = Decoder_256(self.nz,self.nef,self.nc)
    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        R = self.decoder(z)
        return  R, mu,logvar, z
    
