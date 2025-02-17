import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import kornia
from losses.did_mdn import vgg19ca

def l1(x, y, mask = None):
    if mask:
        l1_loss = torch.sum(torch.sum(torch.abs(x-y), dim=1) * mask) / torch.sum(mask)
    else:
        l1_loss = torch.abs(x-y)
    return l1_loss


def l2(x, y, mask = None):
    if mask:
        l2_loss = torch.sum(torch.sqrt(torch.sum(torch.pow((x-y), 2), dim=1) + 1e-8) * mask) / torch.sum(mask)
    else:
        l2_loss = torch.sqrt(torch.sum(torch.pow((x-y), 2), dim=1) + 1e-8)
    return l2_loss

class IdentityLoss(nn.Module):
    def __init__(self, ckpt_path, margin=0.3):
        super().__init__()
        self.margin = margin
        self.rainNet = vgg19ca()
        self.rainNet.load_state_dict(torch.load(ckpt_path))
        self.rainNet = self.rainNet.cuda()
    
    def forward(self,x_render,xrec,mask=None):

        # input to rain recognition network
        if mask is not None:
            id_fake = self.rainNet(xrec * mask[:,None,...])
        else:
            id_fake = self.rainNet(xrec)
        id_render = self.rainNet(x_render)

        id_fake = F.normalize(id_fake,dim=1)
        id_render = F.normalize(id_render,dim=1)

        # cosine similarity
        sim = torch.cosine_similarity(id_fake, id_render, dim=1)
        margin = torch.full_like(sim,self.margin)
        loss = torch.max(margin,1.0 - sim)

        return loss
    

class HOGLayer(nn.Module):
    def __init__(self, nbins=10, pool=8, max_angle=torch.pi, stride=1, padding=1, dilation=1,
                 mean_in=False, max_out=False):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        self.max_out = max_out
        self.mean_in = mean_in
        # Sobel算子，水平和垂直方向
        mat = torch.FloatTensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        if self.mean_in:
            return self.forward_v1(x)
        else:
            return self.forward_v2(x)

    # 灰度图
    def forward_v1(self, x):
        if x.size(1) > 1:
            # 多通道转单通道
            x = x.mean(dim=1)[:,None,:,:]

        # Sobel算子卷积
        # x: [batch, 1, H, W]
        # gxy: [batch, 2, H, W]
        gxy = F.conv2d(x, self.weight, None, self.stride,
                       self.padding, self.dilation, 1)
        # 2. Mag/ Phase
        # mag: [batch, H, W]
        # norm: [batch, 1, H, W]
        # phase: [batch, H, W]
        mag = gxy.norm(dim=1)
        norm = mag[:, None, :, :]
        phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

        # 3. Binning Mag with linear interpolation
        # 梯度角度分段
        # phase_int: [batch, 1, H, W]
        phase_int = phase / self.max_angle * self.nbins
        phase_int = phase_int[:, None, :, :]

        # 获取直方图
        n, c, h, w = gxy.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
        out.scatter_(1, phase_int.floor().long() % self.nbins, norm) # 将梯度大小以角度为索引分散到直方图中，两个方向的加权
        out.scatter_add_(1, phase_int.ceil().long() % self.nbins, 1 - norm)

        return self.pooler(out)

    # 彩色图像
    def forward_v2(self, x):
        batch_size, in_channels, height, width = x.shape
        weight = self.weight.repeat(3, 1, 1, 1)
        gxy = F.conv2d(x, weight, None, self.stride,
                        self.padding, self.dilation, groups=in_channels)

        gxy = gxy.view(batch_size, in_channels, 2, height, width)

        if self.max_out:
            # 取所有通道梯度的最大值
            gxy = gxy.max(dim=1)[0][:,None,:,:,:]

        #2. Mag/ Phase
        mags = gxy.norm(dim=2)
        norms = mags[:,:,None,:,:]
        phases = torch.atan2(gxy[:,:,0,:,:], gxy[:,:,1,:,:])

        #3. Binning Mag with linear interpolation
        phases_int = phases / self.max_angle * self.nbins
        phases_int = phases_int[:,:,None,:,:]

        out = torch.zeros((batch_size, in_channels, self.nbins, height, width),
                          dtype=torch.float, device=gxy.device)
        out.scatter_(2, phases_int.floor().long()%self.nbins, norms)
        out.scatter_add_(2, phases_int.ceil().long()%self.nbins, 1 - norms)

        out = out.view(batch_size, in_channels * self.nbins, height, width)
        out = torch.cat((self.pooler(out), self.pooler(x)), dim=1) # 每个通道的直方图和原始图像
        return out

class HOGLoss(nn.Module):
    def __init__(self, nbins=10, pool=8, mean_in=False):
        super().__init__()
        self.hog = HOGLayer(nbins=nbins, pool=pool, mean_in=mean_in)
        self.hog = self.hog.cuda()
    
    def forward(self,x_render,x_rec,mask=None):
        if mask is not None:
            hog_rec = self.hog(x_rec * mask[:,None,...])
        else:
            hog_rec = self.hog(x_rec)
        hog_render = self.hog(x_render)

        hog_render = F.normalize(hog_render, dim=1)
        hog_rec = F.normalize(hog_rec, dim=1)

        loss = F.kl_div(hog_render.softmax(-1).log(),hog_rec.softmax(-1),reduction='none').sum(dim=1)
        loss = loss.mean(1).mean(1)
        return loss


class GLCMLayer(nn.Module):
    def __init__(self,  vmin=0, vmax=255, levels=8, kernel_size=5, distanceCount=1, angleCount=8, pool=32, mean_in=False, mean_out=False):
        '''
        Parameters
        ----------
        vmin: int
            minimum value of input image
        vmax: int
            maximum value of input image
        levels: int
            number of grey-levels of GLCM
        kernel_size: int
            Patch size to calculate GLCM around the target pixel
        distance: int
            pixel pair distance offsets count [pixel] (1.0, 2.0, and etc.)
        angle: float
            pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)
        pool: int
            avg pool size
        '''
        super(GLCMLayer, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.levels = levels
        self.kernel_size = kernel_size
        self.distances = np.linspace(1,distanceCount,distanceCount)
        self.angles = np.linspace(0,360,angleCount+1)[:-1]


        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)
        self.mean_in = mean_in
        self.mean_out = mean_out


    def get_glcm(self,x,distance,angle):
        # x: [B, C, H, W]
        if self.mean_in and x.size(1) > 1:
            # 多通道转单通道
            x = x.mean(dim=1)[:,None,:,:]

        mi, ma = self.vmin, self.vmax
        ks = self.kernel_size
        levels = self.levels
        b,c,h,w = x.shape

        # digitize
        bins = torch.linspace(mi, ma+1, levels+1).to(x.device)
        gl1 = torch.bucketize(x,bins) - 1

        # make shifted image
        dx = distance*np.cos(np.deg2rad(angle))
        dy = distance*np.sin(np.deg2rad(-angle))

        mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
        mat = torch.from_numpy(mat).to(x.device)
        mat = mat.unsqueeze(0).repeat(b,1,1)

        gl2 = kornia.geometry.transform.warp_affine(gl1.type(torch.float32),
                                            mat,(h,w),
                                            mode='nearest',padding_mode='border')
        gl2 = gl2.type(torch.uint8)

        glcm = torch.zeros((levels, levels, b, c, h, w), dtype=torch.uint8).to(x.device)

        # make glcm
        for i in range(levels):
            for j in range(levels):
                mask = ((gl1==i) & (gl2==j))
                glcm[i,j, mask] = 1

        kernel = torch.ones((ks, ks), dtype=torch.uint8)
        for i in range(levels):
            for j in range(levels):
                glcm[i,j] = kornia.filters.filter2d(glcm[i,j].type(torch.float32), kernel[None,...])

        glcm = glcm.type(torch.float32)
        glcm = glcm.view(levels*levels,b,c,h,w)
        glcm = glcm.permute(1,0,2,3,4)

        if self.mean_out:
            glcm = glcm.mean((-2,-1))

        return glcm
    
    def forward(self,x):
        b,c,h,w = x.shape
        glcms = []
        for distance in self.distances:
            glcm_dist = []
            for angle in self.angles:
                glcm_dist.append(self.get_glcm(x,distance,angle))
            glcms.append(torch.stack(glcm_dist,dim=1))
        glcms = torch.stack(glcms,dim=1)
        if self.mean_out:
            glcms = glcms.view(b,-1)
        else:
            glcms = glcms.view(b,-1,h,w)
            glcms = self.pooler(glcms)
        return glcms


class GLCMLoss(nn.Module):
    def __init__(self, vmin=0, vmax=255, levels=8, kernel_size=5, distanceCount=1, angleCount=8, pool=32, mean_in=False, mean_out=False):
        super(GLCMLoss, self).__init__()
        self.glcm = GLCMLayer(vmin, vmax, levels, kernel_size, distanceCount, angleCount, pool, mean_in, mean_out)
        self.mean_out = mean_out
    
    def forward(self,x_render,x_rec,mask=None):
        if mask is not None:
            hog_rec = self.glcm(x_rec*mask[:,None,...])
        else:
            hog_rec = self.glcm(x_rec)
        hog_render = self.glcm(x_render)
        loss = l1(hog_render,hog_rec)
        if self.mean_out:
            loss = loss.mean(dim=1)
        else:
            loss = loss.mean(dim=1).mean(dim=1).mean(dim=1)
        return loss
    

class LaplacianLoss(nn.Module):
    def __init__(self, max_level = 5):
        super().__init__()
        self.kernel = self.gauss_kernel()
        self.max_level = max_level

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def conv_gauss(self, x, kernel):
        x = F.pad(x, (2,2,2,2), mode='reflect')
        x = F.conv2d(x, kernel, groups=x.shape[1])
        return x
    
    def downsample(self, x):
        return x[:, :, ::2, ::2]
    
    def upsample(self, x):
        N, C, H, W = x.shape
        cc = torch.cat([x, torch.zeros(N,C,H,W).cuda()], dim = 3)
        cc = cc.view(N, C, H*2, W)
        cc = cc.permute(0,1,3,2)
        cc = torch.cat([cc, torch.zeros(N, C, W, H*2).cuda()], dim = 3)
        cc = cc.view(N, C, W*2, H*2)
        x_up = cc.permute(0,1,3,2)
        return self.conv_gauss(x_up, kernel=4*self.kernel)
    
    def lap_pyramid(self,x):
        current = x
        pyr = []
        for level in range(self.max_level):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    
    def forward(self,x_render,x_rec,mask=None):
        '''
        Based on FBA Matting implementation:
        https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
        '''
        
        if mask is not None:
            pyr_rec = self.lap_pyramid(x_rec*mask[:,None,...])
        else:
            pyr_rec = self.lap_pyramid(x_rec)
        pyr_render = self.lap_pyramid(x_render)

        return sum(l1(A[0], A[1]).mean() * (2**i) for i, A in enumerate(zip(pyr_rec, pyr_render)))
