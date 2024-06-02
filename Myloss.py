import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class L_total(nn.Module):
    def __init__(self, config):
        super(L_total, self).__init__()
        self.L_color = L_color()
        self.L_spa = L_spa()
        self.L_exp = L_exp(16,0.6)
        self.L_TV = L_TV()
        if config.hist_mode == 1:
            self.L_hist = L_hq2(bins=config.hist_bin)
        else:
            print(f'hist mode {config.hist_mode}  is not available') 
            exit()
        self.config = config
    def forward(self, img_lowlight, enhanced_image, A):
        Loss_TV = self.config.w_tv*self.L_TV(A)
        loss_spa = self.config.w_spa*torch.mean(self.L_spa(enhanced_image, img_lowlight))
        loss_col = self.config.w_col*torch.mean(self.L_color(enhanced_image))
        loss_exp = self.config.w_exp*torch.mean(self.L_exp(enhanced_image))
        loss =  Loss_TV + loss_spa + loss_col + loss_exp
        if self.config.w_hist>0:
            loss_hist = self.config.w_hist*torch.mean(self.L_hist(enhanced_image))
            loss += loss_hist
            return loss, loss_hist, loss_exp
        else:
            return loss, 0, loss_exp


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
    def forward(self, x):
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k
			
class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, enhance, org):
        b,c,h,w = org.shape
        if c>3:
            org = org[:,:3,:,:]
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        #weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        #E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):
    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


class L_hq2(nn.Module):
    def __init__(self, bins=20, scale_factor=5, eps=1e-5):
        super(L_hq2, self).__init__()
        self.bins = bins
        self.eps = eps
        self.scale_factor = scale_factor*bins
        bin_borders = np.linspace(0,1,bins+1)
        self.bin_left_border = torch.from_numpy(bin_borders[:-1]).view((bins, 1, 1, 1, 1)) # bins, b, c, h, w
        self.bin_right_border = torch.from_numpy(bin_borders[1:]).view((bins, 1, 1, 1, 1)) # bins, b, c, h, w
        
    def forward(self, x):
        self.bin_left_border = self.bin_left_border.type_as(x)
        self.bin_left_border.requires_grad = False
        self.bin_right_border = self.bin_right_border.type_as(x)
        self.bin_right_border.requires_grad = False
        x = x.unsqueeze(0).repeat(self.bins, 1,1,1,1) # bins, b, c, h, w
        x_bins = torch.sigmoid(self.scale_factor*(x-self.bin_left_border))-torch.sigmoid(self.scale_factor*(x-self.bin_right_border))
        x_bins = x_bins.permute((1,0,2,3,4)) # b, bins, c, h, w
        x_bins = x_bins.mean((3,4)) # b, bins, c
        entropy = -torch.log(x_bins+self.eps)*x_bins
        return -entropy.sum()
