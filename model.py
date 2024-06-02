import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Hist(nn.Module):
	def __init__(self, bins=20, scale_factor=5, eps=1e-5):
		super(Hist, self).__init__()
		self.bins = bins
		self.eps = eps
		self.scale_factor = scale_factor*bins
		if bins != 256:
			bin_borders = np.linspace(0,1,bins+1)
		else:
			bin_borders = np.linspace(0-1/256/2,1+1/256/2,bins+1)
		self.bin_left_border = torch.from_numpy(bin_borders[:-1]).view((bins, 1, 1, 1, 1)) # bins, b, c, h, w
		self.bin_right_border = torch.from_numpy(bin_borders[1:]).view((bins, 1, 1, 1, 1)) # bins, b, c, h, w
	def forward(self, x):
		b,c,h,w = x.shape
		self.bin_left_border = self.bin_left_border.type_as(x)
		self.bin_left_border.requires_grad = False
		self.bin_right_border = self.bin_right_border.type_as(x)
		self.bin_right_border.requires_grad = False
		x = x.unsqueeze(0).repeat(self.bins, 1,1,1,1) # bins, b, c, h, w
		x_bins = torch.sigmoid(self.scale_factor*(x-self.bin_left_border))-torch.sigmoid(self.scale_factor*(x-self.bin_right_border))
		x_bins = x_bins.permute((1,0,2,3,4)) # b, bins, c, h, w
		x_bins = x_bins.mean((3,4)) # b, bins, c
		x_bins = x_bins.view(b, self.bins*c)
		return x_bins

class HGNet(nn.Module):
    def __init__(self, bins=20, scale_factor=5, eps=1e-5, s_curve=5.0):
        super(HGNet, self).__init__()
        self.s = torch.FloatTensor([s_curve]) 
        self.hist_static_model = Hist(bins, scale_factor, eps)
        self.n_hist_conv = [0,0,16,16,16]
        self.k_hist_conv = [0,0,3,1,1]
        self.n_img_conv = [32,32,0,0,0]
        self.n_skip_connect = [0,0,0,0,0,0]
        self.hist_conv_filter_sizes = []
        self.mlp_out = 0
        self.in_channels = [3]
        for i in range(len(self.n_hist_conv)):
            self.hist_conv_filter_sizes.append(self.in_channels[i]*self.k_hist_conv[i]*self.k_hist_conv[i]*self.n_hist_conv[i]+self.n_hist_conv[i])
            self.mlp_out += self.hist_conv_filter_sizes[-1]
            self.in_channels.append(self.n_img_conv[i]+self.n_hist_conv[i]+self.n_skip_connect[i]) 
        self.hist_mlp = nn.Sequential(
            nn.Linear(bins*3, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16),
            nn.LeakyReLU(),
            nn.Linear(16, self.mlp_out)
        )
        self.img_conv = nn.ModuleList([
			nn.Conv2d(self.in_channels[i],self.n_img_conv[i],3,1,1,bias=True) if self.n_img_conv[i]>0 else \
				nn.Identity() \
				 for i in range(len(self.n_img_conv))
		])
        self.out_conv = nn.Conv2d(self.in_channels[-1],3,3,1,1,bias=True)
    def curve(self, x, alpha):
        self.s = self.s.type_as(x)
        s = self.s
        s.requires_grad = False
        min_y = 1 / (1 +torch.exp(s))
        max_y = 1 / (1 +torch.exp(-s))
        a = max_y-min_y
        b = min_y
        y = 1/(a*x+b)-1
        y = 1 + y*torch.exp(-2*s*alpha)
        y = 1/(a*y) - b/a
        return y
    def hist_conv(self, w_filters, b_filters, batch_xs):
        # split images
        x_list=batch_xs.split(1)
        w_filter_list=w_filters.split(1)
        b_filter_list=b_filters.split(1)
        output = []
        # do conv per image
        for x, w_filter, b_filter in zip(x_list, w_filter_list, b_filter_list):
            output.append(F.conv2d(x, w_filter.squeeze(dim=0), b_filter.squeeze(dim=0), padding='same'))
        return torch.cat(output, dim=0)
    def forward(self, x_img):
        # mlp forward to get parameters of hist conv filter
        hist_conv_paras = self.hist_mlp(self.hist_static_model(x_img))
        # split into conv filter
        hist_conv_para_list = torch.split(hist_conv_paras, self.hist_conv_filter_sizes, dim=1) # list of tensor of (b, lwi+lbi)
        hist_conv_filter_weights = [] # list of tensor of (b, ci, ii, 3, 3)
        hist_conv_filter_bias = [] # list of tensor of (b, ci)
        for i in range(len(self.n_hist_conv)):
            if self.n_hist_conv[i]!= 0:
                weights, bias = torch.split(hist_conv_para_list[i], [self.in_channels[i]*self.k_hist_conv[i]*self.k_hist_conv[i]*self.n_hist_conv[i], self.n_hist_conv[i]], dim=1)
                hist_conv_filter_weights.append(weights.view(-1, self.n_hist_conv[i], self.in_channels[i], self.k_hist_conv[i], self.k_hist_conv[i]))
                hist_conv_filter_bias.append(bias)
            else:
                hist_conv_filter_weights.append(None)
                hist_conv_filter_bias.append(None)
        # conv forward
        x_list = [x_img]
        for i in range(len(self.n_hist_conv)):
            if self.n_hist_conv[i]>0 and self.n_img_conv[i]>0:
                x = torch.cat([self.img_conv[i](x_list[-1]), self.hist_conv(hist_conv_filter_weights[i], hist_conv_filter_bias[i], x_list[-1])], dim=1)
            elif self.n_hist_conv[i]>0:
                x = self.hist_conv(hist_conv_filter_weights[i], hist_conv_filter_bias[i], x_list[-1])
            else:
                x = self.img_conv[i](x_list[-1])
            x = torch.relu(x)
            x_list.append(x)
        x = self.out_conv(x)
        return self.curve(x_img, x), x