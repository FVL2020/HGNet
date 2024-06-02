import shutil
import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import argparse
import dataloader
import model
import Myloss
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('ZeroDCE_w_HGConv') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
	# tensorboard
	if os.path.exists(os.path.join(config.snapshots_folder, 'tb')):
		shutil.rmtree(os.path.join(config.snapshots_folder, 'tb'))
	os.makedirs(os.path.join(config.snapshots_folder, 'tb'))
	writer = SummaryWriter(log_dir=os.path.join(config.snapshots_folder, 'tb'))
	global_step = 0
	# load model
	os.environ['CUDA_VISIBLE_DEVICES']=config.device
	net = model.HGNet(bins=config.model_bins, s_curve=config.s_curve, scale_factor=config.hist_s).cuda()
	net.apply(weights_init)
	if config.load_pretrain == True:
		net.load_state_dict(torch.load(config.pretrain_dir))
	if len(config.device)>1:
		net = nn.DataParallel(net)
	# build dataset
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path, config.training_strategy, config.input_size)		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True if config.training_strategy=='random' else False,
					     num_workers=config.num_workers, pin_memory=True)
	# define loss function
	L_total = Myloss.L_total(config)

	# optimizer
	optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)  # adjust beta1 to momentum
	# train
	net.train()
	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):
			img_lowlight = img_lowlight.cuda()
			# forward
			enhanced_image, A  = net(img_lowlight)
			# get loss
			loss, loss_hist, loss_exp = L_total(img_lowlight, enhanced_image, A)
			# optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# log
			if ((iteration+1) % config.log_iter) == 0:
				print(f"epoch: {epoch+1}, iter: {iteration+1}, loss: {loss.item()}, hist loss: {loss_hist.item() if loss_hist!= 0 else 0}, exp loss: {loss_exp.item()}")
				writer.add_scalar('loss', loss.item(), global_step)
				writer.add_scalar('histloss', loss_hist.item() if loss_hist!= 0 else 0, global_step)
				writer.add_scalar('exploss', loss_exp.item(), global_step)
			global_step += 1
		writer.add_image('images/output', torchvision.utils.make_grid(enhanced_image), global_step)
		if img_lowlight.shape[1]>3:
			writer.add_image('images/input', torchvision.utils.make_grid(img_lowlight[:,:3,:,:]), global_step)
		else:
			writer.add_image('images/input', torchvision.utils.make_grid(img_lowlight), global_step)
		# save param
		if ((epoch+1) % config.snapshot_iter) == 0:
			torch.save(net.module.state_dict() if len(config.device)>1 else net.state_dict(), os.path.join(config.snapshots_folder, "Epoch" + str(epoch) + '.pth')) 		
	writer.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="")
	parser.add_argument('--model_bins', type=int, default=20, help='number of bins of model input')
	parser.add_argument('--model', type=str, default="StraightNet6", help='ZeroDCE or ExpAdjustNet')
	parser.add_argument('--loss_mode', type=int, default=0, help='mode of loss')
	parser.add_argument('--regular_mode', type=int, default=0, help='mode of loss')
	parser.add_argument('--hist_mode', type=int, default=1, help='mode of hist loss')
	parser.add_argument('--training_strategy', type=str, default='progressive', help='training strategy, random|recurrence|progressive|average')
	parser.add_argument('--input_size', type=int, default=256, help='size of training img')
	parser.add_argument('--hist_bin', type=int, default=256, help='number of hist bin')
	parser.add_argument('--hist_s', type=float, default=5, help='scale of hist bin smooth')
	parser.add_argument('--s_curve', type=float, default=5.0, help='curve shape param of ExpAdjustNet')
	parser.add_argument('--w_tv', type=float, default=700.0, help='weight of tv loss')
	parser.add_argument('--w_spa', type=float, default=0.0, help='weight of spatial loss')
	parser.add_argument('--w_exp', type=float, default=1.0, help='weight of exp loss')
	parser.add_argument('--w_col', type=float, default=2.0, help='weight of color loss')
	parser.add_argument('--w_hist', type=float, default=0.003, help='weight of hist loss')
	parser.add_argument('--lr', type=float, default=0.0003)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--log_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="")
	parser.add_argument('--load_pretrain', action='store_true', help='load pretrained weights or not')
	parser.add_argument('--pretrain_dir', type=str, default= "")
	parser.add_argument('--device', type=str, default= "0")
	parser.add_argument('--seed', type=int, default= 930)
	config = parser.parse_args()
	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed_all(config.seed)
	np.random.seed(config.seed)
	random.seed(config.seed)
	print(config)
	train(config)
