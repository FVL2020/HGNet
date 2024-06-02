import torch
import torchvision
import torch.optim
import os
import argparse
import model
import numpy as np
from PIL import Image
import argparse
import numpy as np

 
def lowlight(image_path, output_path, model):
	img = Image.open(image_path).convert('RGB')
	img = (np.asarray(img)/255.0)
	img = torch.from_numpy(img).float()
	img = img.permute(2,0,1)
	img = img.cuda().unsqueeze(0)
	model.eval()
	enhanced_image,_ = model(img)
	enhanced_image = enhanced_image.clip(0,1)
	torchvision.utils.save_image(enhanced_image, output_path)
	return enhanced_image

def get_dataset(input_path):
	input_file_path_list = []
	file_list = os.listdir(os.path.join(input_path, 'input'))
	for filename in file_list:
		input_file_path_list.append(os.path.join(input_path, 'input', filename))
	return input_file_path_list

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, default="")
	parser.add_argument('--output', type=str, default="")
	parser.add_argument('--device', type=str, default="0")
	parser.add_argument('--weight', type=str, default="./Epoch199.pth")
	parser.add_argument('--s_curve', type=float, default=5.0, help='curve shape param of ExpAdjustNet')
	parser.add_argument('--hist_s', type=float, default=5, help='scale of hist bin smooth')
	parser.add_argument('--model_bins', type=int, default=20, help='number of bins of model input')
	parser.add_argument('--model', type=str, default="ExpAdjustNet", help='ZeroDCE or ExpAdjustNet')
	config = parser.parse_args()
	print(config)
	input_path = config.input
	output_path = config.output
	os.environ['CUDA_VISIBLE_DEVICES']=config.device
	assert os.path.exists(output_path), 'output path not exits'
	

# test_images
	with torch.no_grad():
		net = model.HGNet(bins=config.model_bins, s_curve=config.s_curve, scale_factor=config.hist_s).cuda()
		net.load_state_dict(torch.load(config.weight))	
		input_file_paths = get_dataset(input_path)
		for input_file_path in input_file_paths:
			# print(input_file_path)
			output_file_path = os.path.join(output_path, os.path.basename(input_file_path))
			pred = lowlight(input_file_path, output_file_path, net)


		

