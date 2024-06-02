import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

class train_strategy():
	def __init__(self, training_strategy, train_list):
		self.luminax_list = []
		for i in train_list:
			self.luminax_list.append(np.mean(np.asarray(Image.open(i).convert('RGB'))))
		self.training_strategy = training_strategy
		self.sorted_index = []
		if not self.training_strategy == 'random':
			self.sorted_index = np.argsort(self.luminax_list).tolist()
		if self.training_strategy == 'average':
			num_folds = 4
			len_fold = len(self.luminax_list)//num_folds
			self.reversed_index = self.sorted_index.copy()
			self.reversed_index.reverse()
			self.presem = self.sorted_index[:len_fold] + self.reversed_index[2*len_fold:3*len_fold] + self.sorted_index[2*len_fold:3*len_fold] + self.reversed_index[:len_fold]
			self.mapping = [self.presem[(i%num_folds)*len_fold + i//4] for i in range(len(self.luminax_list))]
		if self.training_strategy == 'progressive':
			self.sorted_index.reverse()
			self.current_indexes = self.sorted_index[:len(self.sorted_index)//2]
			self.current_indexes.reverse()
			self.stage = 0
			self.n_step = [10,22,36,52,80,9999]
			self.epoch = 0
	def step(self):
		assert self.training_strategy == 'progressive'
		self.stage += 1
		self.current_indexes = self.sorted_index[:len(self.current_indexes)+len(self.sorted_index)//10]
		self.current_indexes.reverse()
		return
	def get(self, index):
		if self.training_strategy == 'random':
			return index
		if self.training_strategy == 'recurrence':
			return self.sorted_index[index]
		if self.training_strategy == 'average':
			return self.mapping[index]
		if self.training_strategy == 'progressive':
			if index == 0:
				self.epoch += 1
				if self.epoch == self.n_step[self.stage]:
					self.step()
			return self.current_indexes[index%len(self.current_indexes)]
def populate_train_list(lowlight_images_path):
	img_list = os.listdir(lowlight_images_path)
	img_list.sort()
	image_list_lowlight = [os.path.join(lowlight_images_path, i) for i in img_list]
	return image_list_lowlight

class lowlight_loader(data.Dataset):
	def __init__(self, lowlight_images_path, training_strategy, input_size = 256):
		assert training_strategy in ['random', 'recurrence', 'progressive', 'average']
		self.data_list = populate_train_list(lowlight_images_path) 
		self.fetch_seq = train_strategy(training_strategy, self.data_list)
		self.size = input_size
		print("Total training examples:", len(self.data_list))

	def __getitem__(self, index):
		data_lowlight_path = self.data_list[self.fetch_seq.get(index)]
		data_lowlight = Image.open(data_lowlight_path).convert('RGB') # (h,w,c)
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()
		return data_lowlight.permute(2,0,1) # (c,h,w)

	def __len__(self):
		return len(self.data_list)
