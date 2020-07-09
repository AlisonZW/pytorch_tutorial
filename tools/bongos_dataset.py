import os
import random 
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
bongos_label = {"cat": 0, "dog": 1}

class BongosDataset(Dataset):
	def __init__(self, data_dir, transform=None):
		self.label_name ={"cat":0, "dog":1}
		self.data_info = self.get_img_info(data_dir)
		self.transform = transform

	def __getitem__(self, index):
		path_img, label = self.data_info[index]
		img = Image.open(path_img).convert('RGB')

		if self.transform is not None:
			img = self.transform(img)

		return img, label 

	def __len__(self):
		return len(self.data_info)

	@staticmethod
	def get_img_info(data_dir):
		data_info = list()
		for root, dirs, _ in os.walk(data_dir):
			for sub_dir in dirs:
				img_names = os.listdir(os.path.join(root, sub_dir))
				img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
				#print(img_names)
				print(len(img_names))
				for i in range(len(img_names)):
					img_name = img_names[i]
					#print("img_names:{}, root:{}, sub_dir:{}, img_names:{}".format(img_names, root, sub_dir, img_names))
					path_img = os.path.join(root, sub_dir, img_name)
					label = bongos_label[sub_dir]
					data_info.append((path_img, int(label)))
		return data_info