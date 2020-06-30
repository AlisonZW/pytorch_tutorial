import os
import random
import shutil

def data_spliter():

	random.seed(1)

	dataset_dir = os.path.join("..","dog_cat")
	split_dir = os.path.join("..","dog_cat","split")
	train_dir = os.path.join(split_dir, "train")
	valid_dir = os.path.join(split_dir, "valid")
	test_dir = os.path.join(split_dir, "test")


	train_pct = 0.8
	valid_pct = 0.1
	test_pct  = 0.1

	for root, dirs, files in os.walk(dataset_dir):
		for sub_dir in dirs:
			imgs = os.listdir(os.path.join(root, sub_dir))
			imgs =list(filter(lambda x: x.endwith('.jpg'), imgs))
			random.shuffle(imgs)
			img_count = len(imgs)

			train_point = int(img_count * train_pct)
			valid_point = int(img_count * (train_pct + valid_pct))

			
