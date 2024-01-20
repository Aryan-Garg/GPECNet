from IPython import embed
import glob
import pandas as pd
import pickle
import os
import torch
from torch import nn
from torch.utils import data
import random
import numpy as np

root_path="./"
dataset_type = 'image'
set_name = 'train'
scene = None
rel_path = '/trajnet_{0}/{1}/stanford'.format(dataset_type, set_name)
part_file = '/{}.txt'.format('*' if scene == None else scene)

#for file in glob.glob(root_path + rel_path + part_file):
#    scene_name = file[len(root_path+rel_path)+1:-6] + file[-5]
#    print(scene_name)
#    print(file)

#scene_name = 'bookstore0'
file = './trajnet_image/train/stanford/bookstore_0.txt'
data = np.loadtxt(fname = file, delimiter = ' ')
#print(data.shape)
data_by_id = {}
for frame_id, person_id, x, y in data:
    if person_id==100:
        if person_id not in data_by_id.keys():
            data_by_id[person_id] = []
        data_by_id[person_id].append([person_id, frame_id, x, y])

all_data_dict = data_by_id.copy()
print("Total People: ", len(list(data_by_id.keys())))

#while len(list(data_by_id.keys()))>0:
related_list = []
curr_keys = list(data_by_id.keys())
print(curr_keys)
    