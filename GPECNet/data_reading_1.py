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
    if person_id not in data_by_id.keys():
        data_by_id[person_id] = []
    data_by_id[person_id].append([person_id, frame_id, x, y])

all_data_dict = data_by_id.copy()
print("Total People: ", len(list(data_by_id.keys())))

related_list = []
#print(curr_keys)
current_batch = []
full_dataset = []
current_size = 0
while len(list(data_by_id.keys()))>0:
    curr_keys = list(data_by_id.keys())
    if current_size<3:
        pass
    else:
#        full_dataset.append(current_batch.copy())
#        current_size = 0
#        current_batch = []
        break
    
    current_batch.append((all_data_dict[curr_keys[0]]))
    current_size+=1
#    print(curr_keys[0])
    print(np.shape(current_batch))
#    print(np.shape(full_dataset))
    del data_by_id[curr_keys[0]]

curr_keys = list(data_by_id.keys())
p1_traj, p2_traj = np.array(all_data_dict[curr_keys[0]]), np.array(all_data_dict[curr_keys[1]])
p1_time, p2_time = p1_traj[:,1], p2_traj[:,1]
p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]
print(p1_time)
print(p2_time)
print(p1_x)      
    

    