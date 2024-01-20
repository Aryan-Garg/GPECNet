from IPython import embed
import glob
import pandas as pd
import pickle
import os
import torch
from torch import nn
#from torch.utils import data
import random
import numpy as np

root_path="./"
dataset_type = 'image'
set_name = 'train'
scene = None
rel_path = '/trajnet_{0}/{1}/stanford'.format(dataset_type, set_name)
part_file = '/{}.txt'.format('*' if scene == None else scene)

for file in glob.glob(root_path + rel_path + part_file):
    scene_name = file[len(root_path+rel_path)+1:-6] + file[-5]
#    print(scene_name)  #aaa_0, bookstore_0, bookstore_1, bookstore_2
#    print(file)  #corresponsing path for aaa_0, bookstore_1, bookstore_2

#scene_name = 'bookstore0'
file = './trajnet_image/train/stanford/aaa_0.txt'
data = np.loadtxt(fname = file, delimiter = ' ')
#print(data.shape)   #(16,4)
data_by_id = {}
for frame_id, person_id, x, y in data:
    if person_id not in data_by_id.keys():
        data_by_id[person_id] = []
    data_by_id[person_id].append([person_id, frame_id, x, y])

all_data_dict = data_by_id.copy()
#print("Total People: ", len(list(data_by_id.keys())))  #4
curr_keys = list(data_by_id.keys())
related_list = []
#print(curr_keys)  #100, 129, 139, 194
current_batch = []
full_dataset = []
full_dataset = []
full_masks = []
current_batch = []
current_size = 0
batch_size = 2
mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]
while len(list(data_by_id.keys()))>0:
    curr_keys = list(data_by_id.keys())
    if current_size<2:
        pass
    else:
        full_dataset.append(current_batch.copy())
        print("Full datset iterative", np.shape(full_dataset))
        mask_batch = np.array(mask_batch)
        full_masks.append(mask_batch[0:len(current_batch), 0:len(current_batch)])
        current_size = 0
        social_id = 0
        current_batch = []
        mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]
    
    current_batch.append((all_data_dict[curr_keys[0]]))
    related_list.append(current_size)   #(0,1,0,1)
    print('related list', related_list)
    current_size+=1
#    print(curr_keys[0])
    print("current batch shape", np.shape(current_batch))
    print("full dataset shape", np.shape(full_dataset))
    del data_by_id[curr_keys[0]]
full_dataset.append(current_batch)  #(2,2,3,4)
mask_batch = np.array(mask_batch)
full_masks.append(mask_batch[0:len(current_batch), 0:len(current_batch)])
print("final full dataset size", np.shape(full_dataset))
print('full mask', np.shape(full_masks))   #(2,2,2)
train  = [full_dataset, full_masks]
print('train', len(train))
traj_new = []
for t in full_dataset:
    t = np.array(t)
    t = t[:,:,2:]
    print(t)
    traj_new.append(t)
    
print('traj_new',np.shape(traj_new))
device = torch.device('cpu')
traj_new = np.array(traj_new)
starting_pos = traj_new[:, :,2,:]
print(starting_pos)
print(starting_pos.shape)
x = torch.DoubleTensor(starting_pos).to(device)
x = x.contiguous().view(-1, x.shape[1]*x.shape[2]) # (x,y,x,y ... )
print(x)
print(x.shape)
#for b in traj_new:
#    starting_pos = b[:,2,:]
#    print(starting_pos)
#    print(starting_pos.shape)
#    print(starting_pos.shape[1])
#    print(starting_pos.shape[2])
#    x = torch.DoubleTensor(starting_pos).to(device)
#    x = x.contiguous().view(-1, x.shape[1]*x.shape[2]) # (x,y,x,y ... )
#    print(x)

#curr_keys = list(data_by_id.keys())
#p1_traj, p2_traj = np.array(all_data_dict[curr_keys[0]]), np.array(all_data_dict[curr_keys[1]])
#p1_time, p2_time = p1_traj[:,1], p2_traj[:,1]
#p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
#p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]
#print(p1_time)
#print(p2_time)
#print(p1_x)      
#    
#
#    