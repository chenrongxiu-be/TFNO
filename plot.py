# -*- coding: utf-8 -*-
"""
Title: Real-time surrogate model of vehicle-bridge interaction
Author: Chen Rongxiu
Created: 24-06-24
Updated: 25-09-12
License: MIT License

Description:
------
Plots

Dependencies:
-------------
numpy, pytorch, einops, matplotlib

Notes:
------
- Make sure to install all dependencies before running.
- The code assumes input data are saved in the "data" folder as described in the README.

"""

import numpy as np
import torch
from einops import repeat
import matplotlib
import matplotlib.pyplot as plt
import os, json, configparser, sys, importlib, random

torch.set_default_dtype(torch.float64)

class Data():
    def __init__(self, bridge, node_num, maxmin_dataset, target_node, target_axis,
                 device):
        self.bridge = bridge
        self.node_num = node_num
        temp = "{:03d}".format(maxmin_dataset)
        self.path_maxmin = os.path.join('data', bridge, f'dataset_{temp}')
        self.tar_node = target_node
        self.tar_axis = target_axis    
        self.device = device
        
    def load_data(self, dsno): 
        temp = "{:03d}".format(dsno)
        path = os.path.join('data', self.bridge, f'dataset_{temp}')
                
        # read node coordinates
        temp = np.load(os.path.join(path, 'grid.npy'), allow_pickle=True)
        data_nodegrid = torch.from_numpy(temp) 
        nodeid1, nodeid2 = self.tar_node[0]-1, self.tar_node[1]
        if self.tar_axis == 'x':  
            grid_z = data_nodegrid[nodeid1:nodeid2, 2].reshape(-1, 1)  
            grid_y = data_nodegrid[nodeid1:nodeid2, 1].reshape(-1, 1)
            data_nodegrid = torch.cat((grid_z, grid_y), dim=-1)
        elif self.tar_axis == 'y':
            data_nodegrid = data_nodegrid[nodeid1:nodeid2, 0::2]  
        else:  # z
            data_nodegrid = data_nodegrid[nodeid1:nodeid2, 0:2]
                 
        temp = np.load(os.path.join(path, 'testdata_def.npy'), allow_pickle=True)
        data_def = torch.from_numpy(temp)
        temp = np.load(os.path.join(path, 'testdata_veh.npy'), allow_pickle=True)
        data_veh = torch.from_numpy(temp)                
        temp = np.load(os.path.join(path, 'testdata_timesize.npy'), allow_pickle=True) 
        data_timesize = torch.from_numpy(temp)        
        temp = np.load(os.path.join(path, 'testdata_caselabel.npy'), allow_pickle=True) 
        data_caselabel = temp.tolist()
        
        # normalize data
        temp = np.load(os.path.join(self.path_maxmin, f'maxmin_def{self.tar_axis}.npy'), allow_pickle=True) 
        def_maxmin = torch.from_numpy(temp)
        temp = np.load(os.path.join(self.path_maxmin, 'maxmin_veh.npy'), allow_pickle=True) 
        veh_maxmin = torch.from_numpy(temp)
    
        if self.tar_axis == 'x':
            node_idx = [nodeid1, nodeid2]
        elif self.tar_axis == 'y':
            node_idx = [self.node_num+nodeid1, self.node_num+nodeid2]
        else:  # 'z'
            node_idx = [2*self.node_num+nodeid1, 2*self.node_num+nodeid2]
            
        data_nodegrid = self.norm_init_data(data_nodegrid, 'grid', [])  
        data_def_tarnode_norm, def_dist = \
            self.norm_init_data(data_def[:, node_idx[0]:node_idx[1], :], 'def', def_maxmin)
        data_veh, _ = self.norm_init_data(data_veh, 'veh', veh_maxmin)    
        
        # build dataloaders
        dataloader =  torch.utils.data.DataLoader(CustomDataset( 
                          data_caselabel,
                          data_def,  
                          data_veh, 
                          data_timesize), 
                          batch_size=1, shuffle=False, generator=torch.Generator(self.device))
        
        return dataloader, data_nodegrid, def_dist, node_idx
        
    def norm_init_data(self, data, data_type, data_minmax):
        a, b = 0, 1
        
        if data_type == 'grid':
            data_max, _ = torch.max(data, dim=0)
            data_min, _ = torch.min(data, dim=0)
            
            for i in range(2):
                data[:, i] = (b - a) * (data[:, i]- data_min[i]) / \
                    (data_max[i] - data_min[i] + 1e-8) + a
            return data
            
        elif data_type == 'def':
            data_min, data_max = data_minmax[0], data_minmax[1]
            normed_data = (b - a) * (data - data_min) / (data_max - data_min + 1e-8) + a  
            
            data_dist = [data_min, data_max]
            
            return normed_data, data_dist
        
        elif data_type == 'veh':
            data_min = repeat(data_minmax[:, 0], 'p -> a k p b', a=1, k=1, b=1) 
            data_max = repeat(data_minmax[:, 1], 'p -> a k p b', a=1, k=1, b=1) 
            
            temp_shape = data.shape
            data_min_ = data_min.expand(temp_shape[0], temp_shape[1], data_min.shape[2], temp_shape[3]) 
            data_max_ = data_max.expand(temp_shape[0], temp_shape[1], data_max.shape[2], temp_shape[3])
            normed_data = (b - a) * (data - data_min_) / (data_max_ - data_min_ + 1e-8) + a
                        
            data_dist = [data_min, data_max]
            
            return normed_data, data_dist        
                    
        else:
            sys.exit('data_type is wrong!')
            
            
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, label, *data):
        super(CustomDataset, self).__init__()    
               
        self.label = label
        self.data = []
        for idata in data: self.data.append(idata)        
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):   
        result = [self.label[idx]]
        for idata in self.data: result.append(idata[idx, ...])
        return tuple(result)


def get_ip(i_iter, ip_def, pred_def, def_dist, tar_nodenum,
           batch_num, ip_timestep, op_timestep, delta_t, ts_idx, model_type):
    if i_iter == 0: # initialize input           
        a, b = 0, 1
        min, max = def_dist[0], def_dist[1]
        ini_def = (b - a) * (0 - min) / (max - min) + a
        ini_def = ini_def.to(device)
        
        if model_type == 'narx' or model_type == 'lstm':
            ip_def = []
            for i in range(tar_nodenum):
                noise = 1. / 1e6 * torch.randn(batch_num, ip_timestep)
                ip_def.append(noise + ini_def * torch.ones(batch_num, ip_timestep))
            ip_veh = data_veh[:, 0, :, 0]
            
        else:  # tfno or fmionet
            noise = 1. / 1e6 * torch.randn(batch_num, tar_nodenum, ip_timestep) 
            ip_def = noise + ini_def * torch.ones(batch_num, tar_nodenum, ip_timestep)   
            ip_veh = data_veh[:, :1, :, 0]
        
        if model_type == 'tfno': 
            timestamp = torch.arange(0, ip_timestep, 1) * delta_t      
        
    else:  # i_iter >= 1
        if model_type == 'narx' or model_type == 'lstm':
            ip_def = []
            for i in range(pred_def.shape[1]):
                ip_def.append(pred_def[:, i, :])
            ip_veh = data_veh[:, 0, :, ts_idx[0]-1]
                
        else:  # tfno or fmionet
            ip_def = pred_def
            ip_veh = data_veh[:, :1, :, ts_idx[0]-1]
        
        if model_type == 'tfno': 
            timestamp = torch.arange(ts_idx[0], (ts_idx[0]+ip_timestep), 1) * delta_t
        
    if model_type == 'tfno':
        timestamp = repeat(timestamp, 't -> b k t', b=batch_num, k=1)         
        return ip_def, ip_veh, timestamp
    
    elif model_type == 'fmionet':
        # for input
        ip_timestamp = torch.arange(ts_idx[0] - op_timestep, ts_idx[0], 1) * delta_t
        # for output
        op_timestamp = torch.arange(ts_idx[0], ts_idx[0] + op_timestep, 1) * delta_t
        timestamp = [ip_timestamp, op_timestamp]        
        return ip_def, ip_veh, timestamp
    
    else:  # narx and lstm
        return ip_def, ip_veh


def recurrent(i_iter, delta_ts, ip_def, ip_veh, timestamp, hn, cn,
              model, model_type):
    # predict    
    if model_type == 'tfno':
        pred_def = model(ip_def, ip_veh, timestamp)
        
    elif model_type == 'narx' or model_type == 'lstm':            
        for i in range(len(model)):
            if model_type == 'narx':
                temp = model[i](ip_def[i], ip_veh)
            else:  # lstm
                temp, (temp_hn, temp_cn) = model[i](ip_def[i], ip_veh, hn[i], cn[i])
                hn[i], cn[i] = temp_hn, temp_cn                    
            
            temp = temp.unsqueeze(1)  
            if i==0:
                pred_def = temp
            else:
                pred_def = torch.cat((pred_def, temp), dim=1)  

    elif model_type == 'fmionet':
        pred_def = model(ip_def, ip_veh, timestamp[0], timestamp[1])
        
    else:            
        pass
    
    # if iteration exceeds maximum timesize, truncate redundant part
    if delta_ts > 0:
        pred_def_ = pred_def[..., :-delta_ts] 
    else:
        pred_def_ = pred_def    
    return pred_def_


def denorm(data, min, max):
    a, b = 0, 1
    min, max = min.to(device), max.to(device)
    return (data - a) * (max - min) / (b - a) + min


def plot_cm(pred_def, tar_def, delta_t, factor=1000):
    node_num = pred_def.shape[0]
    timestep = pred_def.shape[1]
    
    def_lim = [-2, 6]
    def_err_lim = [0, 1]
    colors = [(0.12,0.27,0.43),(0.22,0.4,0.58),(0.32,0.56,0.68),(0.45,0.74,0.84),
              (0.67,0.86,0.88),(1,0.9,0.72),(1,0.82,0.44),
              (0.97,0.67,0.35),(0.94,0.54,0.28),(0.91,0.38,0.33)]
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)  
    colors = [(0.45,0.74,0.84),(0.67,0.86,0.88),(1,0.9,0.72),(1,0.82,0.44)]
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)  
    
    fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(15,2), dpi=800)
    
    t = np.arange(0, timestep, 1) * delta_t
    
    X, Y = np.meshgrid(t, np.arange(1, node_num+1, 1))
    c = axs[0].pcolor(X, Y, factor*tar_def.detach().cpu().numpy(), vmin=def_lim[0], vmax=def_lim[1], cmap=cmap1)
    fig.colorbar(c, ax=axs[0])
    c = axs[1].pcolor(X, Y, factor*pred_def.detach().cpu().numpy(), vmin=def_lim[0], vmax=def_lim[1], cmap=cmap1)
    fig.colorbar(c, ax=axs[1])
    mae = factor * torch.abs(tar_def - pred_def)
    c = axs[2].pcolor(X, Y, mae.detach().cpu().numpy(), vmin=def_err_lim[0], vmax=def_err_lim[1], cmap=cmap2)
    fig.colorbar(c, ax=axs[2])   
    
    title = ['Deflection']
    subtitle = ['ref.', 'pred.', 'absolute error']
    unit = ['mm']
    tlim = timestep * delta_t
    for j in range(3):
        axs[j].set_title(f'{title[0]} {subtitle[j]} ({unit[0]})')
        axs[j].set_xlim(0, tlim)
        axs[j].set_yticklabels(axs[j].get_yticks().astype(int))
                  
    return fig



# ----------------------------------
# ---------- Configs ---------------
# ----------------------------------
delta_t = 0.005
ip_timestep = 512
op_timestep = 512
bridge = 'continuous'
# choose the following (1) ~ (3) configurations according to the target data,
# but do not alter them.
# ----- (1) for the continuous bridge deck -----
node_num = 45
target_datasets = [1, 2]
target_node = [12, 22]
target_axis = 'z'
# ----- (1-2) for the continuous bridge deck (fine mesh) -----
# node_num = 45
# target_datasets = [1]
# target_node = [1, 18]
# target_axis = 'z'
# ----- (1-3) for the continuous bridge deck (fine mesh) -----
# node_num = 45
# target_datasets = [1]
# target_node = [1, 133]
# target_axis = 'z'
# ----- (2) for the cable-stayed bridge deck -----
# node_num = 160
# target_datasets = [1, 2]
# target_node = [1, 46]
# target_axis = 'z'
# ----- (3) for the cable-stayed bridge pylon -----
# node_num = 160
# target_datasets = [1, 2]
# target_node = [103, 122]
# target_axis = 'x'

model_type = 'tfno'  # tfno, fmionet, lstm, or narx
title = '240624_TrainTFNO'
# ----------------------------------
# ----------------------------------
# ----------------------------------


### CUDA ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
torch.cuda.empty_cache() 


### Load data ###
print('Read data...')
DataClass = Data(bridge, node_num, target_datasets[-1], target_node, target_axis, device)
for i_dsno in target_datasets:
    dataloader, nodegrid, def_dist, node_idx = DataClass.load_data(i_dsno)
    if 'dataloaders' not in locals():
        dataloaders = [dataloader]
    else:
        dataloaders += [dataloader]
tar_nodenum = node_idx[1] - node_idx[0]


### Load model ###   
config_path = os.path.join('config', f'config_{model_type}_{title}.ini')
cf = configparser.ConfigParser()
try:
    cf.read(config_path, encoding='utf-8')
except:
    cf.read(config_path, encoding='shift-jis')
# Read configs and load model according to model type
if model_type == 'tfno':
    embed_dim = cf.getint('model', 'embed_dim')   
    trans_layernum = cf.getint('model','trans_layernum')
    trans_headnum = cf.getint('model','trans_headnum')
    trans_hiddendim = cf.getint('model','trans_hiddendim')
    trans_dropout = cf.getfloat('model','trans_dropout')    
    lift_channels = json.loads(cf.get('model', 'lift_channels'))
    lift_dropout = cf.getfloat('model','lift_dropout')
    fourier_layernum = cf.getint('model','fourier_layernum')
    n_modes = json.loads(cf.get('model', 'n_modes'))    
    proj_channels = json.loads(cf.get('model', 'proj_channels'))
    proj_dropout = cf.getfloat('model','proj_dropout')
    
    spec = importlib.util.spec_from_file_location('train_tfno', r'train_tfno.py')
    foo = importlib.util.module_from_spec(spec)
    sys.modules['train_tfno'] = foo
    spec.loader.exec_module(foo)
    mymodel = foo.Build_model(nodegrid, ip_timestep, embed_dim, tar_nodenum,
                              trans_layernum, trans_headnum, trans_hiddendim, trans_dropout, 
                              lift_channels, lift_dropout, fourier_layernum, n_modes, proj_channels, proj_dropout).to(device)
    
    model_path = os.path.join('model', title, 'model.pth')
    checkpoint = torch.load(model_path) 
    mymodel.load_state_dict(checkpoint['model_state_dict'])
        
elif model_type == 'fmionet':    
    layer_sizes_branch1 = json.loads(cf.get('model', 'layer_sizes_branch1'))
    layer_sizes_branch2 = json.loads(cf.get('model', 'layer_sizes_branch2'))
    layer_sizes_trunk = json.loads(cf.get('model', 'layer_sizes_trunk'))
    fourier_layernum = cf.getint('model','fourier_layernum')
    n_modes = json.loads(cf.get('model', 'n_modes'))
    proj_channels = json.loads(cf.get('model', 'proj_channels'))
    proj_dropout = cf.getfloat('model','proj_dropout')
    
    spec = importlib.util.spec_from_file_location('train_fmionet', r'train_fmionet.py')
    foo = importlib.util.module_from_spec(spec)
    sys.modules['train_fmionet'] = foo
    spec.loader.exec_module(foo)
    mymodel = foo.Build_model(nodegrid, ip_timestep,
                              layer_sizes_branch1, layer_sizes_branch2, layer_sizes_trunk, 
                              fourier_layernum, n_modes, proj_channels, proj_dropout).to(device)
    
    model_path = os.path.join('model', title, 'model.pth')
    checkpoint = torch.load(model_path) 
    mymodel.load_state_dict(checkpoint['model_state_dict'])
    
elif model_type == 'lstm':
    hidden_size = cf.getint('model','hidden_size')
    num_layers = cf.getint('model','num_layers')
    
    spec = importlib.util.spec_from_file_location('train_lstm', r'train_lstm.py')
    foo = importlib.util.module_from_spec(spec)
    sys.modules['train_lstm'] = foo
    spec.loader.exec_module(foo)
    
    submodel_num = tar_nodenum
    mymodel = []
    for i in range(submodel_num):
        mymodel.append(foo.Build_model(hidden_size, num_layers).to(device))
        
        model_path = os.path.join('model', title, f'model{str(i).zfill(2)}.pth')
        checkpoint = torch.load(model_path) 
        mymodel[i].load_state_dict(checkpoint['model_state_dict'])
                        
elif model_type == 'narx':
    hidden_size = cf.getint('model','hidden_size')
    num_layers = cf.getint('model','num_layers')
    
    spec = importlib.util.spec_from_file_location('train_narx', r'train_narx.py')
    foo = importlib.util.module_from_spec(spec)
    sys.modules['train_narx'] = foo
    spec.loader.exec_module(foo)
    
    submodel_num = tar_nodenum
    mymodel = []
    for i in range(submodel_num):
        inputsize = 29 + ip_timestep # input = [veh_para, y(t-1),...,y(t-ip_timestep-1)]
        output_size = op_timestep
        mymodel.append(foo.Build_model(inputsize, output_size, hidden_size, num_layers).to(device))
        
        model_path = os.path.join('model', title, f'model{str(i).zfill(2)}.pth')
        checkpoint = torch.load(model_path) 
        mymodel[i].load_state_dict(checkpoint['model_state_dict'])
                
else:
    sys.exit('model_type must be tfno, fmionet, lstm, or narx!')


### Inference ###   
print('Inference...')
data_iter = iter(dataloaders[0])
for _ in range(random.randint(0, 99)):  # randomly pick a sample
    data_caselabel, data_def, data_veh, data_timesize = next(data_iter)  

data_def, data_veh = data_def.to(device), data_veh.to(device)
    
timestep = torch.max(data_timesize).item()

stride = op_timestep  
pred_iters = int((timestep-1) / stride) + 1
ip_def, pred_def, ts_idx = None, None, None
batch_num = data_def.shape[0]

for i_iter in range(pred_iters):   
    if model_type == 'tfno':           
        ip_def, ip_veh, timestamp = \
            get_ip(i_iter, ip_def, pred_def, def_dist, tar_nodenum,
                   batch_num, ip_timestep, op_timestep, delta_t, ts_idx, model_type)
            
        # prepare timestep index 
        ts_idx = [i_iter*stride + 1, i_iter*stride + 1 + op_timestep]
        delta_ts = ts_idx[1] - timestep 
        if delta_ts > 0:
            ts_idx[1] = timestep
        
        # predict
        pred_def = recurrent(i_iter, delta_ts, ip_def, ip_veh, timestamp, [], [],
                             mymodel, model_type)
    
    elif model_type == 'fmionet':
        # prepare timestep index 
        ts_idx = [i_iter*stride + 1, i_iter*stride + 1 + op_timestep]
        delta_ts = ts_idx[1] - timestep 
        if delta_ts > 0:
            ts_idx[1] = timestep
        
        ip_def, ip_veh, timestamp = \
            get_ip(i_iter, ip_def, pred_def, def_dist, tar_nodenum,
                   batch_num, ip_timestep, op_timestep, delta_t, ts_idx, model_type)
        
        # predict
        pred_def = recurrent(i_iter, delta_ts, ip_def, ip_veh, timestamp, [], [],
                             mymodel, model_type)
    
    elif model_type == 'lstm':
        # prepare timestep index 
        ts_idx = [i_iter*stride + 1, i_iter*stride + 1 + op_timestep]
        delta_ts = ts_idx[1] - timestep 
        if delta_ts > 0:
            ts_idx[1] = timestep
            
        ip_def, ip_veh = \
            get_ip(i_iter, ip_def, pred_def, def_dist, tar_nodenum,
                   batch_num, ip_timestep, op_timestep, delta_t, ts_idx, model_type)
                        
        hn, cn = [None]*len(mymodel), [None]*len(mymodel)
        # predict
        pred_def = recurrent(i_iter, delta_ts, ip_def, ip_veh, [], hn, cn,
                             mymodel, model_type)
        
    else:  # narx        
        # prepare timestep index 
        ts_idx = [i_iter*stride + 1, i_iter*stride + 1 + op_timestep]
        delta_ts = ts_idx[1] - timestep 
        if delta_ts > 0:
            ts_idx[1] = timestep
            
        ip_def, ip_veh = \
            get_ip(i_iter, ip_def, pred_def, def_dist, tar_nodenum,
                   batch_num, ip_timestep, op_timestep, delta_t, ts_idx, model_type)
                        
        pred_def = recurrent(i_iter, delta_ts, ip_def, ip_veh, [], [], [],
                             mymodel, model_type)
    
    # Denormalization and concatenation
    pred_def_dn = denorm(pred_def, def_dist[0], def_dist[1])
    if i_iter == 0:
        b, n, t = pred_def.shape       
        pred_def_cat = torch.cat((torch.zeros(b, n, 1), pred_def_dn), dim=-1)  # zero response at the 0-th timestep    
    else:
        pred_def_cat = torch.cat((pred_def_cat, pred_def_dn), dim=-1)


### Plot ###   
print('Plot ...')
prediction = pred_def_cat[0, :, :timestep]
target = data_def[0, node_idx[0]:node_idx[1], :timestep]
fig = plot_cm(prediction, target, delta_t)
fig.savefig(os.path.join('plot', f'DisplacementField_Model.{model_type}.png'))


