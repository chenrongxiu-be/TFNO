# -*- coding: utf-8 -*-
"""
Title: Real-time surrogate model of vehicle-bridge interaction
Author: Chen Rongxiu
Created: 24-06-24
Updated: 25-09-12
License: MIT License

Description:
------
Train NARX model

Dependencies:
-------------
numpy, pytorch, einops

Notes:
------
- Make sure to install all dependencies before running.
- The code assumes input data are saved in the "data" folder as described in the README.

"""

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
import os, json, configparser, sys, shutil
from timeit import default_timer

#-------main-------
def main():
    ### Initial settings ### 
    torch.set_default_dtype(torch.float64)
    
    # config
    configfile = os.path.join('config', 'config_narx.ini')
    cf = configparser.ConfigParser()
    try:
        cf.read(configfile, encoding='utf-8')
    except:
        cf.read(configfile, encoding='shift-jis')
    
    title = cf.get('main','title') 
    print(f'The title of training: {title} \n')    
    model_path = os.path.join('model', title) 
    if os.path.exists(model_path): 
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    
    config_path = os.path.join('config', f'config_narx_{title}.ini')
    if os.path.exists(config_path): 
        os.remove(config_path)
    shutil.copyfile(configfile, config_path)  # backup of config file
    
    # data
    delta_t = cf.getfloat('data', 'delta_t')     
    ip_timestep = cf.getint('data','ip_timestep')
    op_timestep = cf.getint('data','op_timestep')    
    bridge = cf.get('data', 'bridge')     
    # choose the following (1) ~ (3) configurations according to the target data,
    # but do not alter them.
    # (1) for the continuous bridge deck
    node_num = 45
    target_datasets = [1, 2]
    target_node = [12, 22]
    target_axis = 'z'
    # (2) for the cable-stayed bridge deck
    # node_num = 160
    # target_datasets = [1, 2]
    # target_node = [1, 46]
    # target_axis = 'z'
    # (3) for the cable-stayed bridge pylon
    # node_num = 160
    # target_datasets = [1, 2]
    # target_node = [103, 122]
    # target_axis = 'x'
    
    # model
    hidden_size = cf.getint('model','hidden_size')
    num_layers = cf.getint('model','num_layers')
    
    # train        
    batchsize_train = cf.getint('train','batchsize_train')     
    epoch_num = cf.getint('train','epoch_num')     
    adam_lr = cf.getfloat('train','adam_lr')
    adam_weight_decay = cf.getfloat('train','adam_weight_decay')       
    scheduler_step = cf.getint('train','scheduler_step')
    scheduler_gamma = cf.getfloat('train','scheduler_gamma')
    
    # loss
    w_dataloss = cf.getfloat('loss','w_dataloss')
    w_freqloss = cf.getfloat('loss','w_freqloss')
    loss_weight = [w_dataloss, w_freqloss]
    
    
    ### CUDA ###
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    torch.cuda.empty_cache() 
    
    
    ### Load data ###
    print('Read data...')
    DataClass = Data(bridge, node_num, target_datasets[-1], target_node, target_axis,
                     batchsize_train, device)
    for i_dsno in target_datasets:
        temp_train_loader, nodegrid, def_dist, node_idx = DataClass.load_data(i_dsno)   
        if 'train_loaders' not in locals():
            train_loaders = [temp_train_loader]
        else:
            train_loaders += [temp_train_loader]
    tar_nodenum = node_idx[1] - node_idx[0]
    
    
    for model_idx in range(node_idx[1]-node_idx[0]):
        # Unlike neural operators, each NARX is trained for a single node only.
        
        ### Build model ###   
        print(f'Build model ({model_idx})...')
        inputsize = 29 + ip_timestep # input = [veh_para, y(t-1),...,y(t-ip_timestep-1)]
        output_size = op_timestep
        mymodel = Build_model(inputsize, output_size, hidden_size, num_layers).to(device)
        mymodel.apply(weights_init) 
        
        ### Train model ###
        optimizer = torch.optim.Adam(mymodel.parameters(), lr=adam_lr, weight_decay=adam_weight_decay)    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)  
        
        print(f'Start training the model ({model_idx}):')
        trainer = Trainer(model_path, device,
                          ip_timestep, op_timestep, delta_t,
                          def_dist, node_idx,
                          loss_weight)
        trainer.train(model_idx, mymodel, epoch_num, train_loaders, optimizer, scheduler, batchsize_train)
        
    
    
#-------data-------
class Data():
    def __init__(self, bridge, node_num, maxmin_dataset, target_node, target_axis,
                 batchsize_train, device):
        self.bridge = bridge
        self.node_num = node_num
        temp = "{:03d}".format(maxmin_dataset)
        self.path_maxmin = os.path.join('data', bridge, f'dataset_{temp}')
        self.tar_node = target_node
        self.tar_axis = target_axis
        self.bs_train = batchsize_train  
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
        
        temp = np.load(os.path.join(path, 'traindata_def.npy'), allow_pickle=True)
        traindata_def = torch.from_numpy(temp)
        temp = np.load(os.path.join(path, 'traindata_veh.npy'), allow_pickle=True)
        traindata_veh = torch.from_numpy(temp)                
        temp = np.load(os.path.join(path, 'traindata_timesize.npy'), allow_pickle=True) 
        traindata_timesize = torch.from_numpy(temp)        
        temp = np.load(os.path.join(path, 'traindata_caselabel.npy'), allow_pickle=True) 
        traindata_caselabel = temp.tolist()
                
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
        traindata_def_tarnode_norm, def_dist = \
            self.norm_init_data(traindata_def[:, node_idx[0]:node_idx[1], :], 'def', def_maxmin)
        traindata_veh, _ = self.norm_init_data(traindata_veh, 'veh', veh_maxmin)    
        
        # build dataloaders
        train_loader = torch.utils.data.DataLoader(CustomDataset( 
                           traindata_caselabel,
                           traindata_def, traindata_def_tarnode_norm, 
                           traindata_veh, 
                           traindata_timesize), 
                           batch_size=self.bs_train, shuffle=True, generator=torch.Generator(self.device))
        
        return train_loader, data_nodegrid, def_dist, node_idx
        
    def norm_init_data(self, traindata, data_type, data_minmax):
        a, b = 0, 1
        
        if data_type == 'grid':
            data_max, _ = torch.max(traindata, dim=0)
            data_min, _ = torch.min(traindata, dim=0)
            
            for i in range(2):
                traindata[:, i] = (b - a) * (traindata[:, i]- data_min[i]) / \
                                  (data_max[i] - data_min[i] + 1e-8) + a
            return traindata
            
        elif data_type == 'def':
            data_min, data_max = data_minmax[0], data_minmax[1]
            normed_data = (b - a) * (traindata - data_min) / (data_max - data_min + 1e-8) + a 
            
            data_dist = [data_min, data_max]
            
            return normed_data, data_dist
        
        elif data_type == 'veh':
            data_min = repeat(data_minmax[:, 0], 'p -> a k p b', a=1, k=1, b=1) 
            data_max = repeat(data_minmax[:, 1], 'p -> a k p b', a=1, k=1, b=1) 
            
            temp_shape = traindata.shape
            data_min_ = data_min.expand(temp_shape[0], temp_shape[1], data_min.shape[2], temp_shape[3]) 
            data_max_ = data_max.expand(temp_shape[0], temp_shape[1], data_max.shape[2], temp_shape[3])
            normed_data = (b - a) * (traindata - data_min_) / (data_max_ - data_min_ + 1e-8) + a
                        
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
            
    
#-------model-------
class Build_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Build_model, self).__init__()
        
        '''
        NARX. 
        
        Input
        -----
        x_def: previous deflection time series, size=(batch, timestep)
        x_veh: vehicle parameters, size=(batch, para_num)
        
        Output
        ------
        out: future deflection time series, size=(batch, timestep)
        
        '''
        # NARX 
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
        ])
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
                
    def forward(self, x_def, x_veh):     
        x = torch.cat((x_def, x_veh), dim=1)  
        
        x = self.activation(self.fc0(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        next_y = self.fc_out(x)
        return next_y
        
        
        
#-------train-------   
def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if not (m.bias is None): nn.init.zeros_(m.bias)
        
        
class Trainer:
    def __init__(self, model_path, device,
                 ip_timestep, op_timestep, delta_t,
                 def_dist, node_idx, 
                 loss_weight,
                 lowest_loss=100, 
                 list_trainloss=None, 
                 start_ep=0, 
                 list_lr=[],
                 list_epoch_time_train=[]):
        
        self.model_path = model_path
        self.device = device 
        
        self.ip_timestep = ip_timestep
        self.op_timestep = op_timestep
        self.delta_t = delta_t
        self.def_dist = def_dist
        self.node_idx = node_idx
        
        self.loss_weight = loss_weight
        
        self.lowest_loss = lowest_loss         
        self.list_trainloss = list_trainloss
        
        self.start_ep = start_ep
        self.list_lr = list_lr
        
        self.list_epoch_time_train = list_epoch_time_train
        
    def train(self, model_idx, model, epoch_num, train_loaders, optimizer, scheduler, batchsize_train):        
        n_train_sam, n_train_batch = 0, 0
        for _, train_loader in enumerate(train_loaders):
            n_train_sam += len(train_loader.dataset)
            temp = n_train_sam // batchsize_train + 1 if (len(train_loader.dataset)%batchsize_train) != 0 else len(train_loader.dataset) // batchsize_train
            n_train_batch += temp
                        
        sys.stdout.flush()
        
        t1 = default_timer()
        for ep in range(epoch_num):
            # training loop
            print(f'\nEpoch: {self.start_ep+ep+1}th/{self.start_ep+epoch_num}')
            print('# Training:')
            model.train()
            t2 = default_timer()
            
            train_loss = self.batch_loop(model_idx, model, train_loaders, optimizer, self.start_ep+ep+1, n_train_batch)
            
            epoch_time_train = default_timer() - t2
            self.list_epoch_time_train += [epoch_time_train]
            print("-Training of {:4d}th epoch costed {:0.2f} sec.".format(self.start_ep+ep+1, epoch_time_train))
            
            ave_trainloss = torch.mean(train_loss, dim=0, keepdim=True)
            if self.list_trainloss is None:
                self.list_trainloss = train_loss.cpu()
            else:
                self.list_trainloss = torch.cat((self.list_trainloss, train_loss.cpu()), dim=0)            
            print(f' Average train_loss: {ave_trainloss}. \n')                        
            del train_loss, ave_trainloss
            
            # log the learning rate
            self.list_lr.append(scheduler._last_lr)
            
            # step scheduler
            scheduler.step()            
            
            epoch_time = default_timer() - t2
            print('# Summary:')
            print("-{:4d}th epoch costed {:0.2f} sec.".format(self.start_ep+ep+1, epoch_time)) 
            print('========================================================= \n')
        
        # save model
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(self.model_path, 'model.pth')) 
                   
        # print training info 
        total_time = default_timer() - t1
        time_per_epoch = round(total_time / epoch_num, 6)
        print(f'\nAll {self.start_ep+epoch_num} epochs finished.')
        print('The entire training process costed {:0.2f} sec in total.'.format(total_time))
        print(' ({:0.2f} sec per epoch)'.format(time_per_epoch))
            
                    
    def batch_loop(self, model_idx, model, dataloaders, optimizer, ep, n_batch):
        loss_log = None
        count_batch = 0
        for _, dataloader in enumerate(dataloaders):
            for i_batch, \
                (data_caselabel, \
                 data_def, data_def_tarnode_norm, \
                 data_veh, \
                 data_timesize) in enumerate(dataloader):
                    
                count_batch += 1  
                data_def, data_def_tarnode_norm, data_veh = \
                    data_def.to(self.device), data_def_tarnode_norm.to(self.device), data_veh.to(self.device)
                
                # get maximun timesize of the batch
                max_timesize = torch.max(data_timesize).item()
                
                # recurrent  
                stride = self.op_timestep  
                pred_iters = int((max_timesize-1) / stride) + 1
                ip_def, pred_def, ts_idx = None, None, None
                for i_iter in range(pred_iters):                         
                    # prepare timestep index 
                    ts_idx = [i_iter*stride + 1, i_iter*stride + 1 + self.op_timestep]
                    delta_ts = ts_idx[1] - max_timesize 
                    if delta_ts > 0:
                        ts_idx[1] = max_timesize
                        if delta_ts > (self.op_timestep * 9/10): break 
                        
                    # get input data
                    ip_def, ip_veh = self.get_ip(i_iter, ip_def, pred_def, stride, model_idx, 
                                                 data_def.shape[0], 
                                                 data_def_tarnode_norm, data_veh, ts_idx)
                    
                    # predict
                    pred_def = model(ip_def, ip_veh)
                    
                    # if iteration exceeds maximum timestep, truncate redundant part
                    if delta_ts > 0:
                        pred_def_ = pred_def[..., :-delta_ts] 
                    else:
                        pred_def_ = pred_def
                    
                    # generate mask
                    keep_mask = torch.ones_like(pred_def_).bool()
                    batch = pred_def_.shape[0]
                    for i_b in range(batch):
                        if ts_idx[1] > data_timesize[i_b]: 
                            delta_ts = ts_idx[1] - data_timesize[i_b]
                            if delta_ts >= self.op_timestep:
                                keep_mask[i_b, :] = False
                            else:
                                keep_mask[i_b, -delta_ts:] = False
               
                    # get loss        
                    weighted_sum, loss_log = get_loss(ts_idx, data_timesize, pred_def_, keep_mask,
                                                      data_def[:, self.node_idx[0]+model_idx, :max_timesize], 
                                                      data_def_tarnode_norm[:, model_idx, :max_timesize],
                                                      self.loss_weight, loss_log) 
                    # print(f'-{count_batch}th batch/{n_batch}, {i_iter+1}th iter/{pred_iters}:')
                    # print(f' Loss: {loss_log[-1]}')  
                    
                    # update model parameters
                    weighted_sum = weighted_sum * (pred_iters - i_iter)  # respect causality
                    optimizer.zero_grad()
                    weighted_sum.backward()
                    optimizer.step()
                        
        return loss_log
    
    def get_ip(self, i_iter, ip_def, pred_def, stride, model_idx,  
               batch_num, 
               data_def_tarnode_norm, data_veh, ts_idx):
        if i_iter == 0: 
            # initialize input
            a, b = 0, 1
            min, max = self.def_dist[0], self.def_dist[1]
            ini_def = (b - a) * (0 - min) / (max - min) + a
            ini_def = ini_def.to(self.device)
            
            noise = 1. / 1e6 * torch.randn(batch_num, self.ip_timestep) 
            ip_def = noise + ini_def * torch.ones(batch_num, self.ip_timestep)
            
            ip_veh = data_veh[:, 0, :, 0]
                        
        else:  # i_iter >= 1
            if self.ip_timestep > stride:  
                ip_def = torch.cat((ip_def[..., stride: ], 
                                    data_def_tarnode_norm[:, model_idx, ts_idx[0]-self.ip_timestep:ts_idx[0]-self.ip_timestep+stride]), dim=-1)
            else:
                lower_bound = (stride - self.ip_timestep - 1) + ts_idx[0] - self.ip_timestep
                upper_bound = lower_bound + self.ip_timestep
                ip_def = data_def_tarnode_norm[:, model_idx, lower_bound:upper_bound]
            
            ip_veh = data_veh[:, 0, :, ts_idx[0]-1]
                            
        return ip_def, ip_veh
        
    
def get_loss(ts_idx, data_timesize, pred_def, keep_mask,
             data_def, data_def_tarnode_norm,
             loss_weight, loss_log):
                
    def total_variance(x, mask):  
        return torch.sum(torch.abs(x[..., 1:] * mask[..., 1:] - x[..., :-1] * mask[..., :-1])) / (mask[..., 1:].sum() + 1e-8)
    
    lossfn = nn.MSELoss(reduction='none')
    
    # data loss
    dataloss = lossfn(pred_def * keep_mask, data_def_tarnode_norm[..., ts_idx[0]:ts_idx[1]] * keep_mask)
    dataloss = dataloss.sum() / (keep_mask.sum() + 1e-8) 
              
    # frequency loss
    tar_fft_def = torch.abs(torch.fft.rfft(data_def_tarnode_norm[..., ts_idx[0]:ts_idx[1]].clone() * keep_mask, dim=-1, norm='forward'))
    pred_fft_def = torch.abs(torch.fft.rfft(pred_def * keep_mask, dim=-1, norm='forward'))
    freqloss_def = lossfn(pred_fft_def, tar_fft_def)
    freqloss_def = freqloss_def.sum() / (keep_mask[..., :keep_mask.shape[-1]//2+1].sum() + 1e-8)
    
    # we additionally add a loss defined at the beginning and ending timesteps of each iteration,
    #  such that the boundary between each two iterations would be smoother.
    bound_limit = 16 
    if bound_limit > keep_mask.shape[-1]: bound_limit = keep_mask.shape[-1] // 2
    bcloss_def = lossfn(torch.cat((pred_def[..., :bound_limit] * keep_mask[..., :bound_limit], 
                                   pred_def[..., -bound_limit:] * keep_mask[..., -bound_limit:]), dim=-1), 
                        torch.cat((data_def_tarnode_norm[..., ts_idx[0]:ts_idx[0]+bound_limit] * keep_mask[..., :bound_limit], 
                                   data_def_tarnode_norm[..., ts_idx[1]-bound_limit:ts_idx[1]] * keep_mask[..., -bound_limit:]), dim=-1))
    bcloss_def = bcloss_def.sum() / (keep_mask[..., :bound_limit].sum() + keep_mask[..., -bound_limit:].sum() + 1e-8)
        
    # similarly, we add a loss defined at the first timestep of each iteration
    icloss_def = lossfn(pred_def[..., 0] * keep_mask[..., 0], 
                        data_def_tarnode_norm[..., ts_idx[0]-1] * keep_mask[..., 0])
    icloss_def = icloss_def.sum() / (keep_mask[..., 0].sum() + 1e-8)
    
    # total variance loss
    tvloss_def = total_variance(pred_def, keep_mask)
            
    # weighted sum for backpropogation
    weighted_sum = loss_weight[0] * dataloss + loss_weight[1] * freqloss_def + \
                   0.02 * tvloss_def + bcloss_def + icloss_def
                   
    # log the loss
    temp = torch.tensor([dataloss.item()])
    if loss_log == None:
        loss_log = temp
    else:
        loss_log = torch.cat((loss_log, temp))
        
    return weighted_sum, loss_log




#---------------------
if __name__ == '__main__':
    main()
    print('All done!')


