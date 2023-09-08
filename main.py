# -*- coding: utf-8 -*-
"""
main.py

Repeats RNN training across different initial hidden weight rank and saves 
hidden weight change norm, representation similarity, tangent kernel alignment. 
For the default settings below, main.py should take under 30min to execute. 

"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import neurogym as ngym
import argparse
import os
from file_saver_dumper import save_file, load_file, get_storage_path_reference

## Setup arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_data', default=False, type=bool, help='to save or not to save')
parser.add_argument('--comment', default="", type=str, help='comment for saved fname')
parser.add_argument('--n_iter', default=10000, type=int, help='number of training iter')
parser.add_argument('--print_step', default=100, type=int, help='frequency of saving data')

parser.add_argument('--task_mode', default='ngym', type=str, choices=['ngym', 'sMNIST'], help='ngym or sequential MNIST tasks')
parser.add_argument('--task', default='2AF', type=str, choices=['2AF', 'DMS', 'CXT'], help='task')
parser.add_argument('--learning_rate0', default=0.003, type=float, help='base learning rate')

parser.add_argument('--var_name', default='rr', type=str, choices=['rr'], help='the knob')
parser.add_argument('--W0sig', default=1.25, type=float, help='relevant only if var_name=rr or kap2, std for the starting W init')
parser.add_argument('--hidden_size', default=300, type=int, help='number of hidden units')

args = parser.parse_args()

if args.save_data:
    # Define the flag object as dictionnary for saving purposes
    file_reference, storage_path = get_storage_path_reference(__file__, './results/', comment=args.comment)
    os.makedirs(storage_path, exist_ok=True)
    print('saving data to: ' + storage_path)

### Setup task
task_mode = args.task_mode 

t_mult = 1
if task_mode == 'ngym':
    # Environment  
    batch_size = 32
    seq_len = 100
    if args.task == '2AF':
        task = 'PerceptualDecisionMaking-v0'
        timing = {
            'fixation': 0*t_mult,
            'stimulus': 700*t_mult,
            'delay': 0*t_mult,
            'decision': 100*t_mult}
        seq_len = 8*t_mult
        kwargs = {'dt': 100, 'timing': timing}
        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size,
                              seq_len=seq_len)
    elif args.task == 'DMS':
        task = 'DelayMatchSample-v0'
        seq_len = 8*t_mult
        timing = {
            'fixation': 0*t_mult,
            'sample': 100*t_mult,
            'delay': 500*t_mult,
            'test': 100*t_mult,
            'decision': 100*t_mult}
        kwargs = {'dt': 100, 'timing': timing} 
        
        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size,
                              seq_len=seq_len)

    elif args.task == 'CXT':
        task = 'ContextDecisionMaking-v0'
        seq_len = 8*t_mult
        timing = {
            'fixation': 0*t_mult,
            # 'target': 350,
            'stimulus': 200*t_mult,
            'delay': 500*t_mult,
            'decision': 100*t_mult}
        kwargs = {'dt': 100, 'timing': timing} 
        
        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size,
                              seq_len=seq_len)

    # A sample environment from dataset
    env = dataset.env
    # # Visualize the environment with 2 sample trials
    # _ = ngym.utils.plot_env(env, num_trials=2)

    # Network input and output size
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    dt = env.dt
    
elif task_mode == 'sMNIST':
    batch_size = 200
    seq_len = 28
    
    dt = None
    input_size = 28
    output_size = 10
    
    dataMNIST = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(dataMNIST, [55000, 5000])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    # testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    iterData=iter(train_loader)
    def dataset(): 
        x, y = next(iterData)
        x_ = x[:,0].permute(1,0,2)
        return x_, y


### Setup network
W0_Gauss = args.W0sig*np.random.randn(args.hidden_size, args.hidden_size)/np.sqrt(args.hidden_size)

# Define RNN 
# Code to setup RNN is adapted from https://github.com/gyyang/nn-brain/blob/master/RNN%2BDynamicalSystemAnalysis.ipynb
class CTRNN(nn.Module):
    """Continuous-time RNN.

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        self.input2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(hidden) 
        h_new = torch.relu(hidden * self.oneminusalpha +
                           pre_activation * self.alpha)
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden


class Net(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity


## Train the network
# Configure training and net parameters 
running_loss = 0
running_acc = 0
print_step = args.print_step
hidden_size = args.hidden_size


var_name = args.var_name
if args.var_name == 'rr':
    var_list = [-1, args.hidden_size]
    if args.hidden_size < 150:
        var_list = [1, 30, args.hidden_size]
    elif args.hidden_size > 500:
        var_list = [1, 30, 100, 300, args.hidden_size]
    else:
        var_list = [1, 30, 100, args.hidden_size] 
    if task_mode == 'sMNIST':
        var_list = [5, 30, args.hidden_size] 

lr_list = [args.learning_rate0] #[0.001, 0.003, 0.01] 

criterion = nn.CrossEntropyLoss()

# Loop across Wsig and store results, etc. 
delta_Wr_norm_list = []
all_loss_list = []
sign_sim_list = []
rep_sim_list = []
kernel_alignment_list = []
    
# Initial input, for alignment computation 
if task_mode == 'sMNIST':
    try:
        inputs0, _ = dataset()
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    if task_mode == 'ngym':
        for xx in range(10): # use 10x the batch size for sample
            inputs0_ii, _ = dataset()
            if xx==0:
                inputs0 = inputs0_ii
            else:
                inputs0 = np.concatenate((inputs0, inputs0_ii), axis=1)
    else:
        inputs0, _ = dataset()
    inputs0 = torch.from_numpy(inputs0).type(torch.float)

### Repeat training across different initializations 
for var in var_list:

    print('### ' + var_name + '=' + str(var) + ' ###')

    # Loop through hyperparameters 
    bestLoss = 10. #np.Inf 
    for ii in range(len(lr_list)): 
        # Instantiate the network and print information
        net_ii = Net(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dt=dt)
            
        if args.var_name == 'rr':   
            rr = var
            U_, S_, VT_ = np.linalg.svd(W0_Gauss)
            new_S = S_.copy()
            W0new = U_[:,:rr] @ np.diag(new_S[:rr]) @ VT_[:rr, :]
            
            # normalize
            W0new = W0new / np.linalg.norm(W0new) * np.linalg.norm(W0_Gauss) # by norm                            
            net_ii.rnn.h2h.weight.data.copy_(torch.from_numpy(W0new).type(torch.float))
                                    
        # Optimizer 
        n_iter = args.n_iter
        optimizer = optim.SGD(net_ii.parameters(), lr=lr_list[ii], momentum=0.9) # default

        ## Storing initial stuff
        Wr_0 = net_ii.rnn.h2h.weight.detach().numpy().copy()                  
            
        output0, activity0 = net_ii(inputs0)
        if task_mode == 'ngym':
            activity0_ = activity0[env.start_ind['decision']:env.end_ind['decision']].detach().numpy().copy()
        else:
            activity0_ = activity0.detach().numpy().copy()

        # Get K0
        if (task_mode == 'ngym'): 
            for t in range(seq_len): 
                for b in range(batch_size):
                    for k in range(output_size): # output is the inner dim
                        df_1 = torch.unsqueeze(torch.autograd.grad(output0[t,b,k], \
                                                    net_ii.rnn.h2h.weight, retain_graph=True)[0], dim=0) # focus on recurrent weight
                        if (b==0) and (k==0) and (t==0):
                           df = df_1
                        else:
                            df = torch.cat((df, df_1), dim=0)
        elif (task_mode == 'sMNIST'): # output at last step only
            for b in range(batch_size):
                for k in range(output_size):
                    df_1 = torch.unsqueeze(torch.autograd.grad(output0[-1,b,k], net_ii.rnn.h2h.weight, retain_graph=True)[0], dim=0)
                    if (b==0) and (k==0):
                        df = df_1
                    else:
                        df = torch.cat((df, df_1), dim=0)
        else:
            raise NotImplementedError("Per sample grad not implemented for the task")     
        K0 = torch.einsum('bij,aij->ba', df, df)

        ### start training ###
        loss_list = []
        iter_list = []
        for i in range(n_iter):
            try:
                inputs, labels_ = dataset()            
            except StopIteration:
                iterData=iter(train_loader)
                inputs, labels_ = dataset() 
                
            if task_mode != 'sMNIST':
                inputs = torch.from_numpy(inputs).type(torch.float)

            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers            
            
            def get_loss_out(net): 
                output, activity = net(inputs)
                
                if task_mode == 'ngym':
                    labels = torch.from_numpy(labels_.flatten()).type(torch.long)
                    output = output.view(-1, output_size)     
                    loss = criterion(output, labels)
                else:     
                    labels = labels_
                    loss = criterion(output[-1,:,:], labels) # loss at last
                    
                return loss, output, labels, activity
            
            loss, output, labels, activity = get_loss_out(net_ii)
            loss.backward()
            optimizer.step()    # Does the update
            running_loss += loss.item()

            if i % print_step == 0: # (print_step - 1):
                if i != 0:
                    running_loss /= print_step
                #print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
                loss_list.append(running_loss)
                iter_list.append(i)
                running_loss = 0
        ### End training ###

        avLoss = np.mean(np.array(loss_list[-20:])) # average of last 20 sampled loss values 
        if avLoss < bestLoss: # if this loss is better, save results
            bestLoss = avLoss
            print('best LR so far, ' + str(lr_list[ii]))
            net = net_ii

            if ii > 0: # replace the last item, if this loss is better and not the first hyperparam tried
                all_loss_list = all_loss_list[:-1]
                delta_Wr_norm_list = delta_Wr_norm_list[:-1]
                sign_sim_list = sign_sim_list[:-1]
                rep_sim_list = rep_sim_list[:-1]
                kernel_alignment_list = kernel_alignment_list[:-1]
        
            # View weights
            Wr = net.rnn.h2h.weight.detach().numpy()                
            delta_Wr_norm_list.append(np.linalg.norm(Wr-Wr_0))   
            
            all_loss_list.append(loss_list)

            ### get the linearity measures ###
            output, activity_ = net(inputs0)
            if task_mode == 'ngym':
                activity_ = activity_[env.start_ind['decision']:env.end_ind['decision']].detach().numpy().copy()
            else:
                activity_ = activity_.detach().numpy().copy()

            # Get sign similarity
            sign_sim = np.sum(np.sign(activity_)==np.sign(activity0_))/activity_.size
            sign_sim_list.append(sign_sim)
                        
            # Get rep similarity, based on hidden activity at decision time 
            KR0 = activity0_[-1,:,:] @ activity0_[-1,:,:].T # (b,j) @ (j,b) -> (b,b)
            KR = activity_[-1,:,:] @ activity_[-1,:,:].T # (b,j) @ (j,b) -> (b,b)
            rep_sim = np.sum(KR*KR0) / np.linalg.norm(KR0) / np.linalg.norm(KR)
            rep_sim_list.append(rep_sim)
                        
            # Get Kf
            if (task_mode == 'ngym'):
                for t in range(seq_len): 
                    for b in range(batch_size):
                        for k in range(output_size): # output is the inner dim
                            df_1 = torch.unsqueeze(torch.autograd.grad(output[t,b,k], \
                                                        net.rnn.h2h.weight, retain_graph=True)[0], dim=0)
                            if (b==0) and (k==0) and (t==0):
                                df = df_1
                            else:
                                df = torch.cat((df, df_1), dim=0)
            elif (task_mode == 'sMNIST'): # output at last only 
                for b in range(batch_size):
                    for k in range(output_size): # output is the inner dim
                        df_1 = torch.unsqueeze(torch.autograd.grad(output[-1,b,k], \
                                                    net.rnn.h2h.weight, retain_graph=True)[0], dim=0)
                        if (b==0) and (k==0):
                            df = df_1
                        else:
                            df = torch.cat((df, df_1), dim=0)
            else:
                raise NotImplementedError("Per sample grad not implemented for the task")
            Kf = torch.einsum('bij,aij->ba', df, df)
            kernel_alignment = torch.sum(Kf*K0) / torch.norm(Kf) / torch.norm(K0)            
            kernel_alignment_list.append(kernel_alignment.detach().numpy().copy())
        

if args.save_data:
    results = {
        'var_list': var_list,
        'iter_list': iter_list,
        'delta_Wr_norm_list': delta_Wr_norm_list,
        'all_loss_list': all_loss_list,
        'sign_sim_list': sign_sim_list,
        'rep_sim_list': rep_sim_list,
        'kernel_alignment_list': kernel_alignment_list,
    }
    
    save_file(results, storage_path, 'results', file_type='json')
    
    args_dict = vars(args)
    save_file(args_dict, storage_path, 'args_dict', file_type='json')
