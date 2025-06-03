#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819

import sys
import copy
from os import cpu_count

import numpy as np
import torch
import os
import json
from torch import nn
import matplotlib.pyplot as plt
from .transformer_utils import generate_square_subsequent_mask
sys.path.append("..")
from find_trace import *
from generate_STL import generate_property, generate_property_test, get_robustness_score




def property_loss_simp(y_pred, property, loss_function):
    return torch.sum(loss_function(y_pred - property))


def find_group_info(groups, index):
    # Loop through each group in the dictionary
    for group_id, group_info in groups.items():
        # print(1)
        # Check if the index is in the clients list of the current group
        if index in group_info['clients']:
            # If found, return the cp_value list of this group
            return group_info['cp_value']
    # If the index is not found in any group, return None or an appropriate message

    print(f"ATTENTION: NO CORRESPONDING CP VALUE FOUND.")
    return None

def cp_guarantee_ratio_calculation(y_batch, cp_value):

    length_batch = len(y_batch)
    att_length = len(y_batch[0])
    assert length_batch == 100
    assert att_length == 24

    attribute_counts = [0] * 24  # List of 24 zeros

    # Total number of elements
    total_elements = len(y_batch)

    # print(y_batch)
    # print(cp_value)

    # Iterate over each element
    for element in y_batch:
        # Iterate over each attribute index (0 to 23)
        for idx in range(att_length):
            # Check if the attribute satisfies the condition (greater than 1)
            if element[idx] > np.sqrt(cp_value[idx]):
                # Increment the count for this attribute
                attribute_counts[idx] += 1

    # print(attribute_counts)
    # exit(0)
    # Calculate the ratio for each attribute
    attribute_ratios = [count / total_elements for count in attribute_counts]

    # Display the ratios
    # for idx, ratio in enumerate(attribute_ratios):
        # print(f"Attribute {idx+1}: {ratio:.2f}")

    sat_rate = 0
    fail_rate = 0

    for i in range(length_batch):
        for j in range(att_length):
            # print(y_batch[i])
            # print(len(y_batch[i]))
            # print(cp_value)
            # print(len(cp_value))
            # print(cp_value[j])
            # exit(0)
            if np.abs(y_batch[i][j]) <= np.abs(np.sqrt(cp_value[j])):
                sat_rate += 1
            else:
                fail_rate += 1
    
    # print(f"sat rate is {sat_rate}")
    # print(f"fail rate is {fail_rate}")
    # print(f"total for check is {sat_rate + fail_rate}")
    
    cp_rate = sat_rate / (sat_rate + fail_rate)

    # print(f"cp rate is {cp_rate}")

    return sat_rate, fail_rate, cp_rate



def dic_loader(args):

    if args.sep_type == "spec_m":
        cp_pat = f"hdd/cluster_cp_result_with_specm_{args.model}_{args.cp_epoch}_{args.client}/LogiCP_Cluster.json"

        with open(cp_pat, 'r') as file:
            cp_dic = json.load(file)
    
    if args.sep_type == "cp":
        cp_pat = "hdd/cluster_cp_result_with_cp/LogiCP_Cluster.json"

        with open(cp_pat, 'r') as file:
            cp_dic = json.load(file)

    return cp_dic

def dic_loader_without_specm(args):

    if args.sep_type == "spec_m":
        cp_pat = f"hdd_{args.method}/cluster_cp_result_with_spem/{args.method}_Cluster.json"

        with open(cp_pat, 'r') as file:
            cp_dic = json.load(file)
    
    if args.sep_type == "cp":
        cp_pat = f"hdd_{args.method}/cluster_cp_result_with_cp/{args.method}_Cluster.json"

        with open(cp_pat, 'r') as file:
            cp_dic = json.load(file)
    
    if args.sep_type == "value":
        cp_pat = f"hdd_{args.method}/cluster_cp_result_with_value/{args.method}_Cluster.json"

        with open(cp_pat, 'r') as file:
            cp_dic = json.load(file)

    return cp_dic

def new_loss_func(cp_value, y_gt, y_pred):

    """
    
    If the point-wise loss is greater than its corresponding cp region, 
    the loss will be calibrated to the size of cp region, 
    since we assume the gt will appear in the cp region, 
    so we don't want to gradient the model to predict outside of cp region. 
    
    """

    # print(f"new loss function applied.")

    att_len = len(cp_value)
    assert att_len == y_gt.shape[1]

    # temp = (y_pred - y_gt) ** 2
    temp = abs(y_pred - y_gt)
    # print(f"the temp looks like {len(temp)}")
    # print(f"the cp value looks like {cp_value}")

    loss = []
    data_loss = []
    
    for i in range(len(temp)):
        data_loss = []
        for j in range(len(temp[i])):
            if temp[i][j] >= cp_value[j]:
                att_loss = cp_value[j]
            else:
                att_loss = temp[i][j].item()
            # att_loss = temp[i][j] - cp_value[j]
            # relu_loss = m(att_loss)
            data_loss.append(att_loss)
        
        loss.append(data_loss)

    # print(f"the new loss looks like {len(loss)}")
    # print(f"the new loss looks like {len(loss[0])}")

    return loss

def property_loss_eventually(y_pred, property, loss_function, type):
    iterval = 2
    if type == "eventually-upper":
        diff_yp = y_pred - property
        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
        diff_min, ind = torch.min(loss_function(unsqueezed_diff), dim=2)
        return torch.sum(diff_min)
    elif type == "eventually-lower":
        diff_yp = property - y_pred
        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
        diff_min, ind = torch.min(loss_function(unsqueezed_diff), dim=2)
        return torch.sum(diff_min)



def property_loss(X, y_pred, property_by_station_day, loss_function, y, show):
    loss = torch.zeros(1).to('cuda')
    for ind, arr in enumerate(X):   # X shape: 64, 120, 3
        sensor = int(arr[-1,2].item())
        day = int(arr[-1,1].item())
        day = (day+1)%7
        loss += torch.sum(loss_function(y_pred[ind] - torch.tensor(property_by_station_day[sensor][day][0]).to('cuda')))
        loss += torch.sum(loss_function(torch.tensor(property_by_station_day[sensor][day][1]).to('cuda') - y_pred[ind]))
    if show:
        plt.plot(property_by_station_day[sensor][day][0])
        plt.plot(property_by_station_day[sensor][day][1])
        plt.plot(y[ind].detach().cpu())
        plt.plot(y_pred[ind].detach().cpu())
        plt.show()
    return loss



def repackage_hidden(h):

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)




def transformer_prox_train(dataloader, net, args, loss_func, lr, server_model=None):
    
    net.train()
    mu = 1

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter  # local update epochs
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)

    epoch_loss = []
    for it in range(local_eps):
        num_updates = 0
        
        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        batch_loss = []
        for batch_idx, (X, y) in enumerate(dataloader): ## batch first=True
            w_0 = copy.deepcopy(net.state_dict())
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            if it>0:
                w_diff = torch.tensor(0., device=device)
                for w, w_t in zip(server_model.parameters(), net.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)
                loss += mu / 2. * w_diff

            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)




def transformer_ditto_train(dataloader, net, args, loss_func, lr, w_ditto=None, lam=1):
    
    net.train()

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter  # local update epochs
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0
        
        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        batch_loss = []
        for batch_idx, (X, y) in enumerate(dataloader): ## batch first=True
            w_0 = copy.deepcopy(net.state_dict())
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            if w_ditto is not None:
                w_net = copy.deepcopy(net.state_dict())
                for key in w_net.keys():
                    w_net[key] = w_net[key] - lr*lam*(w_0[key]-w_ditto[key])
                net.load_state_dict(w_net)
                optimizer.zero_grad()

            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)

def transformer_prop_pretrain(dataloader, net, args, loss_func, lr, w_glob_keys=None, cp_value=None):

    # print()
    # print(f"the pretrain function for TRANSFORMER is applied.")
    # print()

    # loss function will be changed to calculate the loss for each attribute in y. (24)
    loss_func = nn.MSELoss(reduction='mean')

    # batch size of trainset loader is 64.
    net.batch_size = 64

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.9)
    local_eps_pretrain = args.pretrain_iter
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)

    epoch_loss = []

    for iter in range(local_eps_pretrain):

        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        num_updates = 0
        batch_loss = []

        for X, y in dataloader: ## batch first=True
            X = X.to("cuda")
            y = y.to("cuda")
            src_mask = src_mask.to('cuda')
            tgt_mask = tgt_mask.to('cuda')
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            # print(output.shape)

            # print(output.view(-1, 24).shape) # [64, 24]
            loss = loss_func(output.view(-1, 24), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == local_eps_pretrain: 
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)

def transformer_prop_calib_norm(dataloader, net, args, loss_func, lr, w_glob_keys=None, cp_value=None):

    # print()
    # print(f"the normalization calibration function for TRANSFORMER is applied.")
    # print()
    
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)

    net.eval()

    for X, y in dataloader: ## batch first=True
        X = X.to("cuda")
        y = y.to("cuda")
        src_mask = src_mask.to('cuda')
        tgt_mask = tgt_mask.to('cuda')
        X = X.unsqueeze(2)
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
        # print(output.shape)

        # print(output.view(-1, 24).shape) # [64, 24]
        abs_diff = abs(output.view(-1, 24) - y) # find the distance between pred and gt
        # print(abs_diff.shape)

        # exit(0)
        result_list = abs_diff.tolist() # result in a list with 24 elements, each element is a list with 64 differences. 
        max_values = np.max(result_list, axis=0)

    return max_values

def transformer_prop_calib(dataloader, net, args, loss_func, lr, w_glob_keys=None, max_values = None):

    # print()
    # print(f"the CALIBRATION function for TRANSFORMER is applied.")
    # print()

    max_values = np.array(max_values, dtype=np.float32) 
    # print(f"max_values check is {max_values}")

    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)

    net.eval()
    for X, y in dataloader:
        X = X.to("cuda")
        y = y.to("cuda")
        src_mask = src_mask.to('cuda')
        tgt_mask = tgt_mask.to('cuda')
        X = X.unsqueeze(2)
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
        # print(output)
        # print(output.shape)

        abs_diff = abs(output.view(-1, 24) - y).cpu().detach().numpy() # find the distance between pred and gt
        # print(max_values)
        # print(abs_diff.shape)
        # print(abs_diff[:2])
        # print(len(abs_diff))
        abs_diff = np.array(abs_diff, dtype=np.float32)
        # print(abs_diff[:2])  
        # print(len(abs_diff[0]))
        # print(len(abs_diff))
        # exit(0)
        # print(f'unnormalied diff is {abs_diff}')

        for i in range(len(abs_diff)):
            abs_diff[i] /= max_values
        
        max_value_of_row = np.max(abs_diff, axis=1)
        c_tuda_list = max_value_of_row
        
    return c_tuda_list


def transformer_prop_train(dataloader, net, args, loss_func, lr, w_glob_keys=None, cp_value=None):

    # loss function will be changed to calculate the loss for each attribute in y. (24)
    loss_func_cp = new_loss_func
    loss_func = nn.MSELoss(reduction='mean')

    # batch size of trainset loader is 64
    net.batch_size = 64

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.9)
    local_eps = args.client_iter
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)

    epoch_loss = []
    epoch_cons_loss = []

    for iter in range(local_eps):
        num_updates = 0

        for name, param in net.named_parameters():
            param.requires_grad = True
        # scaled_dot_product_attention
        batch_loss = []
        batch_cons_loss = []
        for X, y in dataloader: ## batch first=True
            X = X.to("cuda")
            y = y.to("cuda")
            src_mask = src_mask.to('cuda')
            tgt_mask = tgt_mask.to('cuda')
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            # src_mask = src_mask.to('cuda')
            # tgt_mask = tgt_mask.to('cuda')
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            # print(output.shape)
            # print(output.view(-1, 24).shape) # [64, 24]
            pred_loss = loss_func(output.view(-1, 24), y)
            pred_loss_cp = loss_func_cp(cp_value = cp_value, y_pred = output.view(-1, 24), y_gt = y)
            # detach()

            subloss = [loss for sublist in pred_loss_cp for loss in sublist]
            # print(len(subloss)) # 1536 = 24 * 64, the abs loss, which need to take power 2 in next line to calculate MSE. 
            cp_loss = np.mean([loss**2 for loss in subloss])

            if args.property_type == 'constraint':
                property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)

            loss = pred_loss + cons_loss + cp_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            num_updates += 1
            batch_loss.append(loss.item())
            batch_cons_loss.append(cons_loss.item())
            
            if num_updates == args.local_updates: 
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)


def transformer_prop_teacher_test(dataloader, net, args, loss_func, rho=False):
    net.eval()
    m = nn.ReLU()
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)
        
    for name, param in net.named_parameters():
        param.requires_grad = False 

    batch_rho = []
    batch_loss = []
    batch_cons_loss = []
    for X, y in dataloader:
        X = X.to("cuda")
        y = y.to("cuda")
        src_mask = src_mask.to('cuda')
        tgt_mask = tgt_mask.to('cuda')
        net.eval()
        X = X.unsqueeze(2)
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
        output = output.view(-1, 24)
    
        if args.property_type == 'constraint':
            property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
            corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
            property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
            corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
            teacher_pred = torch.min(output, corrected_trace_upper)
            teacher_pred = torch.max(output, corrected_trace_lower)
            output = teacher_pred
            cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
        elif args.property_type == 'eventually':
            property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
            property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
            cons_loss = property_loss_eventually(output, property_upper, m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
        else:
            raise NotImplementedError
        pred_loss = loss_func(output, y)
        batch_loss.append(pred_loss.item())
        batch_cons_loss.append(cons_loss.item())

        if rho==True:
            if args.property_type == 'constraint':
                batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_upper - output))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
                batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_lower))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                
            elif args.property_type == 'eventually':
                iterval = 2
                diff_yp = output - property_upper
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                diff_yp = property_lower - output
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
            else:
                raise NotImplementedError
    
    if rho==True:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss), sum(batch_rho)/len(batch_rho)
    else:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss)


def transformer_prop_test(dataloader, net, args, loss_func, rho=False):
    net.eval()
    m = nn.ReLU()
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)
        
    for name, param in net.named_parameters():
        param.requires_grad = False 

    batch_rho = []
    batch_loss = []
    batch_cons_loss = []
    
    for X, y in dataloader:
        X = X.to("cuda")
        y = y.to("cuda")
        src_mask = src_mask.to('cuda')
        tgt_mask = tgt_mask.to('cuda')
        net.eval()
        X = X.unsqueeze(2)
        # print(f"X looks like {X}")
        # exit(0)
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
        
        if args.property_type == 'constraint':
            property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
            corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
            property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
            corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
            cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
        elif args.property_type == 'eventually':
            property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
            property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
            cons_loss = property_loss_eventually(output, property_upper, m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
        else:
            raise NotImplementedError
        
        pred_loss = loss_func(output.view(-1, 24), y) # TODO: add cp quantified uncertainty here
        batch_loss.append(pred_loss.item())
        batch_cons_loss.append(cons_loss.item())

        if rho==True:
            if args.property_type == 'constraint':
                batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_lower - output))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_upper))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
            
            elif args.property_type == 'eventually':
                iterval = 2
                diff_yp = output - property_upper
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                diff_yp = property_lower - output
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
            else:
                raise NotImplementedError
    
    if rho==True:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss), sum(batch_rho)/len(batch_rho)
    else:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss)


def transformer_prop_teacher_test_cp(dataloader, net, args, loss_func, rho=False, cp_value=None):
    net.eval()
    m = nn.ReLU()
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to("cuda")
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to("cuda")

    for name, param in net.named_parameters():
        param.requires_grad = False

    batch_rho = []
    batch_loss = []
    batch_cons_loss = []
    for X, y in dataloader:
        net.eval()
        X = X.unsqueeze(2).to("cuda")
        output = net(src=X, tgt=X[:, -24:, :], src_mask=src_mask, tgt_mask=tgt_mask).to("cuda")
        output = output.view(-1, 24).to("cuda")

        batch_diff = []
        squared_diff = []
        for i in range(len(output)):
            y_pred_i = output[i].cpu().numpy()

            diff_i = (y_pred_i - y[i].cpu().numpy()) ** 2
            batch_diff.append(diff_i)

            sqrt_diff_i = (y_pred_i - y[i].cpu().numpy())
            squared_diff.append(sqrt_diff_i)

        attributes = np.array(squared_diff).T
        first_10_of_one_client = [attribute[:] for attribute in attributes]
        s_r, f_r, cp_r = cp_guarantee_ratio_calculation(squared_diff, cp_value=cp_value)

        if args.property_type == 'constraint':
            property_upper, stl_lib_upper = generate_property_test(X, property_type="upper")
            corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
            property_lower, stl_lib_lower = generate_property_test(X, property_type="lower")
            corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
            teacher_pred = torch.min(output, corrected_trace_upper)
            teacher_pred = torch.max(output, corrected_trace_lower)
            output = teacher_pred
            cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
        elif args.property_type == 'eventually':
            property_upper, _ = generate_property_test(X, property_type="eventually-upper")
            property_lower, _ = generate_property_test(X, property_type="eventually-lower")
            cons_loss = property_loss_eventually(output, property_upper, m,
                                                 "eventually-upper") + property_loss_eventually(output, property_lower,
                                                                                                self.m,
                                                                                                "eventually-lower")
        else:
            raise NotImplementedError
        pred_loss = loss_func(output, y)
        batch_loss.append(pred_loss.item())
        batch_cons_loss.append(cons_loss.item())

        if rho == True:
            if args.property_type == 'constraint':
                batch_rho.append(
                    1 - torch.count_nonzero(m(corrected_trace_upper - output)) / len(corrected_trace_upper) /
                    corrected_trace_upper.shape[1])
                batch_rho.append(
                    1 - torch.count_nonzero(m(output - corrected_trace_lower)) / len(corrected_trace_lower) /
                    corrected_trace_lower.shape[1])

            elif args.property_type == 'eventually':
                iterval = 2
                diff_yp = output - property_upper
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1] // iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append(1 - torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1])
                diff_yp = property_lower - output
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1] // iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append(1 - torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1])
            else:
                raise NotImplementedError

    if rho == True:
        return net, sum(batch_loss) / len(batch_loss), sum(batch_cons_loss) / len(batch_cons_loss), sum(
            batch_rho) / len(batch_rho), first_10_of_one_client, s_r, f_r, cp_r
    else:
        return net, sum(batch_loss) / len(batch_loss), sum(batch_cons_loss) / len(
            batch_cons_loss), first_10_of_one_client, s_r, f_r, cp_r


def transformer_train(dataloader, net, args, loss_func, lr, w_glob_keys=None):
    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0
        if args.method == 'FedRep' and iter < args.head_iter:
            for name, param in net.named_parameters():
                if name in w_glob_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif args.method == 'FedRep' and iter >= args.head_iter:
            for name, param in net.named_parameters():
                if name in w_glob_keys:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in net.named_parameters():
                param.requires_grad = True 
        
        batch_loss = []
        for batch_idx, (X, y) in enumerate(dataloader): ## batch first=True
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)




def transformer_test(dataloader, net, args, loss_func):
    net.eval()

    local_eps = args.client_iter  # local update epochs
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120)
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24)

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0
        
        for name, param in net.named_parameters():
            param.requires_grad = False 
        
        batch_loss = []
        for X, y in dataloader:
            net.eval()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            
            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)



class LocalUpdateProx(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.ldr_cal = dataset["cal"]
        self.idxs = idxs  # client index

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, mu=0.1, server_model=None):
        
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_prox_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, server_model)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([
                {'params': weight_p, 'weight_decay':0.0001}, 
                {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            
            local_eps = self.args.client_iter  # local update epochs
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = True
            
            for it in range(local_eps):   # for # total local ep
                num_updates = 0
                batch_loss = []

                for batch_idx, (X, y) in enumerate(self.ldr_train):
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    loss = self.loss_func(output, y)

                    if it>0:
                        w_diff = torch.tensor(0., device=device)
                        for w, w_t in zip(server_model.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        loss += mu / 2. * w_diff

                    loss.backward()
                    optimizer.step()
                    num_updates += 1
                    batch_loss.append(loss.item())
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs
    
    
    def test(self, net, w_glob_keys, dataset_test=None, ind=-1, idx=-1):
        
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_test(self.ldr_train, net, self.args, self.loss_func)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            net.eval()
            epoch_loss = []
            num_updates = 0
            
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False 

            batch_loss = []
            batch_y_test = []
            batch_pred_test = []
            for X, y in self.ldr_train:
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                loss = self.loss_func(output, y)

                batch_y_test.append(y.detach().cpu().numpy())
                batch_pred_test.append(output.detach().cpu().numpy())
                
                net.zero_grad()
                num_updates += 1
                batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs



class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.idxs = idxs  # client index

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, w_ditto=None, lam=1):
        
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_ditto_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, w_ditto=w_ditto, lam=lam)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            net.train()
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([
                {'params': weight_p, 'weight_decay':0.0001}, 
                {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            
            local_eps = self.args.client_iter  # local update epochs
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = True
            
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0
                batch_loss = []

                for batch_idx, (X, y) in enumerate(self.ldr_train):
                    w_0 = copy.deepcopy(net.state_dict())
                    net.train()
                    net.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    loss = self.loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    if w_ditto is not None:
                        w_net = copy.deepcopy(net.state_dict())
                        for key in w_net.keys():
                            w_net[key] = w_net[key] - lr*lam*(w_0[key]-w_ditto[key])
                        net.load_state_dict(w_net)
                        optimizer.zero_grad()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs
        

    def test(self, net, w_glob_keys, dataset_test=None, ind=-1, idx=-1):

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_test(self.ldr_train, net, self.args, self.loss_func)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            net.eval()
            epoch_loss = []
            num_updates = 0
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False 

            batch_loss = []
            batch_y_test = []
            batch_pred_test = []
            for X, y in self.ldr_train:
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                loss = self.loss_func(output, y)

                batch_y_test.append(y.detach().cpu().numpy())
                batch_pred_test.append(output.detach().cpu().numpy())
                
                net.zero_grad()
                num_updates += 1
                batch_loss.append(loss.item())

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs


def cluster_explore_without_sp(net, w_glob_keys, lr, args, dataloaders, cp_value):

    # loss function will be changed to calculate the loss for each attribute in y. (24)
    loss_func_cp = new_loss_func
    loss_func = nn.MSELoss(reduction='mean')

    # batch size of trainset loader is 64.
    net.batch_size = 64

    # calibration result should be loaded here.
    # cp_path = "hdd/saved_cp_result/"
    # cp_dir = os.path.join(cp_path, "{}.json".format(args.method))

    # with open(cp_dir, 'r') as file:
    #     cp_file = json.load(file)
        
    # cp_value = cp_file['cp_value']

    if net.model_type == 'transformer':
        net, avg_ep_loss = transformer_prop_train(dataloaders, net, args, loss_func, lr, w_glob_keys=w_glob_keys)
        return net.state_dict(), avg_ep_loss
    
    if net.model_type != 'transformer':
        bias_p, weight_p = [], []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        m = nn.ReLU()
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, 
                                    {'params': bias_p, 'weight_decay':0}], 
                                    lr=lr, momentum=0.5)
        epoch_loss = []
        epoch_cons_loss = []
        hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        for name, param in net.named_parameters():
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for iter in range(args.cluster_fine_tune_iter):
            num_updates = 0
            batch_loss = []
            batch_cons_loss = []
            for batch_idx, (X, y) in enumerate(dataloaders):
                net.train()
                optimizer.zero_grad()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                pred_loss = loss_func(output, y)
                pred_loss_cp = loss_func_cp(cp_value = cp_value, y_pred = output, y_gt = y)
                subloss = [loss for sublist in pred_loss_cp for loss in sublist]
                cp_loss = np.mean([loss**2 for loss in subloss])
                
                loss = pred_loss + cp_loss
                loss.backward()
                optimizer.step()

                num_updates += 1
                batch_loss.append(loss.item())
                # batch_cons_loss.append(batch_loss)
                
                # if num_updates == args.local_updates_cp: break
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
    
    return net.state_dict(), sum(epoch_loss)/len(epoch_loss)

# local updates for FedRep, FedAvg
class LocalUpdate(object):
    """
    Federated learning updating class for a single agent.
    """
    # def __init__(self, args, dataset=None, idxs=None):
    #     self.args = args
    #     self.loss_func = nn.MSELoss()
    #     self.ldr_train = dataset["train_private"] # the train private dataset loader
    #     self.ldr_val = dataset["val"]
    #     self.ldr_test = dataset["test"]
    #     self.ldr_cal = dataset["cal"]
    #     self.idxs = idxs

    def __init__(self, args, dataset=None, idxs=None, loss_func=None):
        self.args = args
        self.loss_func = nn.MSELoss(reduction='mean')
        self.ldr_train = dataset["train_private"] # the train private dataset loader
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.ldr_cal = dataset["cal"]
        self.ldr_pre = dataset["pre"]
        self.idxs = idxs


    def pretrain(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        # pretrain should take MSE to calculate the loss.
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of trainset loader is 64. 
        net.batch_size = 64

        bias_p, weight_p = [], []
        for name, p in net.named_parameters():
            if 'bias' in name: 
                bias_p += [p]
            else: 
                weight_p += [p]
            
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, 
                                     {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
        
        local_eps_pretrain = self.args.pretrain_iter  # local update epochs
        epoch_loss = []
        hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

        for name, param in net.named_parameters():
            param.requires_grad = True
        
        for iter in range(local_eps_pretrain):
            num_updates = 0
            batch_loss = []

            for batch_idx, (X, y) in enumerate(self.ldr_pre):
                w_0 = copy.deepcopy(net.state_dict()) # used in ditto
                net.train()
                net.zero_grad()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                # print(f"pretrain output shape check expect (64, 24) is {output.shape}")
                assert output.shape == y.shape

                loss = self.loss_func(output, y)
                loss.backward()
                optimizer.step()

                num_updates += 1
                batch_loss.append(loss.item())
                
                if num_updates == self.args.pretrain_iter:
                    break
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # return updated net, average epoch loss, the current index
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs
    
    def calib(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        # loss function will be changed to calculate the loss for each attribute in y. (24)
        # self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of calibset loader is 64.
        net.batch_size = 64

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_train(self.ldr_cal, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            net.eval() # the model will not update in calibration
            cal_loss = []

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for X, y in self.ldr_cal:
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                # print(f"calib output shape check expect (64, 24) is {output.shape}")
                assert output.shape == y.shape

                # TODO: write a function to get calibration loss
                abs_diff = (output - y) ** 2 # find the distance between pred and gt
                abs_diff_t = abs_diff.t() # take transpose, because there are 24 attributes in the y.
                # print(abs_diff_t.shape)
                # exit(0)
                result_list = abs_diff_t.tolist() # result in a list with 24 elements, each element is a list with 64 differences. 

                loss = self.loss_func(output, y)
                cal_loss.append(loss.item())

            # print(f'cp_result list is {result_list}')
            return result_list

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        # loss function will be changed to calculate the loss for each attribute in y. (24)
        self.loss_func_cp = new_loss_func
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of trainset loader is 64.
        net.batch_size = 64

        # calibration result should be loaded here.
        cp_dic = dic_loader_without_specm(self.args)
        cp_value = find_group_info(cp_dic, idx)
    
        # print(cp_dic)
        # exit(0)

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            bias_p, weight_p = [], []
            net.batch_size = 64
            for name, p in net.named_parameters(): # rnn_1.weight_ih Parameter containing:
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            local_eps = self.args.client_iter  # local update epochs
            
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0
                
                if self.args.method == 'FedRep' and iter < self.args.head_iter:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif self.args.method == 'FedRep' and iter >= self.args.head_iter:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                else:
                    for name, param in net.named_parameters():
                        param.requires_grad = True 

                batch_loss = []
                for X, y in self.ldr_train:
                    # print(X.shape) # [64, 120]
                    # print(y.shape) # [64, 24]
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    pred_loss = self.loss_func(output, y) # the orginal loss

                    assert output.shape == y.shape

                    # the cp loss
                    pred_loss_cp = self.loss_func_cp(cp_value = cp_value, y_pred = output, y_gt = y)
                    subloss = [loss for sublist in pred_loss_cp for loss in sublist]
                    cp_loss = np.mean([loss**2 for loss in subloss])

                    # the loss in only cp case should only equal to cp + pred
                    loss = pred_loss + cp_loss
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs
        
    
    def calib_old(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        net.batch_size = 64

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            local_eps = self.args.client_iter 
            epoch_loss = []
            net.batch_size = 64

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for iter in range(local_eps):   # for # total local ep
                num_updates = 0
                
                if self.args.method == 'FedRep' and iter < self.args.head_iter:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif self.args.method == 'FedRep' and iter >= self.args.head_iter:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                else:
                    for name, param in net.named_parameters():
                        param.requires_grad = True 


                batch_loss = []
                for X, y in self.ldr_cal:
                    net.train()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                    # print(output.shape) # [64, 24]
                    # print(y.shape) # [64, 24]

                    abs_diff = torch.abs(output - y) # find the distance between pred and gt
                    # print(f"size check {abs_diff.shape}")
                    # print(f"{abs_diff}")

                    abs_diff_t = abs_diff.t() # take transpose, because there are 24 attributes in the y.
                    # print(f"size check {abs_diff_t.shape}")
                    # print(f"{abs_diff_t}")

                    result_list = abs_diff_t.tolist() # result in a list with 24 elements, each element is a list with 64 differences. 
                    # print(f"size check {len(result_list)}")
                    # print(f"length check {len(result_list[0])}")

                    # print(f"{result_list}")
                    loss = self.loss_func(output, y)
                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs, result_list
        
    def train_cluster(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        # loss function will be changed to calculate the loss for each attribute in y. (24)
        self.loss_func_cp = new_loss_func
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of trainset loader is 64.
        net.batch_size = 64

        # calibration result should be loaded here.
        cp_dic = dic_loader_without_specm(self.args)
        cp_value = find_group_info(cp_dic, idx)

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            local_eps = self.args.client_iter  # local update epochs

            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0

                for name, param in net.named_parameters():
                    param.requires_grad = True 

                batch_loss = []
                for X, y in self.ldr_train:
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    pred_loss = self.loss_func(output, y) # the orginal loss

                    assert output.shape == y.shape

                    # the cp loss
                    pred_loss_cp = self.loss_func_cp(cp_value = cp_value, y_pred = output, y_gt = y)
                    subloss = [loss for sublist in pred_loss_cp for loss in sublist]
                    cp_loss = np.mean([loss**2 for loss in subloss])

                    # the loss in only cp case should only equal to cp + pred
                    loss = pred_loss + cp_loss
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs

    def test(self, net, w_glob_keys, dataset_test=None, ind=-1, idx=-1):

        # loss function for test should always be MSE. 
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of testset loader is 15.
        net.batch_size = 15

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_test(self.ldr_val, net, self.args, self.loss_func)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            epoch_loss = []

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_loss = []
            batch_y_test = []
            batch_pred_test = []

            for X, y in self.ldr_test: # test set for dataset. 
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                loss = self.loss_func(output, y)

                batch_y_test.append(y.detach().cpu().numpy())
                batch_pred_test.append(output.detach().cpu().numpy())
                
                net.zero_grad()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs


def compute_cluster_id(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf)

    # print(f'index id in function is {idxs_users}')

    # exit(0)
    
    # load cluster models 
    for cluster in range(args.cluster):
        for c in idxs_users:
            local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)
            w_local, loss, idx = local.test(net=cluster_models[cluster] .to(args.device), idx=c, w_glob_keys=None)
            cluster_loss[cluster][c] = loss
    
    cluster_id = {}
    for cluster in range(args.cluster):
        cluster_id[cluster] = []
    
    client_lst = [c for c in idxs_users]
    i = 0
    while i < len(idxs_users):
        min_index = np.argwhere(cluster_loss == np.min(cluster_loss))
        if len(cluster_id[min_index[0][0]]) < int(args.client*args.frac/args.cluster) and min_index[0][1] in client_lst:
            cluster_id[min_index[0][0]].append(min_index[0][1])
            i += 1
            client_lst.remove(min_index[0][1])
        cluster_loss[min_index[0][0], min_index[0][1]] = np.inf

    return cluster_id


def compute_cluster_id_eval(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf)
    
    # load cluster models 
    for cluster in range(args.cluster):
        for c in idxs_users:
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
            w_local, loss, cons_loss, idx = local.test(net=cluster_models[cluster] .to(args.device), idx=c, w_glob_keys=None)
            cluster_loss[cluster][c] = loss
    
    cluster_id = {}
    for cluster in range(args.cluster):
        cluster_id[cluster] = []
    
    client_lst = [c for c in idxs_users]
    i = 0
    while i < len(idxs_users):
        min_index = np.argwhere(cluster_loss == np.min(cluster_loss))
        if len(cluster_id[min_index[0][0]]) < int(args.client*args.frac/args.cluster) and min_index[0][1] in client_lst:
            cluster_id[min_index[0][0]].append(min_index[0][1])
            i += 1
            client_lst.remove(min_index[0][1])
        cluster_loss[min_index[0][0], min_index[0][1]] = np.inf

    return cluster_id




def cluster_id_property(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf) # 5 * 100 with inf

    for cluster in range(args.cluster):
        for c in idxs_users:
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
            w_local, loss, cons_loss, idx = local.test(net=cluster_models[cluster] .to(args.device), idx=c, w_glob_keys=None)
            cluster_loss[cluster][c] = cons_loss
    
    cluster_id = {}
    for cluster in range(args.cluster):
        cluster_id[cluster] = []
    
    client_lst = [c for c in idxs_users]
    i = 0
    while i < len(idxs_users):
        min_index = np.argwhere(cluster_loss == np.min(cluster_loss))
        if len(cluster_id[min_index[0][0]]) < int(args.client*args.frac/args.cluster) and min_index[0][1] in client_lst:
            cluster_id[min_index[0][0]].append(min_index[0][1])
            i += 1
            client_lst.remove(min_index[0][1])
        cluster_loss[min_index[0][0], min_index[0][1]] = np.inf

    return cluster_id



def cluster_explore(net, w_glob_keys, lr, args, dataloaders, cp_value):

    # loss function will be changed to calculate the loss for each attribute in y. (24)
    loss_func_cp = new_loss_func
    loss_func = nn.MSELoss(reduction='mean')

    # batch size of trainset loader is 64.
    net.batch_size = 64

    # calibration result should be loaded here.
    # cp_path = "hdd/saved_cp_result/"
    # cp_dir = os.path.join(cp_path, "{}.json".format(args.method))

    # with open(cp_dir, 'r') as file:
    #     cp_file = json.load(file)
        
    # cp_value = cp_file['cp_value']

    if net.model_type == 'transformer':
        net, avg_ep_loss = transformer_prop_train(dataloaders, net, args, loss_func, lr, w_glob_keys=w_glob_keys, cp_value=cp_value)
        return net.state_dict(), avg_ep_loss
    
    if net.model_type != 'transformer':
        bias_p, weight_p = [], []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        m = nn.ReLU()
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, 
                                    {'params': bias_p, 'weight_decay':0}], 
                                    lr=lr, momentum=0.5)
        epoch_loss = []
        epoch_cons_loss = []
        hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        for name, param in net.named_parameters():
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for iter in range(args.cluster_fine_tune_iter):
            num_updates = 0
            batch_loss = []
            batch_cons_loss = []
            for batch_idx, (X, y) in enumerate(dataloaders):
                net.train()
                optimizer.zero_grad()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                pred_loss = loss_func(output, y)
                pred_loss_cp = loss_func_cp(cp_value = cp_value, y_pred = output, y_gt = y)
                subloss = [loss for sublist in pred_loss_cp for loss in sublist]
                cp_loss = np.mean([loss**2 for loss in subloss])
                
                if args.property_type == 'constraint':
                    property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                    cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
                
                elif args.property_type == 'eventually':
                    property_upper = generate_property_test(X, property_type = "eventually-upper")
                    property_lower = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, m, "eventually-upper") + property_loss_eventually(output, property_lower, m, "eventually-lower")
                
                else:
                    raise NotImplementedError
                
                loss = pred_loss + cons_loss + cp_loss
                loss.backward()
                optimizer.step()

                num_updates += 1
                batch_loss.append(loss.item())
                # batch_cons_loss.append(batch_loss)
                
                # if num_updates == args.local_updates_cp: break
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
    
    return net.state_dict(), sum(epoch_loss)/len(epoch_loss)


class LocalUpdateProp(object):
    """
    Federated learning updating class for a single agent with property mining.
    """
    # def __init__(self, args, dataset=None, idxs=None):
    #     self.args = args
    #     self.loss_func = nn.MSELoss(reduction='mean')
    #     self.m = nn.ReLU()
    #     self.ldr_train = dataset["train_private"]
    #     self.ldr_val = dataset["val"]
    #     self.ldr_test = dataset["test"]
    #     self.idxs = idxs

    def __init__(self, args, dataset=None, idxs=None, loss_func=None):
        self.args = args
        self.loss_func = nn.MSELoss(reduction='mean')
        self.ldr_train = dataset["train_private"] # the train private dataset loader
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.ldr_cal = dataset["cal"]
        self.ldr_pre = dataset["pre"]
        self.ldr_nor = dataset['nor']
        self.idxs = idxs

    def pretrain(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        # pretrain should take MSE to calculate the loss.
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of trainset loader is 64. 
        net.batch_size = 64

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_prop_pretrain(self.ldr_pre, net, self.args, self.loss_func, lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:

            print()
            print(f"the pretrain function for OTHER is applied.")
            print()

            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name: 
                    bias_p += [p]
                else: 
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, 
                                        {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.9)
            
            local_eps_pretrain = self.args.pretrain_iter  # local update epochs
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = True
            
            for iter in range(local_eps_pretrain):
                num_updates = 0
                batch_loss = []

                for batch_idx, (X, y) in enumerate(self.ldr_pre):
                    w_0 = copy.deepcopy(net.state_dict()) # used in ditto
                    net.train()
                    net.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                    # print(f"pretrain output shape check expect (64, 24) is {output.shape}")
                    assert output.shape == y.shape

                    loss = self.loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.pretrain_iter:
                        break
            
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # return updated net, average epoch loss, the current index
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs
    
    def calib_norm(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        net.batch_size = 50

        if net.model_type == 'transformer':
            max_values = transformer_prop_calib_norm(self.ldr_nor, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys)
            return max_values
        
        else:
            net.eval() # the model will not update in calibration
            cal_loss = []

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for X, y in self.ldr_nor:
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                # print(f"calib output shape check expect (64, 24) is {output.shape}")
                assert output.shape == y.shape

                # TODO: write a function to get calibration loss
                abs_diff = abs(output - y) # find the distance between pred and gt
                # print(abs_diff.shape)
                # abs_diff_t = abs_diff.t() # take transpose, because there are 24 attributes in the y.
                # print(abs_diff_t.shape)
                # exit(0)
                result_list = abs_diff.tolist() # result in a list with 24 elements, each element is a list with 64 differences. 

                max_values = np.max(result_list, axis=0)

                # print(max_values)
                # print(max_values.shape)

            # print(f'cp_result list is {result_list}')
            return max_values


    def calib(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, max_values = None):

        # loss function will be changed to calculate the loss for each attribute in y. (24)
        # self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of calibset loader is 64.
        net.batch_size = 90
        cal_loss = []

        if net.model_type == 'transformer':
            c_tuda_list = transformer_prop_calib(self.ldr_cal, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys, max_values=max_values)
            return c_tuda_list
        
        else:
            net.eval() # the model will not update in calibration

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for X, y in self.ldr_cal:
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                # print(f"calib output shape check expect (64, 24) is {output.shape}")
                assert output.shape == y.shape

                # TODO: write a function to get calibration loss
                abs_diff = abs(output - y).detach().numpy() # find the distance between pred and gt
                # print(max_values)
                # print(abs_diff.shape)
                # print(abs_diff[:2])
                # print(len(abs_diff))
                abs_diff = np.array(abs_diff, dtype=np.float32)
                # print(abs_diff[:2])  
                # print(len(abs_diff[0]))
                # print(len(abs_diff))
                # exit(0)
                # print(f'unnormalied diff is {abs_diff}')
                max_values = np.array(max_values, dtype=np.float32) 

                for i in range(len(abs_diff)):
                    abs_diff[i] /= max_values
                
                assert abs_diff.shape == (90, 24)
                # print(abs_diff.shape)
                # print(abs_diff)
                # exit(0)
                max_value_of_row = np.max(abs_diff, axis=1)# take transpose, because there are 24 attributes in the y.
                assert max_value_of_row.shape == (90, )
                # print(max_value_of_row.shape)
                # print(max_value_of_row)
                # exit(0)
                # result_list = abs_diff_t.tolist() # result in a list with 24 elements, each element is a list with 64 differences. 
                c_tuda_list = max_value_of_row

                loss = self.loss_func(output, y)
                cal_loss.append(loss.item())

                # exit(0)

            # print(f'cp_result list is {result_list}')
            return c_tuda_list


    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, cp_value=None):

        # loss function will be changed to calculate the loss for each attribute in y. (24)
        self.loss_func_cp = new_loss_func
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of trainset loader is 64.
        net.batch_size = 64

        # calibration result should be loaded here.
        # cp_path = "hdd/saved_cp_result/"
        # cp_dir = os.path.join(cp_path, "{}.json".format(self.args.method))

        # with open(cp_dir, 'r') as file:
        #     cp_file = json.load(file)

        # cp_dic = dic_loader(self.args)
        # cp_value = find_group_info(cp_dic, idx)
        cp_value = cp_value
        
        # print(f"CP value is loaded.")
        # print(f"cp value looks like {cp_value}")

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_prop_train(self.ldr_train, net, self.args, self.loss_func, lr, w_glob_keys=w_glob_keys, cp_value=cp_value)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name: 
                    bias_p += [p]
                else: 
                    weight_p += [p]
        
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.9)
            local_eps = self.args.client_iter   # local update epochs
            
            epoch_loss = []
            epoch_cons_loss = []

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            
            for iter in range(local_eps):
                num_updates = 0
                
                for name, param in net.named_parameters():
                    param.requires_grad = True 

                # cpu
                batch_loss = []
                batch_cons_loss = []
                for X, y in self.ldr_train:
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    # X = X.to("cuda")
                    # y = y.to("cuda")
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    pred_loss = self.loss_func(output, y)

                    # print(f"train output shape check expect (64, 24) is {output.shape}")
                    # print(f"train output shape check expect (64, 24) is {y.shape}")
                    assert output.shape == y.shape

                    pred_loss_cp = self.loss_func_cp(cp_value = cp_value, y_pred = output, y_gt = y)
                    subloss = [loss for sublist in pred_loss_cp for loss in sublist]
                    cp_loss = np.mean([loss**2 for loss in subloss])

                    # print(f"after cp loss shape check expect (64, 24) is {pred_loss}")
                    # print("==============================================================")

                    if self.args.property_type == 'constraint':
                        property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                        corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                        property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                        corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                        cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                    elif self.args.property_type == 'eventually':
                        property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
                        property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
                        cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")

                    else:
                        raise NotImplementedError
                    
                    # print(f"pred_loss is {pred_loss}")
                    # print(f"cons_loss is {cons_loss}")
                    # # print(f"cons_loss is {pred_loss_cp}")
                    # print(f"cp_loss is {cp_loss}")
                    # # exit(0)

                    # loss = cons_loss + pred_loss
                    loss = pred_loss + cons_loss + cp_loss
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(pred_loss.item())
                    batch_cons_loss.append(cons_loss.item())
                    
                    # if num_updates == self.args.local_updates_cp: break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs

    def test(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False):
        
        # loss function for test should always be MSE. 
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of testset loader is 15.
        net.batch_size = 15

        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []

        if net.model_type == 'transformer':
            if rho == True:
                net, ep_ls, ep_cons_ls, ep_rho= transformer_prop_test(self.ldr_test, net, self.args, self.loss_func, rho=True)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_test(self.ldr_test, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs

        else:
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []

            for X, y in self.ldr_test:
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                pred_loss = self.loss_func(output, y)
                batch_loss.append(pred_loss.item())
                
                if self.args.property_type == 'constraint':
                    property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)

                elif self.args.property_type == 'eventually':
                    property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
                    property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
            
                else:
                    raise NotImplementedError
                
                batch_cons_loss.append(cons_loss.item())

                if rho==True:
                    if self.args.property_type == 'constraint':
                        batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_lower - output))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                        batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_upper))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
                    
                    elif self.args.property_type == 'eventually':
                        iterval = 2
                        diff_yp = output - property_upper
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                        diff_yp = property_lower - output
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                    else:
                        raise NotImplementedError

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
            if rho==True:
                epoch_rho.append(sum(batch_rho)/len(batch_rho))
            
        if rho==True:
            return sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs, sum(epoch_rho)/len(epoch_rho)
        else:
            # return net.state_dict(), 0, 0, self.idxs
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs
        

    def test_teacher(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False):

        # print(cp_value)
        
        # print(f"cp_value for client {idx} is {cp_value}")
        # loss function for validation should be MSE to corporate with evaluation metric. 
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of testset loader is 5.
        net.batch_size = 100

        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []
        num_updates = 0

        if net.model_type == 'transformer':
            if rho == True:
                net, ep_ls, ep_cons_ls, ep_rho= transformer_prop_teacher_test(self.ldr_val, net, self.args, self.loss_func, rho=True)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_teacher_test(self.ldr_val, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs
        
        else:
            net.batch_size = 100
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            for X, y in self.ldr_val:
                # print(f"y shape is {y.shape}")
                # exit(0)
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                # print(f"output shape is {output.shape}")
                # exit(0)
                # for batch contains 15 data, each data is the abs diff between y_pred and y of all 24 attributes. 
                ## the cp marginal ratio for validation set is calculated here. 
                batch_diff = [] 
                squared_diff = []
                for i in range(len(output)):
                    y_pred_i = output[i].numpy()
                    sqrt_diff_i = (y_pred_i - y[i].numpy())
                    diff_i = (y_pred_i - y[i].numpy())**2
                    batch_diff.append(diff_i)
                    squared_diff.append(sqrt_diff_i)

                print(f"===================\n")
                pred_loss = self.loss_func(output, y)

                if self.args.property_type == 'constraint':
                    property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                    teacher_pred = torch.min(output, corrected_trace_upper)
                    teacher_pred = torch.max(output, corrected_trace_lower)
                    output = teacher_pred
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                elif self.args.property_type == 'eventually':
                    property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
                    property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
                else:
                    raise NotImplementedError

                batch_loss.append(pred_loss.item())
                batch_cons_loss.append(cons_loss.item())

                if rho==True:
                    if self.args.property_type == 'constraint':
                        batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_upper - output))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
                        batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_lower))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                    
                    elif self.args.property_type == 'eventually':
                        iterval = 2
                        diff_yp = output - property_upper
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                        diff_yp = property_lower - output
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                    else:
                        raise NotImplementedError

                num_updates += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
            if rho==True:
                epoch_rho.append(sum(batch_rho)/len(batch_rho))
            
        if rho==True:
            return sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs, sum(epoch_rho)/len(epoch_rho)
        else:
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs
        
    
    def test_teacher_cp(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False, cp_value=None, Visual=False):

        # print(cp_value)
        
        # print(f"cp_value for client {idx} is {cp_value}")
        # loss function for validation should be MSE to corporate with evaluation metric. 
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of testset loader is 5.
        net.batch_size = 100

        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []
        num_updates = 0

        if net.model_type == 'transformer':
            if rho == True:
                net, ep_ls, ep_cons_ls, ep_rho, first_10_of_one_client, s_r, f_r, cp_r= transformer_prop_teacher_test_cp(self.ldr_val, net, self.args, self.loss_func, rho=True, cp_value=cp_value)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho, first_10_of_one_client, s_r, f_r, cp_r
            else:
                net, ep_ls, ep_cons_ls, first_10_of_one_client, s_r, f_r, cp_r = transformer_prop_teacher_test_cp(self.ldr_val, net, self.args, self.loss_func, rho=False, cp_value=cp_value)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs,  first_10_of_one_client, s_r, f_r, cp_r
        
        else:
            net.batch_size = 100
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            for X, y in self.ldr_val:
                # print(f"y shape is {y.shape}")
                # exit(0)
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                # print(f"output shape is {output.shape}")
                # exit(0)
                # for batch contains 15 data, each data is the abs diff between y_pred and y of all 24 attributes. 
                ## the cp marginal ratio for validation set is calculated here. 
                batch_diff = [] 
                squared_diff = []
                for i in range(len(output)):
                    y_pred_i = output[i].cpu().numpy()
                    sqrt_diff_i = (y_pred_i - y[i].cpu().numpy())
                    diff_i = (y_pred_i - y[i].cpu().numpy())**2
                    batch_diff.append(diff_i)
                    squared_diff.append(sqrt_diff_i)
                
                # print(len(batch_diff))
                # print(squared_diff)
                # print(squared_diff[0])

                # Convert your data (a list of lists) into a format where each attribute is its own list
                attributes = np.array(squared_diff).T
                # print(attributes[0])
                # print(len(attributes))
                # print(len(attributes[0][0:10]))
                first_10_of_one_client = [attribute[:] for attribute in attributes]
                # print(first_10_of_one_client)
                # print(len(first_10_of_one_client))
                # print(len(first_10_of_one_client[0]))
                # exit(0)


                # # Create a figure and axis
                # plt.figure(figsize=(12, 6))

                # # Plot the values for each attribute
                # for attribute_index, values in enumerate(attributes):
                #     # Use scatter plot to show each of the 100 values for the attribute
                #     plt.scatter([attribute_index + 1] * len(values), values, label=f'Attribute {attribute_index + 1}', alpha=0.6)

                #     # Calculate the mean value for each attribute
                #     mean_value = np.mean(values)
                    
                #     # Calculate upper and lower bounds based on the radius
                #     upper_bound = 0 + np.sqrt(cp_value[attribute_index])
                #     lower_bound = 0 - np.sqrt(cp_value[attribute_index])

                #     # Plot the upper and lower bounds as lines
                #     plt.plot([attribute_index + 1 - 0.2, attribute_index + 1 + 0.2], [upper_bound, upper_bound], color='red', linestyle='--')
                #     plt.plot([attribute_index + 1 - 0.2, attribute_index + 1 + 0.2], [lower_bound, lower_bound], color='blue', linestyle='--')

                #     # Optionally: shade the area between the upper and lower bound
                #     plt.fill_between([attribute_index + 1 - 0.2, attribute_index + 1 + 0.2], lower_bound, upper_bound, color='gray', alpha=0.2)

                # # Set the x-axis and y-axis labels
                # plt.xlabel('Attribute')
                # plt.ylabel('Value')

                # # Add title
                # plt.title('Values of Each Attribute with Upper and Lower Bounds')

                # # Set x-ticks to represent each attribute
                # plt.xticks(range(1, 25), [f'Attr {i}' for i in range(1, 25)])

                # # Show the plot
                # plt.tight_layout()
                
                # # exit(0)
                # print(cp_value)
                # exit(0)
                # numbers = np.array(cp_value)
                # result = np.power(2, numbers)
                s_r, f_r, cp_r= cp_guarantee_ratio_calculation(squared_diff, cp_value=cp_value)
                # # print(s_r, f_r, cp_r)
                # # plt.show()
                # return s_r, f_r, cp_r

                # print(f"===================\n")
                # # assert len(cp_value) == len(output[0])
                # # exit(0)

                pred_loss = self.loss_func(output, y)

                if self.args.property_type == 'constraint':
                    property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                    teacher_pred = torch.min(output, corrected_trace_upper)
                    teacher_pred = torch.max(output, corrected_trace_lower)
                    output = teacher_pred
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                elif self.args.property_type == 'eventually':
                    property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
                    property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
                else:
                    raise NotImplementedError

                batch_loss.append(pred_loss.item())
                batch_cons_loss.append(cons_loss.item())

                if rho==True:
                    if self.args.property_type == 'constraint':
                        batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_upper - output))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
                        batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_lower))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                    
                    elif self.args.property_type == 'eventually':
                        iterval = 2
                        diff_yp = output - property_upper
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                        diff_yp = property_lower - output
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                    else:
                        raise NotImplementedError
                
                # if cp_val == True: 
                num_updates += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
            if rho==True:
                epoch_rho.append(sum(batch_rho)/len(batch_rho))
            
        if rho==True:
            return sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs, sum(epoch_rho)/len(epoch_rho), first_10_of_one_client, s_r, f_r, cp_r
        else:
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs

    def test_teacher_cp_bu(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False,
                            cp_value=None, Visual=False):

            self.loss_func = nn.MSELoss(reduction='mean')

            net.batch_size = 100

            m = nn.ReLU()
            epoch_rho = []
            epoch_loss = []
            epoch_cons_loss = []
            num_updates = 0

            if net.model_type == 'transformer':
                if rho == True:
                    net, ep_ls, ep_cons_ls, ep_rho, first_10_of_one_client, s_r, f_r, cp_r = transformer_prop_teacher_test(
                        self.ldr_val, net, self.args, self.loss_func, rho=True)
                    return ep_ls, ep_cons_ls, self.idxs, ep_rho, first_10_of_one_client, s_r, f_r, cp_r
                else:
                    net, ep_ls, ep_cons_ls, first_10_of_one_client, s_r, f_r, cp_r = transformer_prop_teacher_test(
                        self.ldr_val, net, self.args, self.loss_func, rho=False)
                    return net.state_dict(), ep_ls, ep_cons_ls, self.idxs, first_10_of_one_client, s_r, f_r, cp_r

            else:
                net.batch_size = 100
                hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

                for name, param in net.named_parameters():
                    param.requires_grad = False

                batch_rho = []
                batch_loss = []
                batch_cons_loss = []
                for X, y in self.ldr_val:

                    net.eval()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                    batch_diff = []
                    squared_diff = []
                    for i in range(len(output)):
                        y_pred_i = output[i].cpu().numpy()
                        sqrt_diff_i = (y_pred_i - y[i].cpu().numpy())
                        diff_i = (y_pred_i - y[i].cpu().numpy()) ** 2
                        batch_diff.append(diff_i)
                        squared_diff.append(sqrt_diff_i)

                    attributes = np.array(squared_diff).T
                    first_10_of_one_client = [attribute[:] for attribute in attributes]
                    s_r, f_r, cp_r = cp_guarantee_ratio_calculation(squared_diff, cp_value=cp_value)
                    pred_loss = self.loss_func(output, y)

                    if self.args.property_type == 'constraint':
                        property_upper, stl_lib_upper = generate_property_test(X, property_type="upper")
                        corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                        property_lower, stl_lib_lower = generate_property_test(X, property_type="lower")
                        corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                        teacher_pred = torch.min(output, corrected_trace_upper)
                        teacher_pred = torch.max(output, corrected_trace_lower)
                        output = teacher_pred
                        cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output,
                                                                                                   corrected_trace_lower)
                    elif self.args.property_type == 'eventually':
                        property_upper, _ = generate_property_test(X, property_type="eventually-upper")
                        property_lower, _ = generate_property_test(X, property_type="eventually-lower")
                        cons_loss = property_loss_eventually(output, property_upper, self.m,
                                                             "eventually-upper") + property_loss_eventually(output,
                                                                                                            property_lower,
                                                                                                            self.m,
                                                                                                            "eventually-lower")
                    else:
                        raise NotImplementedError

                    batch_loss.append(pred_loss.item())
                    batch_cons_loss.append(cons_loss.item())

                    if rho == True:
                        if self.args.property_type == 'constraint':
                            batch_rho.append(1 - torch.count_nonzero(m(corrected_trace_upper - output)) / len(
                                corrected_trace_upper) / corrected_trace_upper.shape[1])
                            batch_rho.append(1 - torch.count_nonzero(m(output - corrected_trace_lower)) / len(
                                corrected_trace_lower) / corrected_trace_lower.shape[1])

                        elif self.args.property_type == 'eventually':
                            iterval = 2
                            diff_yp = output - property_upper
                            unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1] // iterval), iterval)
                            diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                            batch_rho.append(1 - torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1])
                            diff_yp = property_lower - output
                            unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1] // iterval), iterval)
                            diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                            batch_rho.append(1 - torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1])
                        else:
                            raise NotImplementedError

                    # if cp_val == True:
                    num_updates += 1

                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                epoch_cons_loss.append(sum(batch_cons_loss) / len(batch_cons_loss))
                if rho == True:
                    epoch_rho.append(sum(batch_rho) / len(batch_rho))

            if rho == True:
                return sum(epoch_loss) / len(epoch_loss), sum(epoch_cons_loss) / len(epoch_cons_loss), self.idxs, sum(
                    epoch_rho) / len(epoch_rho), first_10_of_one_client, s_r, f_r, cp_r
            else:
                return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_cons_loss) / len(
                    epoch_cons_loss), self.idxs


