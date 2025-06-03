#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819

import sys
import copy
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

    for group_id, group_info in groups.items():
        if index in group_info['clients']:
            return group_info['cp_value']

    print(f"CP value not found.")
    return None

def cp_guarantee_ratio_calculation(y_batch, cp_value):

    length_batch = len(y_batch)
    att_length = len(y_batch[0])
    sat_rate = 0
    fail_rate = 0

    for i in range(length_batch):
        for j in range(att_length):

            if np.abs(y_batch[i][j]) <= np.abs(cp_value[j]):
                sat_rate += 1
            else:
                fail_rate += 1
    
    cp_rate = sat_rate / (sat_rate + fail_rate)

    return sat_rate, fail_rate, cp_rate


def dic_loader(args):

    if args.sep_type == "spec_m":
        if args.dataset == 'fhwa':
            cp_pat = f"hdd/cluster_cp_result_with_specm_{args.model}_{args.cp_epoch}_{args.client}/LogiCP_Cluster.json"
            print()
            print(f"the cp region is loaded from {cp_pat}")
            print()
            with open(cp_pat, 'r') as file:
                cp_dic = json.load(file)

        elif args.dataset == 'ct':
            cp_pat = f"ct/cluster_cp_result_with_specm_{args.model}_{args.cp_epoch}_{args.client}/LogiCP_Cluster.json"
            with open(cp_pat, 'r') as file:
                cp_dic = json.load(file)
            print()
            print(f"cp region is loaded from {cp_pat}")
            print()

    if args.sep_type == "value":
        if args.dataset == 'fhwa':
            cp_pat = f"hdd_{args.method}_paper/cluster_cp_result_with_value_{args.model}_{args.cp_epoch}_{args.client}/{args.method}_Cluster.json"
            print(f"cp is loaded from {cp_pat}")
            
            with open(cp_pat, 'r') as file:
                cp_dic = json.load(file)
        
        elif args.dataset == 'ct':
            cp_pat = f"{args.dataset}_{args.method}/cluster_cp_result_with_value_{args.model}_{args.cp_epoch}_{args.client}/{args.method}_Cluster.json"
            print(f"cp is loaded from {cp_pat}")
            
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
        cp_pat = f"{args.dataset}_{args.method}/cluster_cp_result_with_value_{args.model}_{args.cp_epoch}_{args.client}/{args.method}_Cluster.json"
        print(f"cp is loaded from {cp_pat}")
        
        with open(cp_pat, 'r') as file:
            cp_dic = json.load(file)

    return cp_dic

def new_loss_func(cp_value, y_gt, y_pred):

    temp = abs(y_pred - y_gt)
    loss = []
    data_loss = []
    
    for i in range(len(temp)):
        data_loss = []
        for j in range(len(temp[i])):
            if temp[i][j] >= cp_value[j]:
                att_loss = cp_value[j]
            else:
                att_loss = temp[i][j].item()
            data_loss.append(att_loss)
        
        loss.append(data_loss)

    return loss


def loss_func_cp(cp_value, y_gt, y_pred):

    if isinstance(cp_value, list):
        cp_value = torch.tensor(cp_value, dtype=y_pred.dtype, device=y_pred.device)
    
    if cp_value.dim() == 1:
        cp_value = cp_value.unsqueeze(0).expand_as(y_pred)

    error = torch.abs(y_pred - y_gt)
    loss = torch.clamp(error - cp_value, min=0.0)

    return loss.detach().cpu().tolist()

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

def transformer_prop_pretrain(dataloader, net, args, loss_func, lr, w_glob_keys=None, cp_value=None):

    loss_func = nn.MSELoss(reduction='mean')
    net.batch_size = 64

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.9)
    local_eps_pretrain = args.pretrain_iter
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')

    epoch_loss = []

    for iter in range(local_eps_pretrain):

        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        num_updates = 0
        batch_loss = []

        for X, y in dataloader: 
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2).to('cuda')
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
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
    
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')

    net.eval()

    for X, y in dataloader:
        X = X.unsqueeze(2).to('cuda')
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
        abs_diff = abs(output.view(-1, 24) - y) 
        result_list = abs_diff.tolist() 
        max_values = np.max(result_list, axis=0)

    return max_values

def transformer_prop_calib(dataloader, net, args, loss_func, lr, w_glob_keys=None, max_values = None):

    max_values = np.array(max_values, dtype=np.float32) 

    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')

    net.eval()
    for X, y in dataloader:
        X = X.unsqueeze(2).to('cuda')
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)

        abs_diff = abs(output.view(-1, 24) - y).cpu().detach().numpy() 
        abs_diff = np.array(abs_diff, dtype=np.float32)

        for i in range(len(abs_diff)):
            abs_diff[i] /= max_values
        
        max_value_of_row = np.max(abs_diff, axis=1)
        c_tuda_list = max_value_of_row
        
    return c_tuda_list

def transformer_prop_train(dataloader, net, args, loss_func, lr, w_glob_keys=None, cp_value=None, lf = None):

    loss_func_cp = lf
    loss_func = nn.MSELoss(reduction='mean')

    net.batch_size = 64

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.9)
    local_eps = args.client_iter
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')

    epoch_loss = []
    epoch_cons_loss = []

    for iter in range(local_eps):
        num_updates = 0

        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        batch_loss = []
        batch_cons_loss = []
        for X, y in dataloader:
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2).to('cuda')
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            pred_loss = loss_func(output.view(-1, 24), y)
            pred_loss_cp = loss_func_cp(cp_value = cp_value, y_pred = output.view(-1, 24), y_gt = y)

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


def transformer_prop_cp_teacher(dataloader, net, args, loss_func, rho=False, cp_value=None):
    import torch.nn as nn

    net.eval()
    m = nn.ReLU()
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')

    for name, param in net.named_parameters():
        param.requires_grad = False

    batch_rho = []
    batch_loss = []
    batch_cons_loss = []
    pre_correction_diff = []
    post_correction_diff = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.unsqueeze(2).to('cuda')
            y = y.to('cuda')

            output = net(src=X, tgt=X[:, -24:, :], src_mask=src_mask, tgt_mask=tgt_mask)
            output = output.view(-1, 24)

            # Compute CP guarantee BEFORE correction
            for i in range(len(output)):
                y_pred_i = output[i].detach().cpu().numpy()
                y_true_i = y[i].detach().cpu().numpy()
                pre_correction_diff.append(y_pred_i - y_true_i)

            # Generate STL properties
            property_upper, stl_lib_upper = generate_property_test(X, property_type="upper")
            corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
            property_lower, stl_lib_lower = generate_property_test(X, property_type="lower")
            corrected_trace_lower = convert_best_trace(stl_lib_lower, corrected_trace_upper)

            # CP-based correction
            for i in range(len(output)):
                y_pred_i = output[i].detach().cpu().numpy()
                y_corrected_upper = property_upper[i].detach().cpu().numpy()[:24]
                y_corrected_lower = property_lower[i].detach().cpu().numpy()[:24]

                for j in range(len(y_pred_i)):
                    assert y_corrected_upper[j] >= y_corrected_lower[j]
                    if y_pred_i[j] > y_corrected_upper[j]:
                        cp_lower = y_pred_i[j] - cp_value[j]
                        y_pred_i[j] = cp_lower if cp_lower > y_corrected_upper[j] else y_corrected_upper[j]
                    elif y_pred_i[j] < y_corrected_lower[j]:
                        cp_upper = y_pred_i[j] + cp_value[j]
                        y_pred_i[j] = cp_upper if cp_upper < y_corrected_lower[j] else y_corrected_lower[j]

                output[i] = torch.from_numpy(y_pred_i).to(output.device)

            # Compute CP guarantee AFTER correction
            corrected_output = output.detach().cpu().numpy()
            true_values = y.detach().cpu().numpy()
            for i in range(len(corrected_output)):
                post_correction_diff.append(corrected_output[i] - true_values[i])

            # Loss calculations
            cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
            pred_loss = loss_func(output, y)

            batch_loss.append(pred_loss.item())
            batch_cons_loss.append(cons_loss.item())

            if rho and args.property_type == 'constraint':
                batch_rho.append(
                    1 - torch.count_nonzero(m(corrected_trace_lower - output)) / corrected_trace_lower.numel()
                )
                batch_rho.append(
                    1 - torch.count_nonzero(m(output - corrected_trace_upper)) / corrected_trace_upper.numel()
                )

    # Compute CP rates
    s_r_pre, f_r_pre, cp_r_pre = cp_guarantee_ratio_calculation(pre_correction_diff, cp_value=cp_value)
    s_r_post, f_r_post, cp_r_post = cp_guarantee_ratio_calculation(post_correction_diff, cp_value=cp_value)

    print("Transformer CP Guarantee BEFORE Correction:", cp_r_pre)
    print("Transformer CP Guarantee AFTER Correction:", cp_r_post)

    if rho:
        return (
            sum(batch_loss) / len(batch_loss),
            sum(batch_cons_loss) / len(batch_cons_loss),
            sum(batch_rho) / len(batch_rho), s_r_pre, f_r_pre, s_r_post, f_r_post
        )
    else:
        return net.state_dict(), sum(batch_loss) / len(batch_loss), sum(batch_cons_loss) / len(batch_cons_loss)


def transformer_prop_test(dataloader, net, args, loss_func, rho=False):
    net.eval()
    m = nn.ReLU()
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')
        
    for name, param in net.named_parameters():
        param.requires_grad = False 

    batch_rho = []
    batch_loss = []
    batch_cons_loss = []
    
    for X, y in dataloader:
        net.eval()
        X = X.unsqueeze(2).to('cuda')
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
        
        if args.property_type == 'constraint':
            property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
            corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
            property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
            corrected_trace_lower = convert_best_trace(stl_lib_lower, corrected_trace_upper)
            cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
        elif args.property_type == 'eventually':
            property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
            property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
            cons_loss = property_loss_eventually(output, property_upper, m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
        else:
            raise NotImplementedError
        
        pred_loss = loss_func(output.view(-1, 24), y)
        batch_loss.append(pred_loss.item())
        batch_cons_loss.append(cons_loss.item())

        if rho==True:
            if args.property_type == 'constraint':
                batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_lower - output))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_upper))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )

    if rho==True:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss), sum(batch_rho)/len(batch_rho)
    else:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss)

def transformer_prop_cp_rate(dataloader, net, args, loss_func, rho=False, cp_value=None):
    net.eval()
    m = nn.ReLU()
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')

    for name, param in net.named_parameters():
        param.requires_grad = False

    batch_rho = []
    batch_loss = []
    batch_cons_loss = []
    residuals = []  # For CP guarantee rate calculation

    with torch.no_grad():
        for X, y in dataloader:
            X = X.unsqueeze(2).to('cuda')
            y = y.to('cuda')

            output = net(src=X, tgt=X[:, -24:, :], src_mask=src_mask, tgt_mask=tgt_mask)

            # Collect residuals for CP guarantee rate (no correction applied)
            pred_flat = output.view(-1, 24).detach().cpu().numpy()
            true_flat = y.detach().cpu().numpy()
            for i in range(len(pred_flat)):
                residuals.append(pred_flat[i] - true_flat[i])

            if args.property_type == 'constraint':
                property_upper, stl_lib_upper = generate_property_test(X, property_type="upper")
                corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                property_lower, stl_lib_lower = generate_property_test(X, property_type="lower")
                corrected_trace_lower = convert_best_trace(stl_lib_lower, corrected_trace_upper)
                cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
            elif args.property_type == 'eventually':
                property_upper, _ = generate_property_test(X, property_type="eventually-upper")
                property_lower, _ = generate_property_test(X, property_type="eventually-lower")
                cons_loss = (
                    property_loss_eventually(output, property_upper, m, "eventually-upper") +
                    property_loss_eventually(output, property_lower, m, "eventually-lower")
                )
            else:
                raise NotImplementedError

            pred_loss = loss_func(output.view(-1, 24), y)
            batch_loss.append(pred_loss.item())
            batch_cons_loss.append(cons_loss.item())

            if rho and args.property_type == 'constraint':
                batch_rho.append(
                    1 - torch.count_nonzero(m(corrected_trace_lower - output)) / corrected_trace_lower.numel()
                )
                batch_rho.append(
                    1 - torch.count_nonzero(m(output - corrected_trace_upper)) / corrected_trace_upper.numel()
                )

    # Compute CP guarantee rate without correction
    s_r, f_r, cp_r = cp_guarantee_ratio_calculation(residuals, cp_value=cp_value)
    print("Transformer CP Guarantee Rate WITHOUT Correction:", cp_r)

    if rho:
        return sum(batch_loss) / len(batch_loss), sum(batch_cons_loss) / len(batch_cons_loss), sum(batch_rho) / len(batch_rho), s_r, f_r, cp_r
    else:
        return net, sum(batch_loss) / len(batch_loss), sum(batch_cons_loss) / len(batch_cons_loss)


def transformer_train(dataloader, net, args, loss_func, lr, w_glob_keys=None, cp_value=None):

    loss_func_cp = new_loss_func
    loss_func = nn.MSELoss(reduction='mean')

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0

        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        batch_loss = []
        for batch_idx, (X, y) in enumerate(dataloader): ## batch first=True
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2).to('cuda')
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            pred_loss_cp = loss_func_cp(cp_value = cp_value, y_pred = output.view(-1, 24), y_gt = y)

            subloss = [loss for sublist in pred_loss_cp for loss in sublist]
            # print(len(subloss)) # 1536 = 24 * 64, the abs loss, which need to take power 2 in next line to calculate MSE. 
            cp_loss = np.mean([loss**2 for loss in subloss])

            loss = loss + cp_loss
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
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).to('cuda')
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).to('cuda')

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0
        
        for name, param in net.named_parameters():
            param.requires_grad = False 
        
        batch_loss = []
        for X, y in dataloader:
            net.eval()
            X = X.unsqueeze(2).to('cuda')
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            
            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)


def cluster_explore_without_sp(net, w_glob_keys, lr, args, dataloaders, cp_value):

    loss_func_cp = new_loss_func
    loss_func = nn.MSELoss(reduction='mean')

    net.batch_size = 64

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
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net.state_dict(), sum(epoch_loss)/len(epoch_loss)

class LocalUpdate(object):

    def __init__(self, args, dataset=None, idxs=None, loss_func=None):
        self.args = args
        self.loss_func = nn.MSELoss(reduction='mean')
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.ldr_cal = dataset["cal"]
        self.ldr_pre = dataset["pre"]
        self.ldr_nor = dataset['nor']
        self.idxs = idxs


    def pretrain(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        # pretrain should take MSE to calculate the loss.
        self.loss_func = nn.MSELoss(reduction='mean')

        net.batch_size = 64

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_prop_pretrain(self.ldr_pre, net, self.args, self.loss_func, lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs

        else:
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
                    net.train()
                    net.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                    loss = self.loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.pretrain_iter:
                        break
                
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs
    
    def calib_norm(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        net.batch_size = 50

        if net.model_type == 'transformer':
            max_values = transformer_prop_calib_norm(self.ldr_nor, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys)
            return max_values
        
        else:
            net.eval() # the model will not update in calibration

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for X, y in self.ldr_nor:
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                abs_diff = abs(output - y) 
                result_list = abs_diff.tolist() 

                max_values = np.max(result_list, axis=0)

            return max_values
        
    def calib(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, max_values = None):

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
                abs_diff = abs(output - y).cpu().detach().numpy() # find the distance between pred and gt
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
    
    def calib_bu(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):

        # loss function will be changed to calculate the loss for each attribute in y. (24)
        # self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of calibset loader is 64.
        net.batch_size = 90

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
        cp_dic = dic_loader(self.args)
        cp_value = find_group_info(cp_dic, idx)

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys, cp_value=cp_value)
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
        net.batch_size = 20
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
    
    def cp_rate(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False, cp_value=None):
        self.loss_func = nn.MSELoss(reduction='mean')
        net.batch_size = 20
        m = nn.ReLU()

        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []

        if net.model_type == 'transformer':
            if rho:
                ep_ls, ep_cons_ls, ep_rho, srpre, frpre, crate = transformer_prop_cp_rate(self.ldr_val, net, self.args, self.loss_func, rho=True, cp_value=cp_value)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho, srpre, frpre, crate
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_cp_rate(self.ldr_val, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs

        else:
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            # Repackage and move to CUDA based on model_type
            if net.model_type in ["RNN", "GRU"]:
                # For RNN and GRU, hidden state is a single tensor
                hidden_1 = repackage_hidden(hidden_1).to('cuda')
                hidden_2 = repackage_hidden(hidden_2).to('cuda')

            elif net.model_type == "LSTM":
                # For LSTM, hidden state is a tuple of (h_0, c_0)
                hidden_1 = tuple(h.to('cuda') for h in repackage_hidden(hidden_1))
                hidden_2 = tuple(h.to('cuda') for h in repackage_hidden(hidden_2))

            net.to('cuda')
            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            pre_correction_diff = []
            post_correction_diff = []

            with torch.no_grad():
                for X, y in self.ldr_val:
                    X = X.to('cuda')
                    y = y.to('cuda')

                    net.eval()
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                    # Compute CP rate BEFORE correction
                    for i in range(len(output)):
                        y_pred_i = output[i].detach().cpu().numpy()
                        y_true_i = y[i].detach().cpu().numpy()
                        sqrt_diff_i = y_pred_i - y_true_i
                        pre_correction_diff.append(sqrt_diff_i)

                    # Generate STL constraints
                    property_upper, stl_lib_upper = generate_property_test(X, property_type="upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type="lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, corrected_trace_upper)
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)

                    # # CP-based correction
                    # for i in range(len(output)):
                    #     y_pred_i = output[i].detach().cpu().numpy()
                    #     y_corrected_upper = property_upper[i].detach().cpu().numpy()[:24]
                    #     y_corrected_lower = property_lower[i].detach().cpu().numpy()[:24]

                    #     for j in range(len(y_pred_i)):
                    #         assert y_corrected_upper[j] >= y_corrected_lower[j]
                    #         if y_pred_i[j] > y_corrected_upper[j]:
                    #             cp_lower = y_pred_i[j] - cp_value[j]
                    #             y_pred_i[j] = cp_lower if cp_lower > y_corrected_upper[j] else y_corrected_upper[j]
                    #         elif y_pred_i[j] < y_corrected_lower[j]:
                    #             cp_upper = y_pred_i[j] + cp_value[j]
                    #             y_pred_i[j] = cp_upper if cp_upper < y_corrected_lower[j] else y_corrected_lower[j]

                    #     output[i] = torch.from_numpy(y_pred_i).to(output.device)

                    # Compute CP rate AFTER correction
                    corrected_output = output.detach().cpu().numpy()
                    true_values = y.detach().cpu().numpy()
                    for i in range(len(corrected_output)):
                        sqrt_diff_i = corrected_output[i] - true_values[i]
                        post_correction_diff.append(sqrt_diff_i)

                    # Compute losses
                    
                    pred_loss = self.loss_func(output, y)
                    batch_loss.append(pred_loss.item())
                    batch_cons_loss.append(cons_loss.item())

                    if rho and self.args.property_type == 'constraint':
                        batch_rho.append(
                            1 - torch.count_nonzero(m(corrected_trace_lower - output)) / corrected_trace_lower.numel()
                        )
                        batch_rho.append(
                            1 - torch.count_nonzero(m(output - corrected_trace_upper)) / corrected_trace_upper.numel()
                        )

            # Final epoch metrics
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss) / len(batch_cons_loss))
            if rho:
                epoch_rho.append(sum(batch_rho) / len(batch_rho))

            # Compute CP guarantee rates before and after correction
            s_r_pre, f_r_pre, cp_r_pre = cp_guarantee_ratio_calculation(pre_correction_diff, cp_value=cp_value)
            s_r_post, f_r_post, cp_r_post = cp_guarantee_ratio_calculation(post_correction_diff, cp_value=cp_value)

            print("CP Guarantee Rate BEFORE Correction:", cp_r_pre)
            print("CP Guarantee Rate AFTER Correction:", cp_r_post)

            if rho:
                return (
                    sum(epoch_loss) / len(epoch_loss),
                    sum(epoch_cons_loss) / len(epoch_cons_loss),
                    self.idxs,
                    sum(epoch_rho) / len(epoch_rho), 
                    s_r_pre, f_r_pre, cp_r_pre
                )
            else:
                return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_cons_loss) / len(epoch_cons_loss), self.idxs 


def compute_cluster_id(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf)

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
    cluster_loss = np.full((args.cluster, args.client), np.inf) 

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



def cluster_explore(net, w_glob_keys, lr, args, dataloaders, cp_value, lf = loss_func_cp):

    loss_func_cp = lf
    loss_func = nn.MSELoss(reduction='mean')

    net.batch_size = 64

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
        hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        for name, param in net.named_parameters():
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for iter in range(args.cluster_fine_tune_iter):
            num_updates = 0
            batch_loss = []
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

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return net.state_dict(), sum(epoch_loss)/len(epoch_loss)


class LocalUpdateProp(object):
    """
    federated learning updating class with specification mining.
    """

    def __init__(self, args, dataset=None, idxs=None, loss_func=None):
        self.args = args
        self.loss_func = nn.MSELoss(reduction='mean')
        self.ldr_train = dataset["train_private"] 
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
    
    def calib_norm(self, net, w_glob_keys=None, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):
        """
        Compute alpha in distributed CP with normalization
        """
        net.batch_size = 50

        if net.model_type == 'transformer':
            return transformer_prop_calib_norm(
                self.ldr_nor, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys
            )

        net.eval()
        hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        all_diffs = []

        for X, y in self.ldr_nor:
            hidden_1 = repackage_hidden(hidden_1)
            hidden_2 = repackage_hidden(hidden_2)
            output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
            abs_diff = abs(output - y)
            all_diffs.extend(abs_diff.tolist())

        all_diffs = np.array(all_diffs)
        max_values = np.max(all_diffs, axis=0)
        return max_values

    def calib(self, net, w_glob_keys=None, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, max_values=None):
        """
        Compute normalized conformity scores (c_tuda) for calibration.
        """
        net.batch_size = 90

        if net.model_type == 'transformer':
            return transformer_prop_calib(
                self.ldr_cal, net, self.args, self.loss_func, self.args.max_lr,
                w_glob_keys=w_glob_keys, max_values=max_values
            )

        net.eval()
        hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        c_tuda_list = []
        cal_loss = []

        for X, y in self.ldr_cal:
            hidden_1 = repackage_hidden(hidden_1)
            hidden_2 = repackage_hidden(hidden_2)
            output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

            abs_diff = abs(output - y).cpu().detach().numpy().astype(np.float32)
            abs_diff /= max_values  

            max_value_of_row = np.max(abs_diff, axis=1)
            c_tuda_list.extend(max_value_of_row.tolist())

            loss = self.loss_func(output, y)
            cal_loss.append(loss.item())

        return c_tuda_list

    def train(self, net, w_glob_keys, last=False, dataset_test=None, lf = loss_func_cp, ind=-1, idx=-1, lr=0.001, cp_value=None):

        self.loss_func_cp = lf
        self.loss_func = nn.MSELoss(reduction='mean')
        net.batch_size = 64
        cp_value = cp_value
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_prop_train(self.ldr_train, net, self.args, self.loss_func, lr, w_glob_keys=w_glob_keys, cp_value=cp_value, lf=lf)
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

                batch_loss = []
                batch_cons_loss = []
                for X, y in self.ldr_train:
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    pred_loss = self.loss_func(output, y)

                    pred_loss_cp = self.loss_func_cp(cp_value = cp_value, y_pred = output, y_gt = y)
                    subloss = [loss for sublist in pred_loss_cp for loss in sublist]
                    cp_loss = np.mean([loss**2 for loss in subloss])

                    if self.args.property_type == 'constraint':
                        property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                        corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                        property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                        corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                        cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                    else:
                        raise NotImplementedError

                    loss = pred_loss + cons_loss + cp_loss
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(pred_loss.item())
                    batch_cons_loss.append(cons_loss.item())

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs

    def test(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False):
        
        # loss function for test should always be MSE. 
        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of testset loader is 15.
        net.batch_size = 20

        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []

        if net.model_type == 'transformer':
            if rho == True:
                net, ep_ls, ep_cons_ls, ep_rho= transformer_prop_test(self.ldr_val, net, self.args, self.loss_func, rho=True)
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

            for X, y in self.ldr_val:
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
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, corrected_trace_upper)
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)

                else:
                    raise NotImplementedError
                
                batch_cons_loss.append(cons_loss.item())

                if rho==True:
                    if self.args.property_type == 'constraint':
                        batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_lower - output))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                        batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_upper))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )

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
        net.batch_size = 20

        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []
        num_updates = 0

        if net.model_type == 'transformer':
            if rho == True:
                net, ep_ls, ep_cons_ls, ep_rho= transformer_prop_teacher_test_cp(self.ldr_val, net, self.args, self.loss_func, rho=True)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_teacher_test_cp(self.ldr_val, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs
        
        else:
            net.batch_size = 20
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

    def cp_teacher(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False, cp_value=None):
        self.loss_func = nn.MSELoss(reduction='mean')
        net.batch_size = 20
        m = nn.ReLU()

        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []

        if net.model_type == 'transformer':
            if rho:
                ep_ls, ep_cons_ls, ep_rho, srpre, frpre, srpost, frpost = transformer_prop_cp_teacher(self.ldr_val, net, self.args, self.loss_func, rho=True, cp_value=cp_value)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho, srpre, frpre, srpost, frpost
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_cp_teacher(self.ldr_val, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs

        else:
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            net.to('cuda')
            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            pre_correction_diff = []
            post_correction_diff = []

            with torch.no_grad():
                for X, y in self.ldr_val:
                    X = X.to('cuda')
                    y = y.to('cuda')
                    hidden_1 = [h.to('cuda') for h in repackage_hidden(hidden_1)]
                    hidden_2 = [h.to('cuda') for h in repackage_hidden(hidden_2)]

                    net.eval()
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                    # Compute CP rate BEFORE correction
                    for i in range(len(output)):
                        y_pred_i = output[i].detach().cpu().numpy()
                        y_true_i = y[i].detach().cpu().numpy()
                        sqrt_diff_i = y_pred_i - y_true_i
                        pre_correction_diff.append(sqrt_diff_i)

                    # Generate STL constraints
                    property_upper, stl_lib_upper = generate_property_test(X, property_type="upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type="lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, corrected_trace_upper)

                    # CP-based correction
                    for i in range(len(output)):
                        y_pred_i = output[i].detach().cpu().numpy()
                        y_corrected_upper = property_upper[i].detach().cpu().numpy()[:24]
                        y_corrected_lower = property_lower[i].detach().cpu().numpy()[:24]

                        for j in range(len(y_pred_i)):
                            assert y_corrected_upper[j] >= y_corrected_lower[j]
                            if y_pred_i[j] > y_corrected_upper[j]:
                                cp_lower = y_pred_i[j] - cp_value[j]
                                y_pred_i[j] = cp_lower if cp_lower > y_corrected_upper[j] else y_corrected_upper[j]
                            elif y_pred_i[j] < y_corrected_lower[j]:
                                cp_upper = y_pred_i[j] + cp_value[j]
                                y_pred_i[j] = cp_upper if cp_upper < y_corrected_lower[j] else y_corrected_lower[j]

                        output[i] = torch.from_numpy(y_pred_i).to(output.device)

                    # Compute CP rate AFTER correction
                    corrected_output = output.detach().cpu().numpy()
                    true_values = y.detach().cpu().numpy()
                    for i in range(len(corrected_output)):
                        sqrt_diff_i = corrected_output[i] - true_values[i]
                        post_correction_diff.append(sqrt_diff_i)

                    # Compute losses
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                    pred_loss = self.loss_func(output, y)
                    batch_loss.append(pred_loss.item())
                    batch_cons_loss.append(cons_loss.item())

                    if rho and self.args.property_type == 'constraint':
                        batch_rho.append(
                            1 - torch.count_nonzero(m(corrected_trace_lower - output)) / corrected_trace_lower.numel()
                        )
                        batch_rho.append(
                            1 - torch.count_nonzero(m(output - corrected_trace_upper)) / corrected_trace_upper.numel()
                        )

            # Final epoch metrics
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss) / len(batch_cons_loss))
            if rho:
                epoch_rho.append(sum(batch_rho) / len(batch_rho))

            # Compute CP guarantee rates before and after correction
            s_r_pre, f_r_pre, cp_r_pre = cp_guarantee_ratio_calculation(pre_correction_diff, cp_value=cp_value)
            s_r_post, f_r_post, cp_r_post = cp_guarantee_ratio_calculation(post_correction_diff, cp_value=cp_value)

            print("CP Guarantee Rate BEFORE Correction:", cp_r_pre)
            print("CP Guarantee Rate AFTER Correction:", cp_r_post)

            if rho:
                return (
                    sum(epoch_loss) / len(epoch_loss),
                    sum(epoch_cons_loss) / len(epoch_cons_loss),
                    self.idxs,
                    sum(epoch_rho) / len(epoch_rho), 
                    s_r_pre, f_r_pre, s_r_post, f_r_post
                )
            else:
                return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_cons_loss) / len(epoch_cons_loss), self.idxs
        
    
    def cp_rate(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False, cp_value=None):
        self.loss_func = nn.MSELoss(reduction='mean')
        net.batch_size = 20
        m = nn.ReLU()

        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []

        if net.model_type == 'transformer':
            if rho:
                ep_ls, ep_cons_ls, ep_rho, srpre, frpre, crate = transformer_prop_cp_rate(self.ldr_val, net, self.args, self.loss_func, rho=True, cp_value=cp_value)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho, srpre, frpre, crate
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_cp_rate(self.ldr_val, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs

        else:
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            net.to('cuda')
            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            pre_correction_diff = []
            post_correction_diff = []

            with torch.no_grad():
                for X, y in self.ldr_val:
                    X = X.to('cuda')
                    y = y.to('cuda')
                    hidden_1 = [h.to('cuda') for h in repackage_hidden(hidden_1)]
                    hidden_2 = [h.to('cuda') for h in repackage_hidden(hidden_2)]

                    net.eval()
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                    # Compute CP rate BEFORE correction
                    for i in range(len(output)):
                        y_pred_i = output[i].detach().cpu().numpy()
                        y_true_i = y[i].detach().cpu().numpy()
                        sqrt_diff_i = y_pred_i - y_true_i
                        pre_correction_diff.append(sqrt_diff_i)

                    # Generate STL constraints
                    property_upper, stl_lib_upper = generate_property_test(X, property_type="upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type="lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, corrected_trace_upper)

                    # CP-based correction
                    for i in range(len(output)):
                        y_pred_i = output[i].detach().cpu().numpy()
                        y_corrected_upper = property_upper[i].detach().cpu().numpy()[:24]
                        y_corrected_lower = property_lower[i].detach().cpu().numpy()[:24]

                        for j in range(len(y_pred_i)):
                            assert y_corrected_upper[j] >= y_corrected_lower[j]
                            if y_pred_i[j] > y_corrected_upper[j]:
                                cp_lower = y_pred_i[j] - cp_value[j]
                                y_pred_i[j] = cp_lower if cp_lower > y_corrected_upper[j] else y_corrected_upper[j]
                            elif y_pred_i[j] < y_corrected_lower[j]:
                                cp_upper = y_pred_i[j] + cp_value[j]
                                y_pred_i[j] = cp_upper if cp_upper < y_corrected_lower[j] else y_corrected_lower[j]

                        output[i] = torch.from_numpy(y_pred_i).to(output.device)

                    # Compute CP rate AFTER correction
                    corrected_output = output.detach().cpu().numpy()
                    true_values = y.detach().cpu().numpy()
                    for i in range(len(corrected_output)):
                        sqrt_diff_i = corrected_output[i] - true_values[i]
                        post_correction_diff.append(sqrt_diff_i)

                    # Compute losses
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                    pred_loss = self.loss_func(output, y)
                    batch_loss.append(pred_loss.item())
                    batch_cons_loss.append(cons_loss.item())

                    if rho and self.args.property_type == 'constraint':
                        batch_rho.append(
                            1 - torch.count_nonzero(m(corrected_trace_lower - output)) / corrected_trace_lower.numel()
                        )
                        batch_rho.append(
                            1 - torch.count_nonzero(m(output - corrected_trace_upper)) / corrected_trace_upper.numel()
                        )

            # Final epoch metrics
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss) / len(batch_cons_loss))
            if rho:
                epoch_rho.append(sum(batch_rho) / len(batch_rho))

            # Compute CP guarantee rates before and after correction
            s_r_pre, f_r_pre, cp_r_pre = cp_guarantee_ratio_calculation(pre_correction_diff, cp_value=cp_value)
            s_r_post, f_r_post, cp_r_post = cp_guarantee_ratio_calculation(post_correction_diff, cp_value=cp_value)

            print("CP Guarantee Rate BEFORE Correction:", cp_r_pre)
            print("CP Guarantee Rate AFTER Correction:", cp_r_post)

            if rho:
                return (
                    sum(epoch_loss) / len(epoch_loss),
                    sum(epoch_cons_loss) / len(epoch_cons_loss),
                    self.idxs,
                    sum(epoch_rho) / len(epoch_rho), 
                    s_r_pre, f_r_pre, cp_r_pre
                )
            else:
                return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_cons_loss) / len(epoch_cons_loss), self.idxs 
        
    
    def test_teacher_cp(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False, cp_value=None, Visual=False):

        self.loss_func = nn.MSELoss(reduction='mean')

        # batch size of testset loader is 5.
        net.batch_size = 20

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
                    y_pred_i = output[i].numpy()
                    sqrt_diff_i = (y_pred_i - y[i].numpy())
                    diff_i = (y_pred_i - y[i].numpy())**2
                    batch_diff.append(diff_i)
                    squared_diff.append(sqrt_diff_i)

                attributes = np.array(batch_diff).T
                first_10_of_one_client = [attribute[:] for attribute in attributes]
                s_r, f_r, cp_r= cp_guarantee_ratio_calculation(squared_diff, cp_value=cp_value)

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
        
    def cp_teacher_ct(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False, cp_value = None):

        self.loss_func = nn.MSELoss(reduction='mean')

        net.batch_size = 20

        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []
        num_updates = 0

        if net.model_type == 'transformer':
            if rho == True:
                ep_ls, ep_cons_ls, ep_rho, sr, fr, sp, fp = transformer_prop_cp_teacher(self.ldr_val, net, self.args, self.loss_func, rho=True, cp_value=cp_value)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_cp_teacher(self.ldr_val, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs
        
        else:
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            temp = 0
            for X, y in self.ldr_val:

                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)

                property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                corrected_trace_lower = convert_best_trace(stl_lib_lower, corrected_trace_upper)
                output = corrected_trace_lower

                for i in range(len(output)):
                    y_pred_i = output[i].detach().cpu().numpy()
                    y_corrected_upper = property_upper[i].detach().cpu().numpy()[:24]

                    y_corrected_lower = property_lower[i].detach().cpu().numpy()[:24]

                    for j in range(len(y_pred_i)):
                        assert y_corrected_upper[j] >= y_corrected_lower[j]

                        if y_pred_i[j] > y_corrected_upper[j]:
                            # temp+=1
                            cp_lower = y_pred_i[j] - cp_value[j]
                            if cp_lower > y_corrected_upper[j]:
                                y_pred_i[j] = cp_lower
                            else: 
                                y_pred_i[j] = y_corrected_upper[j]

                        elif y_pred_i[j] < y_corrected_lower[j]:
                            # temp+=1
                            cp_upper = y_pred_i[j] + cp_value[j]
                            if cp_upper < y_corrected_lower[j]:
                                y_pred_i[j] = cp_upper
                            else: 
                                y_pred_i[j] = y_corrected_lower[j]
                        
                        else:
                            assert y_pred_i[j] <= y_corrected_upper[j] and y_pred_i[j] >= y_corrected_lower[j]
                            # y_pred_i[j] = max(y_pred_i[j], y_corrected_upper[j])
                            # temp += 1
                            y_pred_i[j] = y_pred_i[j]
                    
                    y_pred_i_tensor = torch.from_numpy(y_pred_i)
                    output[i] = y_pred_i_tensor


                cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                pred_loss = self.loss_func(output, y)
                # print(temp)

                batch_loss.append(pred_loss.item())
                batch_cons_loss.append(cons_loss.item())

                if rho==True:
                    if self.args.property_type == 'constraint':
                        batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_lower - output))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                        batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_upper))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
            if rho==True:
                epoch_rho.append(sum(batch_rho)/len(batch_rho))
            
        if rho==True:
            return sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs, sum(epoch_rho)/len(epoch_rho)
        else:
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs


