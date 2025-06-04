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

# def new_loss_func(cp_value, y_gt, y_pred):

#     temp = abs(y_pred - y_gt)
#     loss = []
#     data_loss = []
    
#     for i in range(len(temp)):
#         data_loss = []
#         for j in range(len(temp[i])):
#             if temp[i][j] >= cp_value[j]:
#                 att_loss = cp_value[j]
#             else:
#                 att_loss = temp[i][j].item()
#             data_loss.append(att_loss)
        
#         loss.append(data_loss)

#     return loss


def new_cp_loss(cp_value, y_gt, y_pred):

    if isinstance(cp_value, list):
        cp_value = torch.tensor(cp_value, dtype=y_pred.dtype, device=y_pred.device)

    if cp_value.dim() == 1:
        cp_value = cp_value.unsqueeze(0).expand_as(y_pred)

    error = torch.abs(y_pred - y_gt)
    excess = torch.clamp(error - cp_value, min=0.0)
    squared_loss = excess ** 2

    return squared_loss.mean()

def new_cp_loss_transformer(cp_value, y_gt, y_pred):

    if y_gt.dim() == 2:
        y_gt = y_gt.unsqueeze(2)

    if isinstance(cp_value, list):
        cp_value = torch.tensor(cp_value, dtype=y_pred.dtype, device=y_pred.device)

    if cp_value.dim() == 1:
        cp_value = cp_value.view(1, -1, 1).expand_as(y_pred)

    error = torch.abs(y_pred - y_gt)
    capped_loss = torch.where(error < cp_value, error, cp_value)

    return capped_loss.mean()

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
            cp_loss = loss_func_cp(cp_value=cp_value, y_gt=y, y_pred=output)

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

def cluster_explore(net, w_glob_keys, lr, args, dataloaders, cp_value, lf = new_cp_loss, lf_transformer = new_cp_loss_transformer):

    loss_func_cp = lf
    loss_func_transformer = lf_transformer
    loss_func = nn.MSELoss(reduction='mean')

    net.batch_size = 64

    if net.model_type == 'transformer':
        net, avg_ep_loss = transformer_prop_train(dataloaders, net, args, loss_func, lr, w_glob_keys=w_glob_keys, cp_value=cp_value, lf=loss_func_transformer)
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
                cp_loss = loss_func_cp(cp_value=cp_value, y_gt=y, y_pred=output)
                
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

    def train(self, net, w_glob_keys, last=False, dataset_test=None, lf = new_cp_loss, ind=-1, idx=-1, lr=0.001, cp_value=None, lf_transformer = new_cp_loss_transformer):

        self.loss_func_cp = lf
        self.loss_func_cp_transformer = lf_transformer
        self.loss_func = nn.MSELoss(reduction='mean')
        net.batch_size = 64
        cp_value = cp_value
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_prop_train(self.ldr_train, net, self.args, self.loss_func, lr, w_glob_keys=w_glob_keys, cp_value=cp_value, lf=self.loss_func_cp_transformer)
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
                    cp_loss = self.loss_func_cp(cp_value=cp_value, y_gt=y, y_pred=output)

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


