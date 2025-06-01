"""
The main file for evaluations.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import numpy as np
import torch
import torch.nn as nn
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id_eval, cluster_id_property, cluster_explore, dic_loader, find_group_info
from utils_training import get_device, to_device, save_model, get_client_dataset, get_shared_dataset, model_init, get_shared_dataset_1, get_shared_dataset_2
import os
import copy
from options import args_parser
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN, MultiRegressionLSTM, MultiRegressionGRU, MultiRegressionRNN
from transformer import TimeSeriesTransformer
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def get_eval_model(net_path, net_type):
    if net_type == 'GRU':
        model = ShallowRegressionGRU(input_dim=2, batch_size=64, time_steps=96, sequence_len=24, hidden_dim=16)
    elif net_type == 'LSTM':
        model = ShallowRegressionLSTM(input_dim=2, batch_size=64, time_steps=96, sequence_len=24, hidden_dim=16)
    elif net_type == 'RNN':
        model = ShallowRegressionRNN(input_dim=2, batch_size=64, time_steps=96, sequence_len=24, hidden_dim=16)
    elif net_type == 'transformer':
        model = TimeSeriesTransformer()
    model = to_device(model, 'cuda')
    model.load_state_dict(torch.load(net_path))
    
    return model


def get_dict_keys(cluster_id, idxs_users):
    d = {}
    for i in idxs_users:
        for key, val in cluster_id.items():
            if i in val:
                d[i] = key
    return d

def sort_client_losses(local_losses):
    """
    Sorts client evaluation losses and returns a list of tuples: (client_index, loss), sorted by loss ascending.
    """
    sorted_losses = sorted(enumerate(local_losses), key=lambda x: x[1])
    return sorted_losses

def new_loss_func(cp_value, y_gt, y_pred):

    m = nn.ReLU()
    temp = (y_pred - y_gt)
    loss = m(0, m(temp, cp_value))

    return loss

def main():
    args = args_parser()
    args.device = get_device()
    
    client_dataset = {}

    # for c in range(args.client): # 100
    #     client_dataset[c] = {}
    #     train_loader_private, trainset_shared, cal_loader_private, cal_shared, val_loader, test_loader, dataset_len = get_shared_dataset_1(c, args.dataset)
        
    #     client_dataset[c]["train_private"] = train_loader_private # private dataset loader
    #     client_dataset[c]["train_shared"] = trainset_shared
    #     client_dataset[c]["val"] = val_loader
    #     client_dataset[c]["test"] = test_loader
    #     client_dataset[c]["len"] = dataset_len
    #     client_dataset[c]["cal"] = cal_loader_private
    #     client_dataset[c]["calib_shared"] = cal_shared
    
    for c in range(args.client): # 100
        client_dataset[c] = {}
        train_loader_private, trainset_shared, calib_loader_private, pre_loader_private, nor_loader_private, val_loader, test_loader, dataset_len = get_shared_dataset_2(c, args.dataset)
        client_dataset[c]["train_private"] = train_loader_private # dataloader for the private training data
        client_dataset[c]["train_shared"] = trainset_shared # the shared data in training 
        client_dataset[c]["val"] = val_loader # dataloader for the validation data
        client_dataset[c]["test"] = test_loader # dataloader for the test data
        client_dataset[c]["nor"] = test_loader # dataloader for the test data
        client_dataset[c]["len"] = dataset_len # [len(train_dataset), len(calib_dataset), len(pre_dataset), len(val_dataset), len(test_dataset), len(shared_dataset)] # 90, 90, 90, 20, 21, 90

        client_dataset[c]["cal"] = calib_loader_private # dataloader for the calibration data
        client_dataset[c]["pre"] = pre_loader_private # dataloader for the pretrain data
    
    print("Loaded client dataset.")

    ############################
    # evaluation on fhwa dataset.
    # args.client = 100

    if args.mode == "eval" and args.method == 'FedSTL':
        model_types = ["RNN"]

        cp_dic = dic_loader(args)

        cluster_id = {int(group): clients_data["clients"] for group, clients_data in cp_dic.items()}
        client2cluster = {client: group for group, clients in cluster_id.items() for client in clients}

        print("==============================================================")
        for type in model_types:
            print("Evaluating FedSTL on model (client teacher)", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
            s_n = 0
            f_n = 0

            if args.dataset == 'fhwa':
                model_path = "hdd/saved_models/"
            elif args.dataset == 'ct':
                model_path = f"hdd/saved_models/"

            for c in range(args.client):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Client_'+str(c)+'_'+'{}'.format(args.client)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                print(net_path)
                model = get_eval_model(net_path, type)
                model.eval()
                cp_value = find_group_info(cp_dic, c)
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc, sr_pre, fr_pre, cp_r_pre = local.cp_rate(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True, cp_value=cp_value)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                print(loss, cons_loss, idx, rho_perc)
                s_n += sr_pre
                f_n += fr_pre
            
            # Sort and display client losses
            sorted_losses = sort_client_losses(local_loss)
            total_loss = []
            index_list = []
            print("Sorted client losses (index, loss):")
            for idx, loss in sorted_losses:
                print(f"Client {idx}: Loss = {loss:.4f}")
                total_loss.append(loss)
                index_list.append(idx)
            
            print(np.mean(total_loss[:70]))
            print(np.mean(total_loss[:60]))
            print(np.mean(total_loss[:80]))
            print(np.mean(total_loss[:50]))
            print(np.mean(total_loss[:100]))
            print(index_list[:100])
            print(len(index_list[:100]))

            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()

            print(s_n / (s_n + f_n))

        # exit(0)
        print("==============================================================")
        for type in model_types:
            print("Evaluating FedSTL on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
            sn = 0
            fn = 0

            if args.dataset == 'fhwa':
                model_path = "hdd/saved_models/"
            elif args.dataset == 'ct':
                model_path = f"hdd/saved_models/"

            cluster_models = []

            for c in range(args.cluster):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Cluster_'+str(c)+'_'+'{}'.format(args.client)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                c_model = get_eval_model(net_path, type)
                c_model.eval()
                cluster_models.append(c_model)

            args.frac = 1
            idxs_users = [i for i in range(args.client)]

            cluster_id = cluster_id_property(cluster_models, client_dataset, args, idxs_users)
            client2cluster = get_dict_keys(cluster_id, idxs_users)

            for c in range(args.client):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Client_'+str(c)+'_'+'{}'.format(args.client)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                model = get_eval_model(net_path, type)
                model.load_state_dict(cluster_models[client2cluster[c]].state_dict())
                model.eval()
                cp_value = find_group_info(cp_dic, c)
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc, sr, fr, cpr = local.cp_rate(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True, cp_value=cp_value)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                print(loss, cons_loss, idx, rho_perc)
                sn += sr
                fn += fr

            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()

            print(sn / (sn + fn))
    
    exit(0)
    if args.mode == "eval" and args.method == 'IFCA':


        print(f"evaluating the IFCA.")
        print()
        model_types = ["LSTM"]

        cp_dic = dic_loader(args)
        # print(cp_dic)

        cluster_id = {int(group): clients_data["clients"] for group, clients_data in cp_dic.items()}
        client2cluster = {client: group for group, clients in cluster_id.items() for client in clients}

        print("==============================================================")
        for type in model_types:
            print("Evaluating FedSTL on model (client teacher)", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
            s_n = 0
            f_n = 0

            if args.dataset == 'fhwa':
                model_path = "hdd/saved_models/"
            elif args.dataset == 'ct':
                model_path = f"{args.dataset}_{args.method}/saved_models/"

            for c in range(args.client):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Client_'+str(c)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                print(net_path)
                model = get_eval_model(net_path, type)
                model.eval()
                cp_value = find_group_info(cp_dic, c)
                local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc, sr, fr, rr = local.cp_rate(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True, cp_value=cp_value)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                print(loss, cons_loss, idx, rho_perc)
                s_n += sr
                f_n += fr
            
            # Sort and display client losses
            sorted_losses = sort_client_losses(local_loss)
            total_loss = []
            index_list = []
            print("Sorted client losses (index, loss):")
            for idx, loss in sorted_losses:
                print(f"Client {idx}: Loss = {loss:.4f}")
                total_loss.append(loss)
                index_list.append(idx)
            
            print(np.mean(total_loss[:70]))
            print(np.mean(total_loss[:60]))
            print(np.mean(total_loss[:80]))
            print(np.mean(total_loss[:50]))
            print(np.mean(total_loss[:100]))
            print(index_list[:100])
            print(len(index_list[:100]))

            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()

            print(s_n / (s_n + f_n))

        # exit(0)
        print("==============================================================")
        for type in model_types:
            print("Evaluating FedSTL on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
            sn = 0
            fn = 0

            if args.dataset == 'fhwa':
                model_path = "hdd/saved_models/"
            elif args.dataset == 'ct':
                model_path = f"{args.dataset}_{args.method}/saved_models/"

            cluster_models = []

            for c in range(args.cluster):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Cluster_'+str(c)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                c_model = get_eval_model(net_path, type)
                c_model.eval()
                cluster_models.append(c_model)

            args.frac = 1
            idxs_users = [i for i in range(args.client)]
            # print(idxs_users)
            # exit(0)
            cluster_id = {int(group): clients_data["clients"] for group, clients_data in cp_dic.items()}
            client2cluster = {client: group for group, clients in cluster_id.items() for client in clients}

            for c in range(args.client):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Client_'+str(c)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                model = get_eval_model(net_path, type)
                model.load_state_dict(cluster_models[client2cluster[c]].state_dict())
                model.eval()
                cp_value = find_group_info(cp_dic, c)
                local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc, sr, fr, rr = local.cp_rate(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True, cp_value=cp_value)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                print(loss, cons_loss, idx, rho_perc)
                sn += sr
                fn += fr

            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()

            print(sn / (sn + fn))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')