"""
The main file for evaluations.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import numpy as np
import torch
import torch.nn as nn
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id_eval, cluster_id_property, cluster_explore
from utils_training import get_device, to_device, save_model, get_shared_dataset, model_init
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

    if args.mode == "eval":
        model_types = ["{}".format(args.model)]

        print("==============================================================")
        for type in model_types:
            print("Evaluating LogiCP on model (client teacher)", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
                
            model_path = "hdd/saved_models/"

            for c in range(args.client):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Client_'+str(c)+'_'+'{}'.format(args.client)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                print(net_path)
                model = get_eval_model(net_path, type)
                model.eval()
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc= local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                print(loss, cons_loss, idx, rho_perc)

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

        print("==============================================================")
        for type in model_types:
            print("Evaluating LogiCP on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []

            model_path = "hdd/saved_models/"

            cluster_models = []

            for c in range(args.cluster):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Cluster_'+str(c)+'_'+'{}'.format(args.client)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                c_model = get_eval_model(net_path, type)
                c_model.eval()
                cluster_models.append(c_model)

            args.frac = 1
            idxs_users = [i for i in range(args.client)]

            cluster_id = cluster_id_property(cluster_models, client_dataset, args, idxs_users)   # cluster: clients
            client2cluster = get_dict_keys(cluster_id, idxs_users)

            for c in range(args.client):
                net_path = os.path.join(model_path, args.dataset+'_'+type+'_'+args.method+'_Client_'+str(c)+'_'+'{}'.format(args.client)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                model = get_eval_model(net_path, type)
                model.load_state_dict(cluster_models[client2cluster[c]].state_dict())
                model.eval()
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                print(loss, cons_loss, idx, rho_perc)

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

        exit(0)
        print("==============================================================")

        for type in model_types:
            print("Evaluating IFCA on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []

            model_path = f"ct_{args.method}/saved_models/"
            
            net_path = os.path.join(model_path, 'ct_'+type+'_IFCA_epoch_'+'{}.pt'.format(args.cp_epoch))
            glob_model = get_eval_model(net_path, type)
            glob_model.eval()

            for c in range(args.client):
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                print(loss, cons_loss, idx, rho_perc)
            
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


        print("==============================================================")
        for type in model_types:
            print("Evaluating IFCA on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
            
            net_path = os.path.join(model_path, 'ct_'+type+'_IFCA_epoch_'+'{}.pt'.format(args.cp_epoch))
            glob_model = get_eval_model(net_path, type)
            glob_model.eval()

            cluster_models = []
            for c in range(args.cluster):
                net_path = os.path.join(model_path, 'ct_'+type+'_IFCA_Cluster_'+str(c)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                c_model = get_eval_model(net_path, type)
                c_model.eval()
                cluster_models.append(c_model)

            args.frac = 1
            idxs_users = [i for i in range(args.client)]
            cluster_id = compute_cluster_id_eval(cluster_models, client_dataset, args, idxs_users)   # cluster: clients
            client2cluster = get_dict_keys(cluster_id, idxs_users)

            for c in range(args.client):
                net_path = os.path.join(model_path, 'ct_'+type+'_IFCA_Client_'+str(c)+'_epoch_'+'{}.pt'.format(args.cp_epoch))
                model = get_eval_model(net_path, type)
                model.load_state_dict(cluster_models[client2cluster[c]].state_dict())
                model.eval()
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                print(loss, cons_loss, idx, rho_perc)
            
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


        exit(0)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')
