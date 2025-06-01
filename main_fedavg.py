"""
The main file for training and evaluating FedAvg.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import numpy as np
import torch
from torch.utils.data import DataLoader
from IoTData import SequenceDataset
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id, cluster_id_property, cluster_explore, dic_loader_without_specm, cluster_explore_without_sp
from utils_training import get_device, to_device, save_model, get_client_dataset, get_shared_dataset, model_init, get_shared_dataset_1, qq_cp, get_result_file, save_cp_result, get_cluster_result, save_cluster_result, get_shared_dataset_2, save_cp_result_with_sep_type
import sys
import os
import copy
from tqdm import tqdm
from options import args_parser
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN, MultiRegressionLSTM, MultiRegressionGRU, MultiRegressionRNN
from transformer import TimeSeriesTransformer
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import matplotlib.pyplot as plt

# save results to txt log file.
# stdoutOrigin = sys.stdout 


# def main():
#     args = args_parser()
#     args.device = get_device()
#     # sys.stdout = open("log/FedAvg"+str(args.model)+".txt", "a")
    
#     client_dataset = {}
#     for c in range(args.client):
#         client_dataset[c] = {}
#         train_loader_private, trainset_shared, val_loader, test_loader, dataset_len = get_shared_dataset(c, args.dataset)
#         client_dataset[c]["train_private"] = train_loader_private
#         client_dataset[c]["train_shared"] = trainset_shared
#         client_dataset[c]["val"] = val_loader
#         client_dataset[c]["test"] = test_loader
#         client_dataset[c]["len"] = dataset_len

#     print("Loaded client dataset.")

#     ############################
#     # loading shared model
#     glob_model, clust_weight_keys = model_init(args)
#     glob_model = to_device(glob_model, args.device)
#     net_keys = [*glob_model.state_dict().keys()]

#     ############################
#     # generate list of local models for each user
#     local_weights = {}      # client ID: weight: val
#     for c in range(args.client):
#         w_local_dict = {}
#         for key in glob_model.state_dict().keys():
#             w_local_dict[key] = glob_model.state_dict()[key]
#         local_weights[c] = w_local_dict
#     print("Loaded client models.")

#     ############################
#     # simple training algorithm
#     if "train" in args.mode:
#         train_loss = []         # glob train loss 
#         eval_loss = []          # glob eval loss 

#         # one communication round 
#         for ix_epoch in range(1, args.epoch+1): 
#             local_loss = []     # loss for all clients in this round
#             glob_weight = {}    # glob weights in this round
#             total_len = 0       # total dataset length

#             m = max(int(args.frac * args.client), 1) # a fraction of all devices
#             if ix_epoch == args.epoch: # in the last round, all users are selected 
#                 m = args.client
#             idxs_users = np.random.choice(range(args.client), m, replace=False)          # select devices for this round
#             print(f"Communication round  {ix_epoch}\n---------")
#             print("Selected:", idxs_users)
            
#             try:
#                 for c_ind, c in enumerate(idxs_users):  # client update iterations
#                     local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
#                     net_local = copy.deepcopy(glob_model)   # init local net
#                     w_local = net_local.state_dict()
#                     net_local.load_state_dict(w_local)
#                     last = (ix_epoch == args.epoch)

#                     w_local, loss, idx = local.train(net=net_local.to(args.device), 
#                                                      idx=c, w_glob_keys=None, 
#                                                      lr=args.max_lr, last=last)
                    
#                     # print(f"loss is {loss}")
#                     local_loss.append(copy.deepcopy(loss))
#                     total_len += client_dataset[c]["len"][0]

#                     # exit(0)

#                     # update shared glob weights
#                     if len(glob_weight) == 0:    # first update
#                         glob_weight = copy.deepcopy(w_local)
#                         for key in glob_model.state_dict().keys():
#                             glob_weight[key] = glob_weight[key]*client_dataset[c]["len"][0]
#                             local_weights[c][key] = w_local[key]
#                     else:
#                         for key in glob_model.state_dict().keys():
#                             glob_weight[key] += w_local[key]*client_dataset[c]["len"][0]
#                             local_weights[c][key] = w_local[key]

#                     print(ix_epoch, idx, loss)
#                     # exit(0)

#             except KeyboardInterrupt:
#                 break
            
#             for k in glob_model.state_dict().keys():    # get weighted average for global weights
#                 glob_weight[k] = torch.div(glob_weight[k], total_len)
            
#             w_local = glob_model.state_dict()
#             for k in glob_weight.keys():
#                 w_local[k] = glob_weight[k]
            
#             if args.epoch != ix_epoch:
#                 glob_model.load_state_dict(glob_weight)

#             loss_avg = sum(local_loss) / len(local_loss)
#             train_loss.append(loss_avg)

#             for c in range(args.client):
#                 glob_model.load_state_dict(local_weights[c])
#                 local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c) 
#                 w_local, loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)  
#                 local_loss.append(copy.deepcopy(loss))
#             eval_loss.append(sum(local_loss)/len(local_loss))
#             print(sum(local_loss)/len(local_loss))

#         model_path = "hdd/saved_models/"
#         save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_FedAvg", ix_epoch)

#     else:
#         print("FedAvg main: Mode should be set to `train`.")

def get_dict_keys(cluster_id, idxs_users):
    d = {}
    for i in idxs_users:
        for key, val in cluster_id.items():
            if i in val:
                d[i] = key
    return d

def convert_np_int_to_int(item):
    if isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, list):
        return [convert_np_int_to_int(i) for i in item]
    return item

def main():
    args = args_parser()
    args.device = get_device()
    # sys.stdout = open("log/FedAvg"+str(args.model)+".txt", "a")
    
    client_dataset = {}

    for c in range(args.client): # 100
        client_dataset[c] = {}
        train_loader_private, trainset_shared, calib_loader_private, pre_loader_private, val_loader, test_loader, dataset_len = get_shared_dataset_2(c, args.dataset)
        client_dataset[c]["train_private"] = train_loader_private # dataloader for the private training data
        client_dataset[c]["train_shared"] = trainset_shared # the shared data in training 
        client_dataset[c]["val"] = val_loader # dataloader for the validation data
        client_dataset[c]["test"] = test_loader # dataloader for the test data
        client_dataset[c]["len"] = dataset_len # [len(train_dataset), len(calib_dataset), len(pre_dataset), len(val_dataset), len(test_dataset), len(shared_dataset)] # 90, 90, 90, 20, 21, 90

        client_dataset[c]["cal"] = calib_loader_private # dataloader for the calibration data
        client_dataset[c]["pre"] = pre_loader_private # dataloader for the pretrain data


    print("Loaded client dataset.")

    ############################
    # loading shared model
    glob_model, clust_weight_keys = model_init(args)
    glob_model = to_device(glob_model, args.device)
    net_keys = [*glob_model.state_dict().keys()]

    ############################
    # generate list of local models for each user
    local_weights = {}      # client ID: weight: val
    for c in range(args.client):
        w_local_dict = {}
        for key in glob_model.state_dict().keys():
            w_local_dict[key] = glob_model.state_dict()[key]
        local_weights[c] = w_local_dict
    print("Loaded client models.")

    cq = round(7/10, 2)
    dq = round(79/100, 2)

    ############################
    # pretrain the model and do calibration in the last round. 
    if args.mode == "pretrain_calib":

        # loading vanila model
        glob_model, clust_weight_keys = model_init(args)
        glob_model = to_device(glob_model, args.device)
        net_keys = [*glob_model.state_dict().keys()]

        ############################
        # generate list of local models for each user
        # for all 100 clients, each copy the global model
        local_weights = {}      # client ID: weight: val
        for c in range(args.client):
            # print(f"length of clients is {args.client}")
            w_local_dict = {}
            for key in glob_model.state_dict().keys():
                w_local_dict[key] = glob_model.state_dict()[key]
            local_weights[c] = w_local_dict
        
        print(f"==========================================")
        print("Loaded vanilla client models.")
        print(f"==========================================")

        # simple training without clusters
        train_loss = []         # glob train loss 
        eval_loss = []          # glob eval loss 

        # one communication round 
        for ix_epoch in range(1, args.epoch+1): 
            local_loss = []         # loss for all clients in this round
            glob_weight = {}        # glob weights in this round
            total_len = 0           # total dataset length
            global_quantile = []      # clients p-th quantile value in calibration 

            m = max(int(args.frac * args.client), 1) # a fraction of all devices
            if ix_epoch == args.epoch: # in the last round, all users are selected 
                m = args.client
            idxs_users = np.random.choice(range(args.client), m, replace=False)          # select devices for this round
            
            try:
                print(f"Communication round  {ix_epoch}\n---------")
                print("Selected:", idxs_users)

                # client update iterations
                for c_ind, c in enumerate(idxs_users):

                    local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                    net_local = copy.deepcopy(glob_model)   # init local net
                    w_local = net_local.state_dict()
                    net_local.load_state_dict(w_local)
                    # last = (ix_epoch == args.epoch)
                    last = False

                    w_local, loss, idx = local.pretrain(net=net_local.to(args.device), idx=c, w_glob_keys=None, lr=args.max_lr, last=last) 
                    local_loss.append(copy.deepcopy(loss))
                    total_len += client_dataset[c]["len"][0]

                    # update shared glob weights
                    if len(glob_weight) == 0:    # first update
                        glob_weight = copy.deepcopy(w_local)
                        for key in glob_model.state_dict().keys():
                            glob_weight[key] = glob_weight[key]*client_dataset[c]["len"][0]
                            local_weights[c][key] = w_local[key]
                    else:
                        for key in glob_model.state_dict().keys():
                            glob_weight[key] += w_local[key]*client_dataset[c]["len"][0]
                            local_weights[c][key] = w_local[key]

                    print(ix_epoch, idx, loss)
                
                if ix_epoch == args.epoch:

                    print(f"==========================================")
                    print(f"calibration starts here.")
                    print(f"epoch checks expect {args.epoch} is {ix_epoch}")
                    print(f"==========================================")

                    for c_ind, c in enumerate(idxs_users):  # client update iterations
                        client_quantile = []
                        local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                        net_local = copy.deepcopy(glob_model)   # init local net
                        w_local = net_local.state_dict()
                        net_local.load_state_dict(w_local)
                        last = False

                        result_list = local.calib(net=net_local.to(args.device), 
                                                     idx=c, w_glob_keys=None, 
                                                     lr=args.max_lr, last=last)
                        
                        # print(f"len checks 1 expect 24 is {len(result_list)}")
                        # print(f"len checks 2 expect 64 is {len(result_list[0])}")
                        # exit(0)
                        
                        for i in range(len(result_list)):
                            # print(f"check 1 expect 24 is {len(y_variable_list)}")
                            y_variable_list_sorted = np.sort(result_list[i])
                            # print(f"check 2 expect 64 is {len(y_variable_list_sorted)}")
                            y_variable_quantile_value = np.quantile(y_variable_list_sorted, dq)
                            client_quantile.append(y_variable_quantile_value)
                        
                        # print(f"check 3 expect 64 is {len(client_quantile)}")
                        
                        global_quantile.append(client_quantile)

            except KeyboardInterrupt:
                break
            
            for k in glob_model.state_dict().keys():    # get weighted average for global weights
                glob_weight[k] = torch.div(glob_weight[k], total_len)
            
            w_local = glob_model.state_dict()
            for k in glob_weight.keys():
                w_local[k] = glob_weight[k]
            
            if args.epoch != ix_epoch:
                glob_model.load_state_dict(glob_weight)

            loss_avg = sum(local_loss) / len(local_loss)
            train_loss.append(loss_avg)

            for c in range(args.client):
                glob_model.load_state_dict(local_weights[c])
                if args.mode == "pretrain_calib":           
                    glob_model.load_state_dict(local_weights[c])
                    local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c) 
                    w_local, loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)  
                    local_loss.append(copy.deepcopy(loss))
            eval_loss.append(sum(local_loss)/len(local_loss))
            print(f"the average test loss is {round(sum(local_loss)/len(local_loss), 3)}\n---------------------------------")

        model_path = "hdd_avg/saved_pretrain_models_cp_only/"
        if args.mode == "pretrain_calib":
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_"+str(args.property_type)+"_glob", ix_epoch)

        cp_size = []
        global_quantile_array = np.array(global_quantile)

        save_cp_result_with_sep_type(args = args, client_dataset = client_dataset, global_quantile_array = global_quantile_array, glob_model = glob_model, dq = dq, cq = cq)

        print("============================")
        print("Calibration result saved, pretrain with calib ends.")

    ############################
    # simple training algorithm
    elif "train_cp" in args.mode and args.cluster > 0:

        ############################
        # loading pretrained model
        glob_model, clust_weight_keys = model_init(args)
        glob_model = to_device(glob_model, args.device)
        # print(clust_weight_keys)
        # eixt(0)
        net_keys = [*glob_model.state_dict().keys()]

        """
        
        20 clients, 90 data, with 4, 76
        
        """

        cq = round(14/20, 2) # quantile for client
        dq = round(76/90, 2) # quantile for local dataset
        ############################
        # generate list of local models for each user

        # for all 100 clients, each copy the pretrained global model
        local_weights = {}      # client ID: weight: val
        for c in range(args.client):
            w_local_dict = {}
            for key in glob_model.state_dict().keys():
                w_local_dict[key] = clust_weight_keys[key]
            local_weights[c] = w_local_dict

        print("Loaded pretrained client models.")

        cluster_weights = {}
        for cluster in range(args.cluster): # 10
            cluster_weights[cluster] = {}

        cluster_models = {}
        for cluster in range(args.cluster):
            cluster_models[cluster] = copy.deepcopy(glob_model)

        train_loss = [] # glob train loss 
        eval_loss = []  # glob eval loss 
        global_quantile = []

        epoch_num = 0
        if args.mode == 'pretrain_calib':
            epoch_num = args.epoch
        elif args.mode == 'train_cp':
            epoch_num = args.cp_epoch
        
        print("epoch number checks: {}".format(epoch_num))

        # one communication round 
        for ix_epoch in range(1, args.epoch+1):
            local_loss = []           # loss for all clients in this round
            glob_weight = {}          # glob weights in this round
            total_len = 0             # total dataset length
            global_quantile = []      # clients p-th quantile value in calibration 
            cluster_len = [0] * args.cluster

            m = max(int(args.frac * args.client), 1) # a fraction of all devices
            idxs_users = np.random.choice(range(args.client), m, replace=False)          # select devices for this round
            print("Selected:", idxs_users)

            last = (ix_epoch == epoch_num)
            if last: 
                args.frac = 1

            cp_dic = dic_loader_without_specm(args)
            cluster_id = {int(group): clients_data["clients"] for group, clients_data in cp_dic.items()}
            client2cluster = {client: group for group, clients in cluster_id.items() for client in clients}

            for c_ind, c in enumerate(idxs_users):  # client update iterations

                """

                Note: the args.batch should be revised, if for test and val.

                """
                local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                net_local = copy.deepcopy(glob_model)   # init local net
                w_local = net_local.state_dict()
                net_local.load_state_dict(w_local)
                for k in local_weights[c].keys():
                    if k not in clust_weight_keys:
                        w_local[k] = local_weights[c][k]
                last = False

                w_local, loss, idx = local.train(net=net_local.to(args.device), 
                                                    idx=c, w_glob_keys=None, 
                                                    lr=args.max_lr, last=last)
                
                local_loss.append(copy.deepcopy(loss))
                total_len += client_dataset[c]["len"][0]
                cluster_len[client2cluster[c]] += client_dataset[c]["len"][0]

                # update shared cluster weights
                if len(cluster_weights[client2cluster[c]]) == 0:
                    cluster_temp = copy.deepcopy(w_local)
                    for k in clust_weight_keys:
                        cluster_weights[client2cluster[c]][k] = cluster_temp[k]*client_dataset[c]["len"][0]
                else:
                    for k in clust_weight_keys:
                        cluster_weights[client2cluster[c]][k] += w_local[k]*client_dataset[c]["len"][0]

                # update shared glob weights
                if len(glob_weight) == 0:    # first update
                    glob_weight = copy.deepcopy(w_local)
                    for key in glob_model.state_dict().keys():
                        glob_weight[key] = glob_weight[key]*client_dataset[c]["len"][0]
                        local_weights[c][key] = w_local[key]
                else:
                    for key in glob_model.state_dict().keys():
                        glob_weight[key] += w_local[key]*client_dataset[c]["len"][0]
                        local_weights[c][key] = w_local[key]

                print(idx, round(loss, 3))
            
            # get weighted average global weights
            for k in glob_model.state_dict().keys():    
                glob_weight[k] = torch.div(glob_weight[k], total_len)

            # update locals 
            w_local = glob_model.state_dict()
            for k in glob_weight.keys():
                w_local[k] = glob_weight[k]
            
            # update globals
            if epoch_num != ix_epoch:
                glob_model.load_state_dict(glob_weight)

            # update clusters 
            for cluster in range(args.cluster):
                for k in clust_weight_keys:
                    cluster_weights[cluster][k] = torch.div(cluster_weights[cluster][k], cluster_len[cluster])

            for cluster in range(args.cluster):
                w_local = net_local.state_dict()
                for k in glob_weight.keys():
                    if k not in clust_weight_keys:
                        w_local[k] = glob_weight[k]
                    else:
                        w_local[k] = cluster_weights[cluster][k]
                cluster_models[cluster].load_state_dict(w_local)
    
            if args.mode == "train_cp":          # cluster model fine-tune
                for cluster in range(args.cluster):
                    net_local = copy.deepcopy(cluster_models[cluster]) 
                    w_local = net_local.state_dict()
                    net_local.load_state_dict(w_local)

                    data_x = [client_dataset[i]["train_shared"][0] for i in cluster_id[cluster]]
                    data_y = [client_dataset[i]["train_shared"][1] for i in cluster_id[cluster]]
                    cluster_data_x = np.concatenate(data_x, axis=0)
                    cluster_data_y = np.concatenate(data_y, axis=0)
                    cluster_train_dataset = SequenceDataset(cluster_data_x, cluster_data_y)
                    cluster_train_loader = DataLoader(cluster_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

                    cp_dic = dic_loader_without_specm(args)
                    cp_value = cp_dic['{}'.format(cluster)]["cp_value"]

                    w_local, loss = cluster_explore_without_sp(net=net_local.to(args.device), 
                                                    w_glob_keys=clust_weight_keys, 
                                                    lr=args.max_lr, args=args, 
                                                    dataloaders=cluster_train_loader, cp_value=cp_value)
                    for key in glob_weight.keys():
                        cluster_weights[cluster][key] = w_local[key]
                    cluster_models[cluster].load_state_dict(w_local)
                    print("cluster fine-tune:", ix_epoch, cluster, loss)
        
            for c_ind, c in enumerate(idxs_users):  # client update iterations
                client_quantile = []
                local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                net_local = copy.deepcopy(glob_model)   # init local net
                w_local = net_local.state_dict()
                net_local.load_state_dict(w_local)
                last = True

                result_list = local.calib(net=net_local.to(args.device), 
                                                idx=c, w_glob_keys=None, 
                                                lr=args.max_lr, last=last)
                
                for i in range(len(result_list)):
                    # print(f"check 1 expect 24 is {len(y_variable_list)}")
                    y_variable_list_sorted = np.sort(result_list[i])
                    # print(f"check 2 expect 64 is {len(y_variable_list_sorted)}")
                    y_variable_quantile_value = np.quantile(y_variable_list_sorted, dq)
                    client_quantile.append(y_variable_quantile_value)
                
                # print(f"check 3 expect 24 is {len(client_quantile)}")
                
                global_quantile.append(client_quantile)
            
            global_quantile_array = np.array(global_quantile)
            save_cp_result_with_sep_type(args = args, client_dataset = client_dataset, global_quantile_array = global_quantile_array, glob_model = glob_model, dq = dq, cq = cq)

            loss_avg = sum(local_loss)/len(local_loss)
            train_loss.append(loss_avg)

            for c in idxs_users:  # client update iterations
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                net_local = copy.deepcopy(cluster_models[client2cluster[c]])
                w_local = net_local.state_dict()
                net_local.load_state_dict(w_local)
                w_local, loss, cons_loss, idx = local.test(net=net_local.to(args.device), 
                                                    idx=c, w_glob_keys=None, 
                                                    lr=args.max_lr, last=last)   
                local_loss.append(copy.deepcopy(loss))

            eval_loss.append(sum(local_loss)/len(local_loss))
            print("===============")
            print(sum(local_loss)/len(local_loss))
            print("===============")

            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Mean_second", sum(local_loss)/len(local_loss))
            print("Error bar:", error)
            print()

        model_path = "hdd_avg/saved_models_cp_only/"
        glob_model.load_state_dict(glob_weight)
        save_model(model_path, glob_model, 
                str(args.dataset)+"_"+str(args.local_updates)+"_"+str(args.model)+"_"+args.method+"_"+str(args.property_type), 
                ix_epoch)
        
        for c in range(args.cluster):
            glob_model.load_state_dict(cluster_weights[c])
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_"+args.method+"_Cluster_"+str(c), ix_epoch)
        
        for c in range(args.client):
            glob_model.load_state_dict(local_weights[c])
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_"+args.method+"_Client_"+str(c), ix_epoch)

    cp_path = "hdd_avg/saved_cluster_result_cp_only/"
    save_cp_result_with_sep_type(args = args, client_dataset = client_dataset, global_quantile_array = global_quantile_array, glob_model = glob_model, dq = dq, cq = cq)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')
        # sys.stdout.close()
        # sys.stdout = stdoutOrigin