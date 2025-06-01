"""
The main file for training and evaluating FedSTL format
with options to compare with other benchmarks.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from IoTData import SequenceDataset
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id, cluster_id_property, cluster_explore, dic_loader, find_group_info
from utils_training import get_device, to_device, save_model, get_client_dataset, get_shared_dataset, model_init, get_shared_dataset_1, get_shared_dataset_2, get_result_file, save_cp_result, get_cluster_result, save_cluster_result, save_cp_result_with_sep_type
import copy
from tqdm import tqdm
from options import args_parser
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN, MultiRegressionLSTM, MultiRegressionGRU, MultiRegressionRNN
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import matplotlib.pyplot as plt

## save results to txt log file.
# stdoutOrigin=sys.stdout


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device
    # sys.stdout = open("log/FedSTL"+str(args.model)+".txt", "a")
    
    client_dataset = {}
    for c in range(args.client): # 100
        client_dataset[c] = {}
        train_loader_private, trainset_shared, calib_loader_private, pre_loader_private, nor_loader_private, val_loader, test_loader, dataset_len = get_shared_dataset_2(c, args.dataset)
        client_dataset[c]["train_private"] = train_loader_private # dataloader for the private training data
        client_dataset[c]["train_shared"] = trainset_shared # the shared data in training 
        client_dataset[c]["val"] = val_loader # dataloader for the validation data
        client_dataset[c]["test"] = test_loader # dataloader for the test data
        client_dataset[c]["nor"] = nor_loader_private # dataloader for the normalization data for calib
        client_dataset[c]["len"] = dataset_len # [len(train_dataset), len(calib_dataset), len(pre_dataset), len(val_dataset), len(test_dataset), len(shared_dataset)] # 90, 90, 90, 20, 21, 90

        client_dataset[c]["cal"] = calib_loader_private # dataloader for the calibration data
        client_dataset[c]["pre"] = pre_loader_private # dataloader for the pretrain data

    print("Loaded client dataset.")

    # 10 clients with 90 data
    if args.client == 1:
        dq = 81
        cq = 1
    # elif args.client == 100: 
    #     dq = round(79/90, 2)
    else: 
        dq = 79
        cq = 7
        # dq = 68
        # cq = 8
    ############################
    # pretrain the model and do calibration in the last round. 
    if "train" in args.mode and args.mode == "pretrain_calib":

        # loading vanila model
        glob_model, clust_weight_keys = model_init(args)
        glob_model = to_device(glob_model, args.device)
        net_keys = [*glob_model.state_dict().keys()]

        ############################
        # generate list of local models for each user
        # for all 100 clients, each copy the global model
        local_weights = {}      # client ID: weight: val
        for c in range(args.client):
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
                    if args.mode == "pretrain_calib":
                        local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                    
                    net_local = copy.deepcopy(glob_model)   # init local net
                    w_local = net_local.state_dict()
                    net_local.load_state_dict(w_local)
                    # last = (ix_epoch == args.epoch)
                    last = False

                    # set local.train as local.pretrain temporarily
                    # w_local, loss, idx = local.pretrain(net=net_local.to(args.device), idx=c, w_glob_keys=None, lr=args.max_lr, last=last) # train local
                    w_local, loss, idx = local.pretrain(net=net_local.to(args.device), idx=c, w_glob_keys=None, lr=args.max_lr, last=last) # train local
                    local_loss.append(copy.deepcopy(loss))
                    total_len += client_dataset[c]["len"][0] # length of each client dataset is 90
                    # print(total_len)
                    # exit(0)

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

                    # print(f"temp exit at the end of pretrain.")
                    # exit(0)

                    print(f"==========================================")
                    print(f"calibration starts here.")
                    print(f"epoch checks expect {args.epoch} is {ix_epoch}")
                    print(f"==========================================")

                    temp = []

                    for c_ind, c in enumerate(idxs_users):  # client update iterations
                        client_quantile = []
                        local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                        net_local = copy.deepcopy(glob_model)   # init local net
                        w_local = net_local.state_dict()
                        net_local.load_state_dict(w_local)
                        last = False

                        max_values_list = local.calib_norm(net=net_local.to(args.device), 
                                                     idx=c, w_glob_keys=None, 
                                                     lr=args.max_lr, last=last)
                        
                        # if c == idxs_users[0]:
                        #     print(f"max is {max_values_list}")

                        c_tuda_list = local.calib(net=net_local.to(args.device), 
                                                     idx=c, w_glob_keys=None, 
                                                     lr=args.max_lr, last=last, max_values = max_values_list)

                        # print(f"length of result list is {len(result_list)}")
                        # for i in range(len(c_tuda_list)):
                            # print(f"check 1 expect 24 is {len(y_variable_list)}")
                        c_tuda_list_sorted = np.sort(c_tuda_list, axis=0)
                        # print(f"shape of c tuda is {c_tuda_list_sorted.shape}")
                        # print(f"c_tuda_list is {c_tuda_list_sorted}")
                        # print(np.array(c_tuda_list_sorted).tolist())
                        # print(f"check 2 expect 90 is {len(c_tuda_list_sorted)}")
                        # exit(0)
                        c_tuda_list_quantile_value = np.array(c_tuda_list_sorted).tolist()[dq-1]
                        # c_tuda_list_quantile_value = np.array(c_tuda_list_sorted).tolist()[0]
                        # print(c_tuda_list_quantile_value)
                        client_quantile_list = c_tuda_list_quantile_value * np.array(max_values_list)
                        # print(client_quantile_list)
                        # exit(0)
                        # client_quantile.append(client_quantile_list)
                        
                        # temp.append(client_quantile)
                        # print(f"client q is {client_quantile}")
                        # normed_client_quantile = [a * b for a, b in zip(client_quantile, max_values_list)]
                        # print(f"final quantile is {normed_client_quantile}")
                        # print(normed_client_quantile)

                        # exit(0)
                        global_quantile.append(client_quantile_list)

                    # print(f"temp length is {len(temp)}")
                    # print(f"second length of temp check is {len(temp[0])}")
                    # print(f'temp looks like {temp[0]}')
                    # print(f"max is {max_values_list}")
                    # print(len(global_quantile))
                    # print(len(global_quantile[0]))
                    # print(global_quantile)
                    # exit(0)

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
                    local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c) 
                    w_local, loss, cons_loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)  
                    local_loss.append(copy.deepcopy(loss))
            eval_loss.append(sum(local_loss)/len(local_loss))
            print(f"the average test loss is {round(sum(local_loss)/len(local_loss), 3)}\n---------------------------------")
        if args.dataset == 'fhwa':
            model_path = f"hdd/saved_pretrain_models/"
        elif args.dataset == 'ct':
            model_path = f"{args.dataset}/saved_pretrain_models/"
        if args.mode == "pretrain_calib":
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_"+str(args.property_type)+"_glob", ix_epoch)

        global_quantile_array = np.array(global_quantile)
        # print(f'shape of global_quantile_array is {len(global_quantile_array[0])}')
        # print(f'shape of global_quantile_array is {len(global_quantile_array)}')
        # exit(0)

        save_cp_result_with_sep_type(args = args, client_dataset = client_dataset, global_quantile_array = global_quantile_array, glob_model = glob_model, dq = dq, cq = cq)

        print("============================")
        print("Calibration result saved, pretrain with calib ends.")

    ############################

    # training with clusters
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

        # cq = round(14/20, 2) # quantile for client
        # dq = round(76/90, 2) # quantile for local dataset

        if args.client == 1:
            dq = 81
            cq = 1
        # elif args.client == 100:
        #     dq = round(79/90, 2)
        else:
            dq = 79
            cq = 7
        ############################
        # generate list of local models for each user

        # for all 100 clients, each copy the pretrained global model

        # local_weights = {}      # client ID: weight: val
        # for c in range(args.client):
        #     w_local_dict = {}
        #     for key in glob_model.state_dict().keys():
        #         w_local_dict[key] = glob_model.state_dict()[key]
        #     local_weights[c] = w_local_dict

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
        cp_size_visual_list = []

        epoch_num = 0
        if args.mode == 'pretrain_calib':
            epoch_num = args.epoch
        elif args.mode == 'train_cp':
            epoch_num = args.cp_epoch
        
        print("epoch number checks: {}".format(epoch_num))
        
        for ix_epoch in range(1, epoch_num+1): # 30 epoch in total
            local_loss = []             # loss for all clients in this round
            glob_weight = {}            # glob weights in this round
            total_len = 0               # total dataset length
            cluster_len = [0] * args.cluster
            global_quantile = []        # the global quantile for calibration in each epoch


            # the cp dic of every round load here. 
            cp_dic = dic_loader(args)
            cluster_id = {int(group): clients_data["clients"] for group, clients_data in cp_dic.items()}
            client2cluster = {client: group for group, clients in cluster_id.items() for client in clients}
            for key, value in cp_dic.items():
                if key == str(0):
                    cp_size_visual_list.append(value["clients_average_check"])

            # a fraction of all devices
            m = max(int(args.frac * args.client), 1)

            if ix_epoch == epoch_num:                          
                m = args.client
            # print(f"m value for check is {m}")
            
            idxs_users = np.random.choice(range(args.client), m, replace=False)  
            print(f"Communication round  {ix_epoch}\n---------")
            print("Selected:", idxs_users)     

            last = (ix_epoch == epoch_num)
            if last: 
                args.frac = 1

            """
            
            previous version of client to cluster.

            """
    
            # cluster_id = cluster_id_property(cluster_models, client_dataset, args, idxs_users)  # cluster: clients
            # client2cluster = get_dict_keys(cluster_id, idxs_users)                              # client:  cluster

            # print(cluster_id)
            # print('====================\n')
            # print(client2cluster)
            # print('====================\n')
            # # exit(0)

            """
            
            new version of client to cluster based on calibration result.
            
            """
            
            for c in idxs_users:
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                net_local = copy.deepcopy(cluster_models[client2cluster[c]]) 
                # print(client2cluster[c])
                # exit(0)
                w_local = net_local.state_dict()
                for k in local_weights[c].keys():
                    if k not in clust_weight_keys:
                        w_local[k] = local_weights[c][k]
                net_local.load_state_dict(w_local)

                # cp_dic = dic_loader(args)
                cp_value = find_group_info(cp_dic, c)
                # print(args.device)
                w_local, loss, idx = local.train(net=net_local.to(args.device), 
                                                 idx=c, w_glob_keys=clust_weight_keys, 
                                                 lr=args.max_lr, last=last, cp_value=cp_value)
                
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
                if len(glob_weight) == 0: 
                    glob_weight = copy.deepcopy(w_local)
                    for key in glob_model.state_dict().keys():
                        glob_weight[key] = glob_weight[key]*client_dataset[c]["len"][0] # weighted update
                        local_weights[c][key] = w_local[key]
                else:
                    for key in glob_model.state_dict().keys():
                        glob_weight[key] += w_local[key]*client_dataset[c]["len"][0]
                        local_weights[c][key] = w_local[key]
                # print(ix_epoch, idx, loss)
                print(f"train with cp process: {ix_epoch}, {idx}, {loss}")
            
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

                    cp_dic = dic_loader(args)
                    cp_value = cp_dic['{}'.format(cluster)]["cp_value"]

                    w_local, loss = cluster_explore(net=net_local.to(args.device), 
                                                    w_glob_keys=clust_weight_keys, 
                                                    lr=args.max_lr, args=args, 
                                                    dataloaders=cluster_train_loader, cp_value=cp_value)
                    
                    for key in glob_weight.keys():
                        cluster_weights[cluster][key] = w_local[key]
                    cluster_models[cluster].load_state_dict(w_local)
                    print("cluster fine-tune:", ix_epoch, cluster, loss)
                
            temp = []
            for c in idxs_users:  # client update iterations
                client_quantile = []
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                net_local = copy.deepcopy(cluster_models[client2cluster[c]])
                w_local = net_local.state_dict()
                net_local.load_state_dict(w_local)
                last = False

                max_values_list = local.calib_norm(net=net_local.to(args.device), 
                                                     idx=c, w_glob_keys=None, 
                                                     lr=args.max_lr, last=last)

                c_tuda_list = local.calib(net=net_local.to(args.device), 
                                                idx=c, w_glob_keys=None, 
                                                lr=args.max_lr, last=last, max_values = max_values_list)

                c_tuda_list_sorted = np.sort(c_tuda_list, axis=0)
                c_tuda_list_quantile_value = np.array(c_tuda_list_sorted).tolist()[dq-1]
                client_quantile_list = c_tuda_list_quantile_value * np.array(max_values_list)
                global_quantile.append(client_quantile_list)

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

            # for c in range(args.client):
            #     glob_model.load_state_dict(local_weights[c])
            #     local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
            #     w_local, loss, cons_loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)   
            #     local_loss.append(copy.deepcopy(loss))

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

            # exit(0)

        model_path = "hdd/saved_models/"
        glob_model.load_state_dict(glob_weight)
        save_model(model_path, glob_model, 
                   str(args.dataset)+"_"+str(args.local_updates)+"_"+str(args.model)+"_"+args.method+"_"+str(args.property_type)+'_'+str(args.client), 
                   ix_epoch)
        
        for c in range(args.cluster):
            glob_model.load_state_dict(cluster_weights[c])
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_"+args.method+"_Cluster_"+str(c)+'_'+str(args.client), ix_epoch)
        
        for c in range(args.client):
            glob_model.load_state_dict(local_weights[c])
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_"+args.method+"_Client_"+str(c)+'_'+str(args.client), ix_epoch)

        plt.plot(cp_size_visual_list)

        # Adding title and labels
        plt.title('Size of cp for group 1')
        plt.xlabel('Epochs')
        plt.ylabel('Values')

        # Show the plot
        plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')
        # sys.stdout.close()
        # sys.stdout=stdoutOrigin