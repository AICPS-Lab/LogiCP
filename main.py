"""
The main file for training and evaluating LogiCP format
with options to compare with other benchmarks.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from IoTData import SequenceDataset
from utils.update import LocalUpdateProp, cluster_explore, dic_loader, find_group_info, cp_loss
from utils_training import to_device, save_model, model_init, get_shared_dataset, save_cp_result_with_sep_type, get_cp_qq_settings
import copy
import matplotlib.pyplot as plt

from options import args_parser

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

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
    
    client_dataset = {}
    for c in range(args.client):
        client_dataset[c] = {}
        train_loader_private, trainset_shared, calib_loader_private, pre_loader_private, nor_loader_private, val_loader, test_loader, dataset_len = get_shared_dataset(c, args.dataset)
        client_dataset[c]["train_private"] = train_loader_private # dataloader for the private training data
        client_dataset[c]["train_shared"] = trainset_shared # the shared data in training 
        client_dataset[c]["val"] = val_loader # dataloader for the validation data
        client_dataset[c]["test"] = test_loader # dataloader for the test data
        client_dataset[c]["nor"] = nor_loader_private # dataloader for the normalization data for calib
        client_dataset[c]["len"] = dataset_len # [len(train_dataset), len(calib_dataset), len(pre_dataset), len(val_dataset), len(test_dataset), len(shared_dataset)]

        client_dataset[c]["cal"] = calib_loader_private # dataloader for the calibration data
        client_dataset[c]["pre"] = pre_loader_private # dataloader for the pretrain data

    print("Loaded client dataset.")
    
    # load cp quantile pf quantile setting here. 
    dq, cq = get_cp_qq_settings(args)

    # pretrain the model and do calibration in the last round. 
    if args.mode == "pretrain_calib":

        # loading vanila model
        glob_model, clust_weight_keys = model_init(args)
        glob_model = to_device(glob_model, args.device)
        net_keys = [*glob_model.state_dict().keys()]

        # generate list of local models for each user
        local_weights = {}     
        for c in range(args.client):
            w_local_dict = {}
            for key in glob_model.state_dict().keys():
                w_local_dict[key] = glob_model.state_dict()[key]
            local_weights[c] = w_local_dict
        
        train_loss = [] 
        eval_loss = []   

        # pretrain round
        for ix_epoch in range(1, args.epoch+1): 
            local_loss = []         # all clients loss list
            glob_weight = {}        # glob weights
            total_len = 0           # dataset length
            global_quantile = []    # global quantile lists for cp

            m = max(int(args.frac * args.client), 1) # a fraction of all devices
            if ix_epoch == args.epoch: # in the last round, all users are selected 
                m = args.client
            idxs_users = np.random.choice(range(args.client), m, replace=False)          # select devices for this round
            
            try:
                print(f"Communication round  {ix_epoch}\n---------")
                print("Selected:", idxs_users)

                # client update epoch
                for c_ind, c in enumerate(idxs_users):
                    if args.mode == "pretrain_calib":
                        local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                    
                    net_local = copy.deepcopy(glob_model) 
                    w_local = net_local.state_dict()
                    net_local.load_state_dict(w_local)
                    last = False
                    w_local, loss, idx = local.pretrain(net=net_local.to(args.device), idx=c, w_glob_keys=None, lr=args.max_lr, last=last) # train local
                    local_loss.append(copy.deepcopy(loss))
                    total_len += client_dataset[c]["len"][0] 

                    # update shared weights
                    if len(glob_weight) == 0:
                        glob_weight = copy.deepcopy(w_local)
                        for key in glob_model.state_dict().keys():
                            glob_weight[key] = glob_weight[key]*client_dataset[c]["len"][0]
                            local_weights[c][key] = w_local[key]
                    else:
                        for key in glob_model.state_dict().keys():
                            glob_weight[key] += w_local[key]*client_dataset[c]["len"][0]
                            local_weights[c][key] = w_local[key]

                    print(ix_epoch, idx, loss)
                
                # at the last round of pretrain, the first CP is performed
                if ix_epoch == args.epoch:

                    print(f"==========================================")
                    print(f"calibration starts here.")
                    print(f"epoch checks expect {args.epoch} is {ix_epoch}")
                    print(f"==========================================")

                    for c_ind, c in enumerate(idxs_users):  
                        local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c) 
                        net_local = copy.deepcopy(glob_model)
                        w_local = net_local.state_dict()
                        net_local.load_state_dict(w_local)
                        last = False

                        # Distributed CP with normalization
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

            except KeyboardInterrupt:
                break
            
            # global weights update
            for k in glob_model.state_dict().keys():    
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

        # save pretrained model
        if args.dataset == 'fhwa':
            model_path = f"hdd/saved_pretrain_models/"
        elif args.dataset == 'ct':
            model_path = f"ct/saved_pretrain_models/"
        if args.mode == "pretrain_calib":
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_"+str(args.property_type)+"_glob", ix_epoch)

        # save cp result
        global_quantile_array = np.array(global_quantile)
        save_cp_result_with_sep_type(args = args, client_dataset = client_dataset, global_quantile_array = global_quantile_array, glob_model = glob_model, dq = dq, cq = cq)

        print("============================")
        print("Calibration result saved, pretrain with calib ends.")


    ############################
    # train_cp with clusters
    elif "train_cp" in args.mode and args.cluster > 0:

        # loading pretrained model
        glob_model, clust_weight_keys = model_init(args)
        glob_model = to_device(glob_model, args.device)
        net_keys = [*glob_model.state_dict().keys()]

        # load cp quantile pf quantile setting here. 
        dq, cq = get_cp_qq_settings(args)

        # load pretrained clients
        local_weights = {}  
        for c in range(args.client):
            w_local_dict = {}
            for key in glob_model.state_dict().keys():
                w_local_dict[key] = clust_weight_keys[key]
            local_weights[c] = w_local_dict

        print("Loaded pretrained client models.")
        
        # load clusters
        cluster_weights = {}
        for cluster in range(args.cluster):
            cluster_weights[cluster] = {}

        cluster_models = {}
        for cluster in range(args.cluster):
            cluster_models[cluster] = copy.deepcopy(glob_model)

        train_loss = [] 
        eval_loss = []  

        epoch_num = 0
        epoch_num = args.cp_epoch
        
        print("epoch number checks: {}".format(epoch_num))
        
        for ix_epoch in range(1, epoch_num+1): 
            local_loss = []             # all clients loss list
            glob_weight = {}            # glob weights 
            total_len = 0               # total dataset length
            cluster_len = [0] * args.cluster
            global_quantile = []        # global quantile lists for cp

            # the cp dic loads here. 
            cp_dic = dic_loader(args)
            cluster_id = {int(group): clients_data["clients"] for group, clients_data in cp_dic.items()}
            client2cluster = {client: group for group, clients in cluster_id.items() for client in clients}

            # a fraction of all devices
            m = max(int(args.frac * args.client), 1)

            if ix_epoch == epoch_num:                          
                m = args.client
            
            idxs_users = np.random.choice(range(args.client), m, replace=False)  
            print(f"Communication round  {ix_epoch}\n---------")
            print("Selected:", idxs_users)     

            last = (ix_epoch == epoch_num)
            if last: 
                args.frac = 1
            
            for c in idxs_users:
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                net_local = copy.deepcopy(cluster_models[client2cluster[c]]) 

                w_local = net_local.state_dict()
                for k in local_weights[c].keys():
                    if k not in clust_weight_keys:
                        w_local[k] = local_weights[c][k]
                net_local.load_state_dict(w_local)

                cp_value = find_group_info(cp_dic, c)
                w_local, loss, idx = local.train(net=net_local.to(args.device), idx=c, w_glob_keys=clust_weight_keys, loss_func=cp_loss, lr=args.max_lr, last=last, cp_value=cp_value)
                
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

            # cluster-wise training
            if args.mode == "train_cp": 
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
            
            # update cp region
            for c in idxs_users: 

                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c) 
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

            # save updated cp region
            global_quantile_array = np.array(global_quantile)
            save_cp_result_with_sep_type(args = args, client_dataset = client_dataset, global_quantile_array = global_quantile_array, glob_model = glob_model, dq = dq, cq = cq)

            loss_avg = sum(local_loss)/len(local_loss)
            train_loss.append(loss_avg)

            for c in idxs_users:  
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)  
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

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')