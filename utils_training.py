"""
Utils and settings for training. 
Implementations for getting device, 
loading data to device, saving models, 
and loading dataset.
"""
import torch
import json
import numpy as np
import torch.nn as nn
from IoTData import SequenceDataset
from torch.utils.data import DataLoader
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id, cluster_id_property
from transformer import TimeSeriesTransformer
import numpy as np
from class_CP_QQ import calc_matrix_M
import copy
import os

weight_keys_mapping = {
    'lstm': ['lstm_1.weight_ih', 'lstm_1.weight_hh', 'lstm_1.bias_ih', 'lstm_1.bias_hh', 
             'lstm_2.weight_ih', 'lstm_2.weight_hh', 'lstm_2.bias_ih', 'lstm_2.bias_hh'],
    'gru': ['gru_1.weight_ih', 'gru_1.weight_hh', 'gru_1.bias_ih', 'gru_1.bias_hh', 
            'gru_2.weight_ih', 'gru_2.weight_hh', 'gru_2.bias_ih', 'gru_2.bias_hh'],
    'rnn': ['rnn_1.weight_ih', 'rnn_1.weight_hh', 'rnn_1.bias_ih', 'rnn_1.bias_hh', 
            'rnn_2.weight_ih', 'rnn_2.weight_hh', 'rnn_2.bias_ih', 'rnn_2.bias_hh'],
    'transformer': ['encoder_input_layer.weight', 'encoder_input_layer.bias',
                    'decoder_input_layer.weight', 'decoder_input_layer.bias',
                    'linear_mapping.weight', 'linear_mapping.bias']
}

def model_init(args):
    if args.dataset == 'fhwa':

        if args.model == 'RNN' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['rnn']
            print(f"================================")
            print(f"LogiCP Loading vanilla RNN model.")
            print(f"================================")

        elif args.model == 'RNN' and args.mode == 'train_cp' and args.method == 'LogiCP':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = "hdd/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"LogiCP Loading pretrained RNN model.")
            print(f"================================")

        elif args.model == 'RNN' and args.mode == 'train_cp' and args.method != 'LogiCP':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd_{args.method}/saved_pretrain_models_{args.seq_type}_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)

        elif args.model == 'LSTM' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['lstm']
            print(f"================================")
            print(f"LogiCP Loading vanilla LSTM model.")
            print(f"================================")

        elif args.model == 'LSTM' and args.mode == 'train_cp' and args.method == 'LogiCP':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"LogiCP oading pretrained LSTM model.")
            print(f"================================")

        elif args.model == 'LSTM' and args.mode == 'train_cp' and args.method != 'LogiCP':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd_{args.method}/saved_pretrain_models_cp_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)

        elif args.model == 'transformer' and args.mode == 'pretrain_calib':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys = weight_keys_mapping['transformer']
            print(f"================================")
            print(f"LogiCP Loading vanilla transformer model.")
            print(f"================================")

        elif args.model == 'transformer' and args.mode == 'train_cp' and args.method == 'LogiCP':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys_path = "hdd/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"LogiCP Loading pretrained transformer model.")
            print(f"================================")

        elif args.model == 'transformer' and args.mode == 'train_cp' and args.method != 'LogiCP':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys_path = f"hdd_{args.method}/saved_pretrain_models_{args.sep_type}_only/"
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)

        elif args.model == 'GRU' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
            print(f"================================")
            print(f"LogiCP Loading vanilla GRU model.")
            print(f"================================")

        elif args.model == 'GRU' and args.mode == 'train_cp' and args.method == 'LogiCP':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"LogiCP ading pretrained GRU model.")
            print(f"================================")

        elif args.model == 'GRU' and args.mode == 'train_cp' and args.method != 'LogiCP':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd_{args.method}/saved_pretrain_models_cp_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)

        else:
            print("Model type:", args.model, "not implemented")

    elif args.dataset == 'ct':

        if args.model == 'RNN' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['rnn']
            print(f"================================")
            print(f"Loading vanilla RNN model.")
            print(f"================================")

        elif args.model == 'RNN' and args.mode == 'train_cp' and args.method == 'LogiCP':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = "ct/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained RNN model.")
            print(f"================================")

        elif args.model == 'RNN' and args.mode == 'train_cp' and args.method != 'LogiCP':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct_{args.method}/saved_pretrain_models_{args.sep_type}_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)

        elif args.model == 'LSTM' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['lstm']
            print(f"================================")
            print(f"Loading vanilla CT LSTM model.")
            print(f"================================")

        elif args.model == 'LSTM' and args.mode == 'train_cp' and args.method == 'LogiCP':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained CT LSTM model.")
            print(f"================================")

        elif args.model == 'LSTM' and args.mode == 'train_cp' and args.method != 'LogiCP':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct_{args.method}/saved_pretrain_models_{args.sep_type}_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)

        elif args.model == 'GRU' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
            print(f"================================")
            print(f"Loading vanilla CT GRU model.")
            print(f"================================")

        elif args.model == 'GRU' and args.mode == 'train_cp' and args.method == 'LogiCP':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained CT GRU model.")
            print(f"================================")

        elif args.model == 'GRU' and args.mode == 'train_cp' and args.method != 'LogiCP':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct_{args.method}/saved_pretrain_models_{args.sep_type}_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)

        elif args.model == 'transformer' and args.mode == 'pretrain_calib':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys = weight_keys_mapping['transformer']
            print(f"================================")
            print(f"Loading vanilla CT transformer model.")
            print(f"================================")

        elif args.model == 'transformer' and args.mode == 'train_cp' and args.method == 'LogiCP':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys_path = "ct/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            print(clust_weight_keys_p)
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained CT transformer model.")
            print(f"================================")

        elif args.model == 'transformer' and args.mode == 'train_cp' and args.method != 'LogiCP':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys_path = f"ct_{args.method}/saved_pretrain_models_{args.sep_type}_only/"
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)

    return glob_model, clust_weight_keys

def get_shared_dataset(client_id, dataset_name):
    """
    load dataset for pretrain, calibration, train, test, evaluation. 
    datasets are divided into private and public part. 
    """
    dataset_array = {}
    if dataset_name == 'fhwa':    
        dataset_path = "fhwa_dataset/" 
    elif dataset_name == 'ct':
        dataset_path = "ct_dataset/"

    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    
    public_len = int(0.214*len(dataset_array["train_x"])) 
    cal_len = public_len + 90
    nor_len = cal_len + 50
    pre_len = int(0.762*len(dataset_array["train_x"])) 
    final_len = int(1*len(dataset_array["train_x"])) 

    train_dataset = SequenceDataset(dataset_array["train_x"][:public_len], dataset_array["train_y"][:public_len])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])
    calib_dataset = SequenceDataset(dataset_array["train_x"][public_len:cal_len], dataset_array["train_y"][public_len:cal_len])
    norm_dataset = SequenceDataset(dataset_array["train_x"][cal_len:nor_len], dataset_array["train_y"][cal_len:nor_len])
    pre_dataset = SequenceDataset(dataset_array["train_x"][nor_len:pre_len], dataset_array["train_y"][nor_len:pre_len])
    shared_dataset = SequenceDataset(dataset_array["train_x"][pre_len:final_len], dataset_array["train_y"][pre_len:final_len])

    dataset_len = [len(train_dataset), len(calib_dataset), len(norm_dataset), len(pre_dataset), len(val_dataset), len(test_dataset), len(shared_dataset)]

    train_loader_private = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    calib_loader_private = DataLoader(calib_dataset, batch_size=90, shuffle=True, drop_last=True)
    pre_loader_private = DataLoader(pre_dataset, batch_size=64, shuffle=True, drop_last=True)
    nor_loader_private = DataLoader(pre_dataset, batch_size=50, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=20, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=20, drop_last=True)

    return train_loader_private, [dataset_array["train_x"][pre_len:final_len], dataset_array["train_y"][pre_len:final_len]], calib_loader_private, pre_loader_private, nor_loader_private, val_loader, test_loader, dataset_len

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

def save_model(path, model, model_name, epoch):
    if not os.path.isdir(path):
        os.makedirs(path)
    save_prefix = os.path.join(path, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close() 

def get_result_file(cp_value, client_quantile, data_quantile):
    """
    
    create a dic to store all necessary result

    """
    result_file = {
        "cp_value": cp_value,
        "client_quantile(k)": client_quantile,
        "data_quantile(l)": data_quantile,
        "length": len(cp_value)
    }

    print(f"cp result file generated.")

    return result_file

def get_cluster_result(sorted_indices, cp_value):
    """
    
    create a dic to store all necessary result

    """
    result_file = {
        "clients": sorted_indices,
        "cp_value": cp_value,
        "clients_average_check": np.mean(cp_value),
        "length": len(cp_value),
        "length_cluster": len(sorted_indices)
    }

    return result_file

def save_cp_result(cp_path, result_file, model):
    """
    
    save cp result to a path for further 

    """
    if not os.path.isdir(cp_path):
        os.makedirs(cp_path)
    cp_prefix = os.path.join(cp_path, model)
    cp_file_name = '{}.json'.format(cp_prefix)
    with open(cp_file_name, 'w') as file:
        json.dump(result_file, file, indent=4)

def default_converter(o):
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def get_dict_keys(cluster_id, idxs_users):
    d = {}
    for i in idxs_users:
        for key, val in cluster_id.items():
            if i in val:
                d[i] = key
    return d

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

def convert_np_int_to_int(item):
    if isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, list):
        return [convert_np_int_to_int(i) for i in item]
    return item

def save_cluster_result(cluster_path, result_file, model):

    if not os.path.isdir(cluster_path):
        os.makedirs(cluster_path)
    cp_prefix = os.path.join(cluster_path, model)
    cp_file_name = '{}_Cluster.json'.format(cp_prefix)
    print("cp result is saved to {}".format(cp_file_name))
    with open(cp_file_name, 'w') as file:
        json.dump(result_file, file, default=default_converter, indent=4)

def get_cp_qq_settings(args):
    """
    Determines the quantile settings for Conformal Prediction (CP) calibration. Result is computed from compute.py
    """
    if args.client == 1:
        dq = 81
        cq = 1
    else:
        dq = 79
        cq = 7
    return dq, cq
    
def save_cp_result_with_sep_type(args, client_dataset, global_quantile_array, glob_model, dq, cq):

    if args.sep_type == 'cp':

            global_avg = [sum(sublist) / len(sublist) for sublist in global_quantile_array]
            sorted_pairs = sorted(enumerate(global_avg), key=lambda x: x[1]) 
            sorted_indices = [index for index, value in sorted_pairs] 

            num_groups = args.cluster
            group_size = len(sorted_indices) // num_groups
            
            if len(sorted_indices) % num_groups == 0:
                print("We can divide the sorted indices into 10 groups.")
            else:
                print("Cannot evenly divide the sorted indices into 10 groups.")
            
            rearranged_list = [global_quantile_array[i] for i in sorted_indices] # Rearrange another_list according to sorted_indices
            grouped_lists = [rearranged_list[i:i+group_size] for i in range(0, len(rearranged_list), group_size)] # Divide the rearranged list into groups of 10
            grouped_indices = [sorted_indices[i:i+group_size] for i in range(0, len(rearranged_list), group_size)] # Generate the clients id of each cluster

            final_cp_file = {}
            for i, group in enumerate(grouped_lists):
                quantile_cq = np.quantile(group, cq, axis=0)
                cp_size_cluster = quantile_cq.tolist()
                cluster_file = get_cluster_result(grouped_indices[i], cp_size_cluster)
                final_cp_file[i] = cluster_file

            if args.method == 'LogiCP':
                cluster_path_cp = "hdd/cluster_cp_result_with_cp/"
            
            elif args.method != 'LogiCP':
                cluster_path_cp = f"hdd_{args.method}/cluster_cp_result_with/"

            save_cluster_result(cluster_path_cp, final_cp_file, args.method)

    elif args.sep_type == 'vanilla':

        quantiles_90 = np.quantile(global_quantile_array, cq, axis=0)
        cp_size = quantiles_90.tolist()
        result_file = get_result_file(cp_size, client_quantile=cq, data_quantile=dq)

        cp_path = "hdd/cluster_cp_result_with_vanilla/"
        save_cp_result(cp_path, result_file, args.method)
    
    elif "train" in args.mode and args.sep_type == "spec_m":

        cluster_weights = {}
        for cluster in range(args.cluster):
            cluster_weights[cluster] = {}

        cluster_models = {}
        for cluster in range(args.cluster):
            cluster_models[cluster] = copy.deepcopy(glob_model)

        m = args.client
        args.frac = 1
        idxs_users = np.random.choice(range(args.client), m, replace=False)   

        cluster_id = cluster_id_property(cluster_models, client_dataset, args, idxs_users)
        client2cluster = get_dict_keys(cluster_id, idxs_users) 

        num_groups = args.cluster
        group_size = 10

        grouped_lists = {key: [global_quantile_array[idx] for idx in indices] for key, indices in cluster_id.items()}
        final_cp_file = {}

        quantiles = []
        for key, arrays in grouped_lists.items():
            stacked_arrays = np.stack(arrays)  
            seventh_smallest_values = np.sort(stacked_arrays, axis=0)[cq-1, :].tolist()
            quantiles.append(seventh_smallest_values)


        for key, value in cluster_id.items():
            final_cp_file[str(key)] = {
                "clients": value,
                "cp_value": quantiles[key],
                "clients_average_check": np.mean(quantiles[key]),
                "length": len(quantiles[key]),
                "length_cluster": len(value)
            }

        converted_data = {key: convert_np_int_to_int(value) for key, value in final_cp_file.items()}
        if args.dataset == 'fhwa' and args.method == "LogiCP":
            cluster_path_spec = f"hdd/cluster_cp_result_with_specm_{args.model}_{args.cp_epoch}_{args.client}/"
        
        elif args.dataset == 'ct' and args.method == 'LogiCP':
            cluster_path_spec = f"{args.dataset}/cluster_cp_result_with_specm_{args.model}_{args.cp_epoch}_{args.client}/"
        save_cluster_result(cluster_path_spec, converted_data, args.method)

    elif "train" in args.mode and args.sep_type == "value":

        cluster_weights = {}
        for cluster in range(args.cluster):
            cluster_weights[cluster] = {}

        cluster_models = {}
        for cluster in range(args.cluster):
            cluster_models[cluster] = copy.deepcopy(glob_model)

        m = args.client
        args.frac = 1
        idxs_users = np.random.choice(range(args.client), m, replace=False)   

        cluster_id = compute_cluster_id(cluster_models, client_dataset, args, idxs_users)
        client2cluster = get_dict_keys(cluster_id, idxs_users) 

        num_groups = args.cluster
        group_size = 10

        grouped_lists = {key: [global_quantile_array[idx] for idx in indices] for key, indices in cluster_id.items()}
        final_cp_file = {}

        quantiles = []
        for key, arrays in grouped_lists.items():
            stacked_arrays = np.stack(arrays)
            seventh_smallest_values = np.sort(stacked_arrays, axis=0)[cq-1, :].tolist()
            quantiles.append(seventh_smallest_values)

        for key, value in cluster_id.items():
            final_cp_file[str(key)] = {
                "clients": value,
                "cp_value": quantiles[key],
                "clients_average_check": np.mean(quantiles[key]),
                "length": len(quantiles[key]),
                "length_cluster": len(value)
            }

        converted_data = {key: convert_np_int_to_int(value) for key, value in final_cp_file.items()}

        if args.dataset == 'fhwa':
            cluster_path_spec = f"hdd_{args.method}/cluster_cp_result_with_{args.sep_type}_{args.model}_{args.cp_epoch}_{args.client}/"
            print(f'the CP result is saved to {cluster_path_spec}')
        elif args.dataset == 'ct':
            cluster_path_spec = f"{args.dataset}_{args.method}/cluster_cp_result_with_{args.sep_type}_{args.model}_{args.cp_epoch}_{args.client}/"
            print(f'the CP result is saved to {cluster_path_spec}')
        save_cluster_result(cluster_path_spec, converted_data, args.method)



