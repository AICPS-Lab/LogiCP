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
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN, MultiRegressionLSTM, MultiRegressionGRU, MultiRegressionRNN
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id, cluster_id_property, cluster_explore
from transformer import TimeSeriesTransformer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
import scipy.special as sc
from class_CP_QQ import calc_matrix_M, sum_hypergeo, Multi_Boucle
import os
import copy



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
        # if args.model == 'LSTM':
        #     glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        #     clust_weight_keys = weight_keys_mapping['lstm']
        if args.model == 'Test':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
        elif args.model == 'RNN' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['rnn']
            print(f"================================")
            print(f"Loading vanilla RNN model.")
            print(f"================================")
        elif args.model == 'RNN' and args.mode == 'train_cp' and args.method == 'FedSTL':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = "hdd/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained RNN model.")
            print(f"================================")
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'RNN' and args.mode == 'train_cp' and args.method != 'FedSTL':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd_{args.method}/saved_pretrain_models_{args.seq_type}_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(1)
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'LSTM' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['lstm']
            print(f"================================")
            print(f"Loading vanilla LSTM model.")
            print(f"================================")
        elif args.model == 'LSTM' and args.mode == 'train_cp' and args.method == 'FedSTL':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained LSTM model.")
            print(f"================================")
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'LSTM' and args.mode == 'train_cp' and args.method != 'FedSTL':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd_{args.method}/saved_pretrain_models_cp_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(1)
        elif args.model == 'transformer' and args.mode == 'pretrain_calib':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys = weight_keys_mapping['transformer']
            print(f"================================")
            print(f"Loading vanilla transformer model.")
            print(f"================================")
        elif args.model == 'transformer' and args.mode == 'train_cp' and args.method == 'FedSTL':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys_path = "hdd/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            print(clust_weight_keys_p)
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained transformer model.")
            print(f"================================")
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'transformer' and args.mode == 'train_cp' and args.method != 'FedSTL':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys_path = f"hdd_{args.method}/saved_pretrain_models_{args.sep_type}_only/"
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(1)
        elif args.model == 'GRU' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
            print(f"================================")
            print(f"Loading vanilla GRU model.")
            print(f"================================")
        elif args.model == 'GRU' and args.mode == 'train_cp' and args.method == 'FedSTL':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained GRU model.")
            print(f"================================")
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'GRU' and args.mode == 'train_cp' and args.method != 'FedSTL':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"hdd_{args.method}/saved_pretrain_models_cp_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(1)
        else:
            print("Model type:", args.model, "not implemented")

    elif args.dataset == 'sumo':
        if args.model == 'GRU':
            glob_model = MultiRegressionGRU(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
        elif args.model == 'LSTM':
            glob_model = MultiRegressionLSTM(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['lstm']
        elif args.model == 'RNN':
            glob_model = MultiRegressionRNN(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['rnn']
        elif args.model == 'transformer':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys = weight_keys_mapping['transformer']
        else:
            print("Model type:", args.model, "not implemented")

    if args.dataset == 'ct':
        # if args.model == 'LSTM':
        #     glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        #     clust_weight_keys = weight_keys_mapping['lstm']
        if args.model == 'Test':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
        elif args.model == 'RNN' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['rnn']
            print(f"================================")
            print(f"Loading vanilla RNN model.")
            print(f"================================")
        elif args.model == 'RNN' and args.mode == 'train_cp' and args.method == 'FedSTL':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = "ct/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained RNN model.")
            print(f"================================")
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'RNN' and args.mode == 'train_cp' and args.method != 'FedSTL':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct_{args.method}/saved_pretrain_models_{args.sep_type}_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(1)
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"

        elif args.model == 'LSTM' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['lstm']
            print(f"================================")
            print(f"Loading vanilla CT LSTM model.")
            print(f"================================")
        elif args.model == 'LSTM' and args.mode == 'train_cp' and args.method == 'FedSTL':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained CT LSTM model.")
            print(f"================================")
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'LSTM' and args.mode == 'train_cp' and args.method != 'FedSTL':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct_{args.method}/saved_pretrain_models_{args.sep_type}_only/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            # print(1)
        
        elif args.model == 'GRU' and args.mode == 'pretrain_calib':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
            print(f"================================")
            print(f"Loading vanilla CT GRU model.")
            print(f"================================")
        elif args.model == 'GRU' and args.mode == 'train_cp' and args.method == 'FedSTL':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys_path = f"ct/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained CT GRU model.")
            print(f"================================")
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'GRU' and args.mode == 'train_cp' and args.method != 'FedSTL':
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
        elif args.model == 'transformer' and args.mode == 'train_cp' and args.method == 'FedSTL':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys_path = "ct/saved_pretrain_models/" 
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            print(clust_weight_keys_p)
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(f"================================")
            print(f"Loading pretrained CT transformer model.")
            print(f"================================")
            # "hdd/saved_models/fhwa_RNN_constraint_glob_epoch_30.pt"
        elif args.model == 'transformer' and args.mode == 'train_cp' and args.method != 'FedSTL':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys_path = f"ct_{args.method}/saved_pretrain_models_{args.sep_type}_only/"
            clust_weight_keys_p = os.path.join(clust_weight_keys_path, '{}_{}_{}_glob_epoch_{}.pt'.format(args.dataset, args.model, args.property_type, args.epoch))
            clust_weight_keys = torch.load(clust_weight_keys_p)
            print(1)
    return glob_model, clust_weight_keys



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



def get_client_dataset(client_id, dataset_name):
    """
    Getting client dataset files by dataset name. 
    """
    dataset_array = {}
    if dataset_name == 'fhwa':
        dataset_path = "hdd/FHWA_dataset/torch_dataset/"
    elif dataset_name == 'sumo':
        dataset_path = "hdd/SUMO_dataset/learn_dataset/"
    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    train_dataset = SequenceDataset(dataset_array["train_x"], dataset_array["train_y"])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])
    dataset_len = [len(train_dataset), len(val_dataset), len(test_dataset)]
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=5, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=5, drop_last=True)

    return train_loader, val_loader, test_loader, dataset_len



def get_shared_dataset(client_id, dataset_name):
    """
    Getting client and shared dataset files by dataset name. 
    Training data is seperated to a small shared group and a large private group.
    """
    dataset_array = {}
    if dataset_name == 'fhwa':
        # dataset_path = "hdd/FHWA_dataset/torch_dataset/"
        dataset_path = "/home/dian/FedSTL_new/torch_dataset_ver5/"
        
    elif dataset_name == 'sumo':
        dataset_path = "hdd/SUMO_dataset/learn_dataset/"
    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    
    public_len = int(0.5*len(dataset_array["train_x"]))

    train_dataset = SequenceDataset(dataset_array["train_x"][public_len:], dataset_array["train_y"][public_len:])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])

    dataset_len = [len(train_dataset), len(val_dataset), len(test_dataset)]

    train_loader_private = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

    return train_loader_private, [dataset_array["train_x"][:public_len], dataset_array["train_y"][:public_len]], val_loader, test_loader, dataset_len

def get_shared_dataset_1(client_id, dataset_name):
    """

    Getting client and shared dataset files by dataset name. 

    Training data is seperated to a small shared group and a large private group.

    Split the dataset into tr, cal, te.

    """
    dataset_array = {}
    if dataset_name == 'fhwa':
        # dataset_path = "hdd/FHWA_dataset/torch_dataset/"
        dataset_path = "FHWA_dataset/torch_dataset/"
    elif dataset_name == 'sumo':
        dataset_path = "hdd/SUMO_dataset/learn_dataset/"

    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id + 779)+"_x.npy"
        print(f'x_file is looks like {x_file}')
        x_cal_file = dataset_path+fold+"_"+str(client_id + 779)+"_x.npy"
        x_pre_file = dataset_path+fold+"_"+str(client_id + 779)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id+779)+"_y.npy"
        y_cal_file = dataset_path+fold+"_"+str(client_id + 779)+"_y.npy"
        y_pre_file = dataset_path+fold+"_"+str(client_id + 779)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
        dataset_array[fold+"_cal_x"] = np.load(x_cal_file, allow_pickle=True)
        dataset_array[fold+"_cal_y"] = np.load(y_cal_file, allow_pickle=True)
        dataset_array[fold+"_pre_x"] = np.load(x_pre_file, allow_pickle=True)
        dataset_array[fold+"_pre_y"] = np.load(y_pre_file, allow_pickle=True)
    
    # print(len(dataset_array["train_y"])) # length of trainset of each client is 180 
    
    # print(dataset_array["train_y"].shape) # each data sequence has 120 time steps, [180, 120]
    # print(f"\n--------------------")
    # print(dataset_array["train_y"][0]) 
    # print(f"\n--------------------")
    # print(dataset_array["train_y"][0].shape) # 120
    # print(dataset_array["test_x"].shape) # length of testset and valiset of each client is 10, [10, 120]
    # print(f"\n--------------------")
    # print(dataset_array["test_x"][0]) 
    # print(f"\n--------------------")
    # print(dataset_array["test_x"][0].shape) # 120
    # print(dataset_array["train_cal_x"].shape) # length of calibset and valiset of each client is 180, [180, 120]
    # print(f"\n--------------------")
    # print(dataset_array["train_cal_x"][0])
    # print(f"\n--------------------")
    # print(dataset_array["train_cal_x"][0].shape) # 120

    """
    
    length of pretrainset, trainset and calibset is 180. length of testset and valset is 10. 

    each data is a sequence with 120 time steps. (96 time steps + 24 time units)
    
    each data is based on previous 120 time steps to predict future 24 time units. 

    """
    # exit(0)

    print(len(dataset_array["train_x"])) # 22132 ？
    
    public_len = int(0.0041*len(dataset_array["train_x"])) # 90
    cal_len = int(0.00815*len(dataset_array["train_cal_x"])) # 90
    pre_len = int(0.0122*len(dataset_array["train_pre_x"])) # 90

    print(public_len, cal_len, pre_len) # 90， 180， 270

    # exit(0)

    train_dataset = SequenceDataset(dataset_array["train_x"][:public_len], dataset_array["train_y"][:public_len])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])
    calib_dataset = SequenceDataset(dataset_array["train_cal_x"][public_len:cal_len], dataset_array["train_cal_y"][public_len:cal_len])
    pre_dataset = SequenceDataset(dataset_array["train_pre_x"][cal_len:pre_len], dataset_array["train_pre_y"][cal_len:pre_len])

    print(dataset_array["train_x"][public_len:].shape, dataset_array["train_y"][public_len:].shape) # (90, 120) (90, 24)
    print(dataset_array["val_x"].shape, dataset_array["val_y"].shape) # (10, 120) (10, 24)

    exit(0)

    """

    train_dataset = seqdataset([90, 120], [90, 24])
    the train_dataset will be a dataset with 90 data, and each data has 120 features as x, and 24 features as y. 
    
    """

    dataset_len = [len(train_dataset), len(calib_dataset), len(pre_dataset), len(val_dataset), len(test_dataset)] # 90, 90, 90, 10, 10

    train_loader_private = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    calib_loader_private = DataLoader(calib_dataset, batch_size=64, shuffle=True, drop_last=True)
    pre_loader_private = DataLoader(pre_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=5, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=5, drop_last=True)

    """
    
    The batch size of val_loader and test_loader should not be greater than 10
    
    because the length of testset and valset is 10.
    
    """

    # for i, batch in enumerate(train_loader_private):
    #     print(f"Batch {i + 1} length: {len(batch)}")
    # exit(0)

    return train_loader_private, [dataset_array["train_x"][:public_len], dataset_array["train_y"][:public_len]], calib_loader_private, [dataset_array["train_cal_x"][:cal_len], dataset_array["train_cal_y"][:cal_len]], pre_loader_private,[dataset_array["train_pre_x"][:pre_len], dataset_array["train_pre_y"][:pre_len]], val_loader, test_loader, dataset_len

def get_shared_dataset_2(client_id, dataset_name):
    # print(client_id)
    # exit(0)

    # print(1)
    """

    Getting client and shared dataset files by dataset name. 

    Training data is seperated to a small shared group and a large private group.

    Split the dataset into tr, cal, te.

    """
    dataset_array = {}
    if dataset_name == 'fhwa':
        # dataset_path = "hdd/FHWA_dataset/torch_dataset/"
        # dataset_path = "FHWA_dataset/torch_dataset_new/"
        # dataset_path = "torch_dataset_new/"       
        dataset_path = "torch_dataset_ver5/"       
    elif dataset_name == 'sumo':
        dataset_path = "hdd/SUMO_dataset/learn_dataset/"
    elif dataset_name == 'ct':
        # dataset_path = "/home/Datasets/ct_v3/"
        dataset_path = "/home/Datasets/ct_v4_100/"
        # dataset_path = "/home/Datasets/ct_v2/"
        # print(f"\n")
        # print(f'ct dataset is correctly loaded from {dataset_path}.')
        # print(f'\n')
    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    
    # print(len(dataset_array["train_y"])) # length of trainset of each client is 360
    
    # print(dataset_array["train_x"].shape) # each data sequence has 120 time steps, [360, 120]
    # print(f"\n--------------------")
    # print(dataset_array["train_x"][0]) 
    # print(f"\n--------------------")
    # print(dataset_array["train_x"][0].shape) # 120
    # print(dataset_array["test_x"].shape) # length of testset and valiset of each client is 10, [10, 120]
    # print(f"\n--------------------")
    # print(dataset_array["test_x"][0]) 
    # print(f"\n--------------------")
    # print(dataset_array["test_y"][0].shape) # 24
    # print(dataset_array["train_cal_x"].shape) # length of calibset and valiset of each client is 180, [180, 120]
    # print(f"\n--------------------")
    # print(dataset_array["train_cal_x"][0])
    # print(f"\n--------------------")
    # print(dataset_array["train_cal_x"][0].shape) # 120

    """
    
    length of pretrainset, trainset and calibset is 180. length of testset and valset is 10. 

    each data is a sequence with 120 time steps. (96 time steps + 24 time units)
    
    each data is based on previous 120 time steps to predict future 24 time units. 

    """
    # exit(0)

    # print(len(dataset_array["train_x"])) # 369
    # print(len(dataset_array["test_x"]))
    # exit(0)
    
    public_len = int(90/420*len(dataset_array["train_x"])) # 90
    cal_len = int(180/420*len(dataset_array["train_x"])) # 180
    nor_len = int(230/420*len(dataset_array["train_x"]))
    pre_len = int(320/420*len(dataset_array["train_x"])) # 270
    final_len = int(1*len(dataset_array["train_x"])) # 360
    # cp_val_len = 

    # print(public_len, cal_len, pre_len, final_len) # 90， 180， 270, 360

    # exit(0)   

    train_dataset = SequenceDataset(dataset_array["train_x"][:public_len], dataset_array["train_y"][:public_len])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])
    calib_dataset = SequenceDataset(dataset_array["train_x"][public_len:cal_len], dataset_array["train_y"][public_len:cal_len])
    norm_dataset = SequenceDataset(dataset_array["train_x"][cal_len:nor_len], dataset_array["train_y"][cal_len:nor_len])
    pre_dataset = SequenceDataset(dataset_array["train_x"][nor_len:pre_len], dataset_array["train_y"][nor_len:pre_len])
    shared_dataset = SequenceDataset(dataset_array["train_x"][pre_len:final_len], dataset_array["train_y"][pre_len:final_len])

    # print(dataset_array["train_x"][:public_len].shape, dataset_array["train_y"][:public_len].shape) # (90, 120) (90, 24)
    # print(dataset_array["val_x"].shape, dataset_array["val_y"].shape) # (10, 120) (10, 24)

    # exit(0)

    """

    train_dataset = seqdataset([90, 120], [90, 24])
    the train_dataset will be a dataset with 90 data, and each data has 120 features as x, and 24 features as y. 
    
    """

    dataset_len = [len(train_dataset), len(calib_dataset), len(norm_dataset), len(pre_dataset), len(val_dataset), len(test_dataset), len(shared_dataset)] # 90, 90, 90, 20, 21, 90

    # print(dataset_len)

    # exit(0)

    train_loader_private = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    calib_loader_private = DataLoader(calib_dataset, batch_size=90, shuffle=True, drop_last=True)
    pre_loader_private = DataLoader(pre_dataset, batch_size=64, shuffle=True, drop_last=True)
    nor_loader_private = DataLoader(pre_dataset, batch_size=50, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=20, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=20, drop_last=True)
    train_loader_shared = DataLoader(shared_dataset, batch_size=64, shuffle=True, drop_last=True)

    """
    
    The batch size of val_loader and test_loader should not be greater than 10
    
    because the length of testset and valset is 10.
    
    """

    # for i, batch in enumerate(train_loader_private):
    #     print(f"Batch {i + 1} length: {len(batch)}")
    # exit(0)

    # return train_loader_private, [dataset_array["train_x"][pre_len:final_len], dataset_array["train_y"][pre_len:final_len]], calib_loader_private, [dataset_array["train_x"][public_len:cal_len], dataset_array["train_y"][public_len:cal_len]], pre_loader_private,[dataset_array["train_x"][cal_len:pre_len], dataset_array["train_y"][cal_len:pre_len]], val_loader, test_loader, dataset_len
    return train_loader_private, [dataset_array["train_x"][pre_len:final_len], dataset_array["train_y"][pre_len:final_len]], calib_loader_private, pre_loader_private, nor_loader_private, val_loader, test_loader, dataset_len

def qq_cp(num_clients, num_data, converage = 0.1, epsi = 0, mid = False):
    """
    
    m = 20 # number of machines
    n = 100 # numbert of points per machine
    M = calc_matrix_M(m, n, .0, mid=False)
    print(M)

    a = .1 # coverage

    mm = list(np.ravel(M))
    v = min(i for i in mm if i > (1-a))
    k = int(np.where(M == v)[0])
    l = int(np.where(M == v)[1])
    print(k, ', ', l, ', ', M[k, l])
    
    """
    M = calc_matrix_M(num_clients, num_data, epsi, mid)
    mm = list(np.ravel(M))
    v = min(i for i in mm if i > (1-converage))
    k = int(np.where(M == v)[0])
    l = int(np.where(M == v)[1])
    print(k, ', ', l, ', ', M[k, l])
    return k, l

def get_result_file(cp_value, client_quantile, data_quantile):
    """
    
    create a dic to store all necessary result

    """
    result_file = {
        "cp_value": cp_value,
        "client_quantile(k)": client_quantile,
        "data_quantile(l)": data_quantile,
        "length": len(cp_value)
        # "unsorted_client_quantile_list": unsorted_loss,
        # "sorted_client_quantile_list": sorted_loss,
        # "length_quantile_list": len(sorted_loss)
    }

    print(f"result file generated.")

    # print(f"result file looks like {result_file}")

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

    # print(f"result file generated.")

    # print(f"result file looks like {result_file}")

    return result_file

def save_cp_result(cp_path, result_file, model):
    """
    
    save cp result to a path for further 

    """
    if not os.path.isdir(cp_path):
        os.makedirs(cp_path)
    cp_prefix = os.path.join(cp_path, model)
    # print(f"cp_prefix is {cp_prefix}\n -------------------")
    cp_file_name = '{}.json'.format(cp_prefix)
    # print("cp result is saved to {}".format(cp_file_name))
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
    """
    
    save cluster result to a path for further 

    """
    if not os.path.isdir(cluster_path):
        os.makedirs(cluster_path)
    cp_prefix = os.path.join(cluster_path, model)
    # print(f"cp_prefix is {cp_prefix}\n -------------------")
    cp_file_name = '{}_Cluster.json'.format(cp_prefix)
    print("cp result is saved to {}".format(cp_file_name))
    with open(cp_file_name, 'w') as file:
        json.dump(result_file, file, default=default_converter, indent=4)
    


def save_cp_result_with_sep_type(args, client_dataset, global_quantile_array, glob_model, dq, cq):

    if args.sep_type == 'cp':

            global_avg = [sum(sublist) / len(sublist) for sublist in global_quantile_array]
            sorted_pairs = sorted(enumerate(global_avg), key=lambda x: x[1]) # Enumerate and sort by element value
            sorted_indices = [index for index, value in sorted_pairs] # Extract the sorted indices

            # length check
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
                # print(f"cluster {i+1}: {group}")
                quantile_cq = np.quantile(group, cq, axis=0)
                cp_size_cluster = quantile_cq.tolist()
                cluster_file = get_cluster_result(grouped_indices[i], cp_size_cluster)
                final_cp_file[i] = cluster_file
                # final_cp_file[i]["cp_value_cluster_raw_data_list"] = group
                # final_cp_file[i]["raw_data_length"] = len(group)
            
            if args.method == 'FedSTL':
                cluster_path_cp = "hdd/cluster_cp_result_with_cp/"
            
            elif args.method != 'FedSTL':
                cluster_path_cp = f"hdd_{args.method}/cluster_cp_result_with/"

            # cluster_path_cp = "hdd/cluster_cp_result_with_cp/"
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

        # Convert the data
        converted_data = {key: convert_np_int_to_int(value) for key, value in final_cp_file.items()}
        if args.dataset == 'fhwa' and args.method == "FedSTL":
            cluster_path_spec = f"hdd/cluster_cp_result_with_specm_{args.model}_{args.cp_epoch}_{args.client}/"
        
        elif args.dataset == 'ct' and args.method == 'FedSTL':
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



