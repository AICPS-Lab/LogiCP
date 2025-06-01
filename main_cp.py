import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from IoTData import SequenceDataset
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id, cluster_id_property, cluster_explore
from utils_training import get_device, to_device, save_model, get_client_dataset, get_shared_dataset, model_init, get_shared_dataset_1, get_shared_dataset_2, get_result_file, save_cp_result

from config import Config 

def train_clients(clients, client_dataset, glob_model, config):
    local_weights = {c: copy.deepcopy(glob_model.state_dict()) for c in clients}
    for c in clients:
        local = LocalUpdate(config=config, dataset=client_dataset[c], idxs=c)
        net_local = copy.deepcopy(glob_model).to(config.device)
        # Train local model
        local_weights[c], loss = local.train(net=net_local)
        print(f"Client {c}: Loss {loss}")
    return local_weights

def update_global_model(glob_model, local_weights, client_dataset):
    new_glob_state = copy.deepcopy(glob_model.state_dict())
    total_len = sum(client_dataset[c]["len"][0] for c in local_weights.keys())
    for key in new_glob_state.keys():
        new_glob_state[key] = sum(local_weights[c][key] * client_dataset[c]["len"][0] for c in local_weights.keys()) / total_len
    glob_model.load_state_dict(new_glob_state)

def main(config):
    client_dataset = {}
    for c in range(config.client): # 100
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
    glob_model = initialize_model(config).to(config.device)

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        selected_clients = np.random.choice(list(client_dataset.keys()), int(config.frac * len(client_dataset)), replace=False)
        local_weights = train_clients(selected_clients, client_dataset, glob_model, config)
        update_global_model(glob_model, local_weights, client_dataset)

    save_model(glob_model, config)

if __name__ == '__main__':
    config = Config()  # Load configurations
    try:
        main(config)
    except Exception as e:
        print(f"An error occurred: {e}")