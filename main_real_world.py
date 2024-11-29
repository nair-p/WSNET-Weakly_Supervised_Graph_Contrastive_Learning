import argparse
import json
import os.path
import pickle as pkl
import torch
from snorkel.labeling import LFAnalysis

import numpy as np
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score

from models import WSGCLNet
from utils import load_data, pretty_print_results, clean_weak_label_matrix


def get_hyperparams(config_file:str) -> dict:
    '''
    This function takes the path to the config file as a string input and returns a dictionary of hyperparameter settings
    Inputs:
    config_file - string path to the config file

    Returns:
    hparams - dictionary containing all the hyperparameter values
    '''
    if not os.path.isfile(config_file):
        raise OSError(2, "Config file does not exist")
    
    with open(config_file) as f:
        hparams = json.loads(f.read())

    return hparams


def get_args() -> dict:
    '''
    This function reads in all the arguments and returns them as a dictionary
    '''
    parser = argparse.ArgumentParser(
        prog='WSGCL_real',
        description='Program for running WSGCL on real-world datasets',
    )
    parser.add_argument('--data_path', type=str, default = "./", help='path to the pickle file containing the pyg object of the data')
    parser.add_argument('--results_path', type=str, default = "./", help='path to the directory in which results need to be saved')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of folds to run the experiments. Default is 5')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model. Default is 10')
    parser.add_argument('--config_file', type=str, default='./', help='path to the config.json file containing hyperparameter values')
    args = parser.parse_args()
    return args

def train(data, num_epochs, num_splits, config):
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    node_indices = torch.arange(data.num_nodes)

    # tau = 0.5
    # r = 10
    all_results = []
    for fold, (train_mask, test_mask) in enumerate(kf.split(node_indices)):
        print("Creating model...")
        losses = []
        # start = time.time()
        model = WSGCLNet(input_dim=data.num_features, n_classes=data.num_classes, tau=config['tau'], r=config['r'], embedding_dim=64)
    
        for epoch in range(1, num_epochs + 1):
            loss, pred_labels = model.train_model(data, train_mask, epoch)
            train_f1 = f1_score(data.y[train_mask].cpu().numpy(), pred_labels, average='weighted')
            print("Epoch : {:02d}, Loss : {:.4f}, F1 score: {:.4f}".format(epoch, loss.item(), train_f1))
            losses.append(loss.item())

        # data.run_time = time.time()-start
        result = model.test_model(data, test_mask)
        print("Test F1 = ", str(result['weighted avg']['f1-score']))
        all_results.append(result)

    return all_results


if __name__=='__main__':
    args = get_args()
    data = load_data(args.data_path, num_epochs=args.num_epochs)

    configs = get_hyperparams(args.config_file)
    data = clean_weak_label_matrix(data)
    all_results = train(data, args.num_epochs, args.num_splits, configs)
    pkl.dump(all_results, open(args.results_path,'wb'))
