import os
import pickle as pkl
import torch
from typing import Tuple
from torch_geometric.data import Data
import networkx.algorithms.community as nx_comm
import numpy as np
from tqdm import tqdm
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import LFAnalysis
from hyperlm import HyperLabelModel
from torch_geometric.utils import to_networkx
from sklearn.metrics import pairwise_distances


def pretty_print_results(all_results:list, metric: str='f1-score', result_level: str='micro avg', method=None) -> float:
    '''
    This function takes a classification report dictionary as input and prints the avg result of the given metric
    Inputs:
        all_results - list of dictionaries returned by the classification report function of sklearn.metrics.
                      each item in the list corresponds to the results from a seed
        metric - the metric to be averaged (default is f1-score. Choices are precision, recall, f1-score)
        result_level - individual class level result or micro/macro/weighted avg results

    Returns:
        float - prints out & returns the mean and std of the results
    '''
    results = []
    for item in all_results:
        results.append(item[result_level][metric])

    # print(result_level+" " + metric + " avg over 5 seeds = ", str(np.mean(results)) + "+-" + str(np.std(results)))
    if not method:
        return np.mean(results), np.std(results)
    else:
        return np.max(results), np.std(results)


def get_community_pos(data:Data, num_epochs:int) -> Data:
    '''
    This function takes a pyg data object as input and finds the positive samples based on the same community as a node
    Inputs:
        data - pyg data object
        num_epochs - number of training epochs (trick for faster calcualtions)

    Returns:
        data - pyg data object with the positive samples pool for each node added as an attribute
    '''
    nx_graph = to_networkx(data, to_undirected=False)
    data.nx_graph = nx_graph
    data.communities = nx_comm.louvain_communities(nx_graph)

    comm_map = {}
    for ind, cc in enumerate(data.communities):
        for node in cc:
            comm_map[node] = ind
    data.community_mapping = comm_map
    community_labels = []
    for node in range(len(data.x)):
        community_labels.append(data.community_mapping[node])
    
    data.community_labels = torch.tensor(community_labels)

    comm_nbrs = {}
    community_pos_options = []

    for node in tqdm(data.nx_graph.nodes):
        comm = data.community_mapping[node]
        comm_nodes = data.communities[comm].copy()
        if len(comm_nodes) != 1:
            comm_nodes.remove(node)
        comm_nbrs[node] = comm_nodes
        community_pos_options.append(np.random.choice(list(comm_nodes), size=num_epochs))
    
    data.community_pos_options = np.array(community_pos_options)
    return data

def compute_agreement(labels_a, labels_b):
    """
    Compute the agreement score between two samples based on their weak labels.
    
    Parameters:
    - labels_a (list or np.array): Weak labels for sample A. Use -1 for abstains.
    - labels_b (list or np.array): Weak labels for sample B. Use -1 for abstains.
    
    Returns:
    - float: Agreement score between the two samples (0 to 1).
    """
    # Convert to numpy arrays for easier manipulation
    labels_a = np.array(labels_a)
    labels_b = np.array(labels_b)
    
    # Identify non-abstain indices
    non_abstain_indices = (labels_a != -1) & (labels_b != -1)
    
    # Extract the valid (non-abstain) labels
    valid_labels_a = labels_a[non_abstain_indices]
    valid_labels_b = labels_b[non_abstain_indices]
    
    # Compute the number of matches
    matches = np.sum(valid_labels_a == valid_labels_b)
    
    # Compute the agreement score
    total_comparisons = len(valid_labels_a)
    agreement_score = matches / total_comparisons if total_comparisons > 0 else 0.0

    # Compute the number of conflicts
    conflicts = np.sum(valid_labels_a != valid_labels_b)
    
    # Compute the conflict score
    conflict_score = conflicts / total_comparisons if total_comparisons > 0 else 0.0
  
    return agreement_score - conflict_score
    

def remove_row_indices(arr):
    """
    Removes elements corresponding to the row index value (not position) from each row of a 2D NumPy array.

    Parameters:
        arr (numpy.ndarray): Input 2D array.

    Returns:
        numpy.ndarray: A list of rows with the row index value removed.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    
    result = []
    for i, row in enumerate(arr):
        # Filter out elements equal to the row index value
        filtered_row = row[row != i]
        result.append(list(filtered_row))
    
    return result

def get_weak_label_pos_and_neg(data:Data, node_id:np.ndarray, mv_labels:np.ndarray, weak_labels:np.ndarray, min_neg_size:int=5) -> Tuple:
    '''
    This function finds positive and negative samples based on the similarity of the weak label distribution
    Inputs:
        weak_labels - (n x m) array of outputs of the LFs
        min_neg_size - the minimum number of negatives to be sampled. Default is 5

    Returns:
        (pos_options, neg_options) - the positive samples and negative samples according to similarity in weak label distribution
    '''
    weak_label_similarities = pairwise_distances(weak_labels, weak_labels, metric='cosine')
    sorted_order = np.argsort(weak_label_similarities, axis=1)
    pos_options = sorted_order[:, -1] # least distance/most similar
    neg_options = sorted_order[:, :1]

    neg_options = np.array(neg_options)
    pos_options = np.array(pos_options)

    return (pos_options, neg_options)

def clean_weak_label_matrix(data: Data) -> Data:
    '''
    This function cleans up the weak label matrix by removing samples where all LFs return -1 and returns the torch.Data object with cleaned weak labels
    Inputs:
        data - torch.Data object

    Returns:
        data - torch.Data object
    '''
    # Remove LFs that abstain for all the data samples
    mask = ~(data.weak_labels.astype(int) == -1).all(axis=0)
    data.weak_labels = data.weak_labels[:, mask]

    # all LFs return -1
    mask = (data.weak_labels.astype(int) == -1).all(axis=1)
    ind_to_change = np.random.choice(range(len(mask)), size=len(mask), replace=False)
    data.weak_labels[ind_to_change, -1] = np.random.choice(range(data.num_classes), size=len(ind_to_change))

    data.mv_labels = MajorityLabelVoter(cardinality=data.num_classes).predict(data.weak_labels.astype(int))
    hlm = HyperLabelModel()
    data.hlm_labels = hlm.infer(data.weak_labels.astype(int))

    return data

def load_data(filepath: str, num_epochs:int) -> Data:
    '''
    This function reads the data from the given filepath and returns the torch.Data object
    Inputs:
        filepath - string path to the data file as a torch.Data object
        num_epochs - number of training epochs

    Returns:
        data - torch.Data object
    '''

    if not os.path.isfile(filepath):
        return OSError(2, "Path to data file is invalid")

    data = pkl.load(open(filepath, 'rb'))
    data = get_community_pos(data, num_epochs)

    return data