import torch
import torch_geometric.utils as utils
import torch_geometric.datasets as Data
import torch_geometric.data as make_dataset
from torch_geometric.data import DataLoader

from snorkel.labeling import labeling_function, LabelingFunction
from snorkel.labeling import PandasLFApplier, LFAnalysis

import os
import json
import pickle as pkl
import numpy as np
import scipy
import scipy.sparse as sp
import networkx as nx
from torch_geometric.utils import to_networkx

from tqdm import tqdm
import pandas as pd
import sys

correct_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

def lf(x, n_classes, prob_of_correct):
	prob_of_abstain = 0.1
	prob_of_predicting = 1 - prob_of_abstain
	prob_of_wrong = prob_of_predicting - prob_of_correct

	
	possible_labels = list(range(n_classes)) + [-1]
	prob_of_classes = np.zeros(shape=n_classes+1) # +1 for abstain
	
	# true_y = x.class_label # the actual label of the sample
	true_y = x.item()

	# prob_of_classes[true_y] = prob_of_predicting*prob_of_correct # the prob of LF giving a label * prob of LF giving right label
	prob_of_classes[true_y] = prob_of_predicting - prob_of_correct
	prob_of_classes[-1] = prob_of_abstain

	class_labels = list(range(n_classes))
	class_labels.remove(true_y)

	for class_label in class_labels:
		prob_of_classes[class_label] = (prob_of_predicting * prob_of_wrong) / len(class_labels)

	selected_label = np.random.choice(possible_labels, size=1, replace=False, p=prob_of_classes)[0] 
	return selected_label


def make_labeling_function(lf_id, n_classes, pa):
	name = f"lf_{lf_id}"
	return LabelingFunction(name, f=lf, resources={"n_classes": n_classes, 'prob_of_correct': pa})

def Dataset(dataset, args=None):
	# this function returns the processed dataset. "dataset" is a string name of the dataset
	# we need to add the following functionality:
	# 1. num_classes
	# 2. num_features
	# 3. dataset[0] should have data.x, data.y, data.edge_index, data.train_mask, data.val_mask, data.test_mask
	dataset_list = ['citeseer', 'pubmed','coauthor','wikics','disease','amazon','github','ppi','test','texas','wisconsin']
	dataset = dataset.strip()
	dataset = dataset.lower()

	if dataset == 'citeseer':
		return getCiteseerData(args)
	elif dataset == 'pubmed':
		return getPubmedData(args)
	elif dataset == 'coauthor':
		return getCoauthorData(args)
	elif dataset == 'disease':
		return getDiseaseData(args)
	elif dataset == 'amazon':
		return getAmazonData(args)
	elif dataset == 'texas':
		return getTexasData(args)
	elif dataset == 'wisconsin':
		return getWisconsinData(args)
	elif dataset == 'arxiv':
		return getArxivData(args)
	elif dataset == 'test':
		return getTestData(args)
	else:
		print("Invalid dataset. Choose from " + str(dataset_list))
		exit()

def change_labels(data, prob_of_correct, n_lfs):
	# this function changes certain labels to make them weak/noisy
	labeled_df = np.array([[]]*len(data.y))
	n_classes = len(np.unique(data.y))
	for n in range(n_lfs):
		n_labels = data.y.cpu().numpy().copy()
		possible_label_indxs = data.train_mask.copy()
		num_labels_to_abstain = int(0.3*len(possible_label_indxs))
		labels_to_abstain = np.random.choice(possible_label_indxs, size=num_labels_to_abstain, replace=False)
		n_labels[labels_to_abstain] = -1

		possible_label_indxs = list(set(possible_label_indxs) - set(labels_to_abstain))

		num_labels_to_change = int(np.ceil((1-prob_of_correct)*len(possible_label_indxs)))
		labels_to_change = np.random.choice(possible_label_indxs, size=num_labels_to_change, replace=False)
		for label_indx in tqdm(labels_to_change):
			actual_label = data.y[label_indx]
			possible_labels = list(range(n_classes))
			possible_labels.remove(actual_label)
			n_labels[label_indx] = np.random.choice(possible_labels, size=1)[0]	
		
		labeled_df = np.concatenate((labeled_df, n_labels.reshape(-1,1)), axis=1)

	return labeled_df

def get_weak_labels(data, args=None):
	# this function takes in a torch_geometric dataset as input and returns a dictionary of different splits using different seeds
	weak_labels = {}
	n_classes = len(np.unique(data.y))

	# data = make_dataset.Data(x=all_data.x, y=all_data.y, edge_index=all_data.edge_index)
	np.random.seed(42)
	num_nodes = data.x.shape[0]
	val_size = int(np.ceil(0.1*num_nodes))
	train_size = num_nodes-val_size

	train_mask = np.zeros(data.x.shape[0])
	val_mask = np.zeros(data.x.shape[0])
	test_mask = np.zeros(data.x.shape[0])

	train_ids = np.random.choice(range(len(data.x)), size=train_size,replace=False)
	rem_ids = [i for i in range(len(data.x)) if i not in train_ids]
	val_ids = np.random.choice(rem_ids, size=val_size,replace=False)
	test_ids = [i for i in range(len(data.x)) if i not in val_ids]

	train_mask[train_ids] = 1
	val_mask[val_ids] = 1
	test_mask[test_ids] = 1

	train_mask = np.nonzero(train_mask)[0]
	val_mask = np.nonzero(val_mask)[0]
	test_mask = np.nonzero(test_mask)[0]

	data.num_classes = n_classes
	data.train_mask = train_mask
	data.test_mask = test_mask
	data.val_mask = val_mask

	n_lfs = 10

	# for pair in tqdm(pairs):
	for pc in correct_probs:
			# pc = pair[1]
			weak_labels[pc] = change_labels(data, pc, n_lfs)

			print("Accuracy = ", pc)
			print(LFAnalysis(L=weak_labels[pc][data.train_mask]).lf_summary(Y=data.y.numpy()[data.train_mask]))
			print()

	data.synth_weak_labels = weak_labels.copy()

	return data

def getArxivData(args=None):
	from ogb.nodeproppred import PygNodePropPredDataset
	all_data = PygNodePropPredDataset(name='ogbn-arxiv')[0] 
	all_data =  get_weak_labels(all_data)
	pkl.dump(all_data, open('data/Mag/all_data.pkl','wb'))
	return all_data

def getTestData(args=None):
	# returns a tiny portion of the citeseer dataset for testing purposes. 
	# Only 10 train nodes and 5 test nodes and 1 seed
	np.random.seed(42)
	num_nodes = 15
	val_size = 5
	train_size = num_nodes-val_size

	adj = np.zeros((num_nodes, num_nodes))
	for i in range(num_nodes):
		for j in range(num_nodes):
			if i != j and np.random.rand() > 0.5:
				adj[i][j] = 1
	features = sp.eye(adj.shape[0]).todense()
	labels = np.random.choice([0,1,2], num_nodes)

	features = torch.FloatTensor(features)
	labels = torch.LongTensor(labels)
	edge_index = [[], []]
	edge_weight = torch.randn(num_nodes)

	for i, row in enumerate(adj):
		for j, col in enumerate(row):
			if col == 1:
				edge_index[0].append(i)
				edge_index[1].append(j)

	data = make_dataset.Data(x=features, y=labels, edge_index=torch.LongTensor(np.array(edge_index)))
	#data.num_features = len(features)

	train_mask = np.zeros(data.x.shape[0])
	val_mask = np.zeros(data.x.shape[0])
	test_mask = np.zeros(data.x.shape[0])

	train_ids = np.random.choice(range(len(data.x)), size=train_size,replace=False)
	rem_ids = [i for i in range(len(data.x)) if i not in train_ids]
	val_ids = np.random.choice(rem_ids, size=val_size,replace=False)
	test_ids = [i for i in range(len(data.x)) if i not in val_ids]

	train_mask[train_ids] = 1
	val_mask[val_ids] = 1
	test_mask[test_ids] = 1

	train_mask = np.nonzero(train_mask)[0]
	val_mask = np.nonzero(val_mask)[0]
	test_mask = np.nonzero(test_mask)[0]

	num_classes = len(np.unique(data.y))
	data.num_classes = num_classes
	data.train_mask = train_mask
	data.test_mask = test_mask
	data.val_mask = val_mask
	data.edge_weight = edge_weight

	return {1: data}
	
def getCiteseerData(args=None):
	# if the data file already exists
	all_data = Data.CitationFull("data/Citeseer/",'Citeseer')[0]
	all_data =  get_weak_labels(all_data)
	pkl.dump(all_data, open('data/Citeseer/all_data.pkl','wb'))
	return all_data

def getPubmedData(args=None):
	# returns train-val-test splits with 5 seeds of Pubmed dataset
	all_data = Data.CitationFull("data/Pubmed/",'Pubmed')[0]
	all_data =  get_weak_labels(all_data)
	pkl.dump(all_data, open('data/Pubmed/all_data.pkl','wb'))
	return all_data

def getAmazonData(args=None):
	# returns train-val-test splits with 5 seeds of Amazon computer dataset
	all_data = Data.Amazon("data/Amazon/",'Computers')[0]
	all_data =  get_weak_labels(all_data)
	pkl.dump(all_data, open('data/Amazon/all_data.pkl','wb'))
	return all_data

def getCoauthorData(args=None):
	# returns train-val-test splits with 5 seeds of Coauthor dataset
	all_data = Data.Coauthor("data/Coauthor/",'Physics')[0]
	all_data = get_weak_labels(all_data)
	pkl.dump(all_data, open('data/Coauthor/all_data.pkl','wb'))
	return all_data

def getTexasData(args=None):
	fulldata = scipy.io.loadmat('data/Texas/texas.mat')
	edge_index = torch.Tensor(fulldata['edge_index'])
	node_feat = torch.Tensor(fulldata['node_feat'])
	label = np.array(fulldata['label'], dtype=np.int64).flatten()
	num_nodes = node_feat.shape[0]
	all_data = make_dataset.Data(x=node_feat, y=label, edge_index=edge_index)

	all_data = get_weak_labels(all_data)
	pkl.dump(all_data, open('data/Texas/all_data.pkl','wb'))
	return all_data

def getWisconsinData(args=None):
	fulldata = scipy.io.loadmat('data/Wisconsin/wisconsin.mat')
	edge_index = torch.Tensor(fulldata['edge_index'])
	node_feat = torch.Tensor(fulldata['node_feat'])
	label = np.array(fulldata['label'], dtype=np.int64).flatten()
	num_nodes = node_feat.shape[0]
	all_data = make_dataset.Data(x=node_feat, y=label, edge_index=edge_index)

	all_data = get_weak_labels(all_data)
	pkl.dump(all_data, open('data/Wisconsin/all_data.pkl','wb'))
	return all_data

def getDiseaseData(args=None):
	def load_disease_data(dataset_str, use_feats, data_path):
		object_to_idx = {}
		idx_counter = 0
		edges = []
		with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
			all_edges = f.readlines()
		for line in all_edges:
			n1, n2 = line.rstrip().split(',')
			if n1 in object_to_idx:
				i = object_to_idx[n1]
			else:
				i = idx_counter
				object_to_idx[n1] = i
				idx_counter += 1
			if n2 in object_to_idx:
				j = object_to_idx[n2]
			else:
				j = idx_counter
				object_to_idx[n2] = j
				idx_counter += 1
			edges.append((i, j))
		adj = np.zeros((len(object_to_idx), len(object_to_idx)))
		for i, j in edges:
			adj[i, j] = 1.  # comment this line for directed adjacency matrix
			adj[j, i] = 1.
		if use_feats:
			features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
		else:
			features = sp.eye(adj.shape[0])
		labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
		return sp.csr_matrix(adj), features.todense(), labels

	adj, feat, labels = load_disease_data("disease_nc", True, "data/Disease/")
	feat = torch.FloatTensor(feat)
	labels = torch.LongTensor(labels)
	num_features = feat.shape[1]
	adj = adj.toarray()
	edge_index = [[], []]

	for i, row in enumerate(adj):
		for j, col in enumerate(row):
			if col == 1:
				edge_index[0].append(i)
				edge_index[1].append(j)

	all_data = make_dataset.Data(x=feat, y=labels, edge_index=torch.LongTensor(np.array(edge_index)))
	all_data = get_weak_labels(all_data)
	pkl.dump(all_data, open('data/Disease/all_data.pkl','wb'))
	return all_data

Dataset(sys.argv[1])
