from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy

from clusterer import Clusterer


class ContrastiveLoss(torch.nn.Module):
    '''
    Class definition for the proposed contrastive loss
    '''
    def __init__(self, tau:float=0.5, num_negative_samples_network: int=10):
        super(ContrastiveLoss, self).__init__()
        self.agg_mi_score = torch.vmap(self.compute_mutual_info_score)
        # self.agg_mi_score = functorch.vmap(self.compute_mutual_info_score)
        self.tau = tau
        self.num_negative_samples_network = num_negative_samples_network

    def compute_mutual_info_score(self, embedding:torch.Tensor, positive_example:torch.Tensor, \
                                  all_samples:List[torch.Tensor]) -> torch.float:
        '''
        This function computes the InfoNCE loss given the positive and negative samples
        Inputs:
            embedding - (n x d') torch tensor containing the node embeddings learned by the WSGCLNet
            positive_example - (1 x d') embedding corresponding to the one positive sample 
            all_samples - (1+num_negative_samples_network x d') list of all the negative samples and the positive sample

        Returns:
            score - InfoNCE loss value
        '''
        numerator = torch.matmul(embedding.detach(), positive_example.t())/self.tau
        denominator = torch.stack([torch.matmul(embedding.detach(), selected_feat.t())/self.tau \
            for selected_feat in all_samples], dim=0)
        denominator_sum = torch.logsumexp(denominator, dim=0)

        score = denominator_sum - numerator
        return score 
    
    def weak_label_loss(self, embeddings:torch.Tensor, pos_options:np.ndarray, neg_options:np.ndarray, negative_embs:torch.Tensor) -> torch.float:
        '''
        This function returns the loss from the weak label loss component. It returns the InfoNCE loss given the positives and 
        negatives based on the similarity of weak label distribution.
        Inputs:
            embedding - (n x d') embeddings learned by the WSGCLNet model
            pos_options - indices of possible positives based on the weak label distribution
            neg_options - indices of possible negatives based on the weak label distribution
        
        Returns:
            loss_sums - InfoNCE loss based on the provided positives and negatives
        '''
        loss_sums = []
        positive_examples = torch.unsqueeze(embeddings[pos_options],-1).permute(0,-1, 1)

        # all_negatives = []
        # for sample in range(self.num_negative_samples_network):
        #     if sample >= len(neg_options[0]):
        #         continue
        #     sample_ind = np.array(neg_options[:,sample]).ravel()
        #     neg_examples = embeddings[sample_ind].t()
        #     all_negatives.append(neg_examples)

        all_negatives = []
        for sample in range(self.num_negative_samples_network):
            negative_example_ind = np.random.choice(list(range(len(negative_embs))), size=len(embeddings)) # repetitions possible, no way around it! 
            neg_examples = negative_embs[negative_example_ind].t()
            all_negatives.append(neg_examples)

        all_negatives = torch.stack(all_negatives).permute(2,0,1)
        all_samples = torch.hstack([positive_examples, all_negatives])

        # print(embeddings.shape, positive_examples.shape, all_samples.shape)
        
        loss_sums = torch.squeeze(self.agg_mi_score(embeddings, positive_examples, all_samples),-1)

        return loss_sums
    
    def network_stucture_loss(self, embeddings:torch.Tensor, negative_embs:torch.Tensor, \
                              community_pos_options:np.ndarray, iter_n:int) -> torch.float:
        '''
        This function calculates the InfoNCE loss based on the positives and negatives given by the community structure of the graph
        Inputs:
            embeddings - (n x d') embeddings learned by the WSGCLNet model
            negative_embs - (n x d') negative embeddings obtained from WSGCLNet model by corrupting the node feature matrix
            community_pos_options - precomputed positive pairs depending on the graph community structure
            inter_n - current iteration number helps to choose a different precomputed positive option for each training epoch

        Returns:
            loss_sums - InfoNCE loss given the structure based positives and negatives
        '''

        loss_sums = []
        num_samples = len(embeddings)
        nodes = np.array(list(range(num_samples)))

        positive_example_ind = community_pos_options[nodes][:,iter_n-1]
        positive_examples = torch.unsqueeze(embeddings[positive_example_ind],-1).permute(0,-1, 1)

        all_negatives = []
        for sample in range(self.num_negative_samples_network):
            negative_example_ind = np.random.choice(list(range(len(negative_embs))), size=num_samples) # repetitions possible, no way around it! 
            neg_examples = negative_embs[negative_example_ind].t()
            all_negatives.append(neg_examples)

        all_negatives = torch.stack(all_negatives).permute(2,0,1)
        all_samples = torch.hstack([positive_examples, all_negatives])

        loss_sums = torch.squeeze(self.agg_mi_score(embeddings, positive_examples, all_samples),-1)

        return loss_sums
    
    def embedding_node_importance(self, node_embeddings:torch.Tensor, num_classes:int) -> np.ndarray:
        '''
        This function calculates the node importance based on distance of embedding from kmeans cluster centroid
        Inputs:
            node_embeddings - node embeddings that are robust to weak labels (learned using WSGCLNet)
            num_classes - number of classes in the data

        Returns:
            inf_e - node influence score based on its embeddings
        '''
        clustering_module = Clusterer(name='Kmeans')
        clustering_module.cluster_kmeans(node_embeddings.detach().numpy(), num_clusters=num_classes)
        dist_from_centroids = clustering_module._return_dists_from_centroid(node_embeddings.detach().numpy())
        cluster_labels = clustering_module._return_labels()
        inf_e = []
        sum_dist_from_centroids = sum(dist_from_centroids)

        for node_id in range(len(dist_from_centroids)):
            node_cluster = cluster_labels[node_id]
            cluster_size = len(np.where(cluster_labels == node_cluster)[0])
            dist_from_centroid = dist_from_centroids[node_id]
            inf_e.append(cluster_size * dist_from_centroid / sum_dist_from_centroids)

        return np.array(inf_e)
    
    def calculate_lf_agreement(self, lfs:np.ndarray, num_classes:int) -> np.ndarray:
        '''
        Calculates the entropy of the LF labels as a measure of their agreement/disagreement
        Inputs:
            lfs - (n x m) matrix which shows the output of m LFs for n nodes in the graph
            num_classes - number of classes in the data
        
        Returns:
            E - entropy of LFs
        '''
        lfs = np.array(lfs,dtype=str)
        counts = []
        for lf_row in lfs:
            counts.append(np.unique(lf_row, return_counts=True)[1])
            
        # counts = np.array([np.unique(lf_row, return_counts=True)[1] for lf_row in lfs])
        ents = [entropy(count) for count in counts]
        upper_bound = np.log(num_classes)
        E = [upper_bound-e for e in ents]
        return np.array(E)
    
    def calculate_final_weights(self, embeddings:torch.Tensor, weak_labels: np.ndarray, num_classes:int=3) -> torch.tensor:
        '''
        This function calculates the final weight given entropy and node importance values.
        Inputs:
            embeddings - (n x d') embeddings learned by the WSGCLNet model
            weak_labels - (n x m) matrix which shows the output of m LFs for n nodes in the graph
            num_classes - the number of classes in the data

        Returns:
            scores - (1 x n) tensor of the weights associated with each of the nodes
        '''
        inf_e = self.embedding_node_importance(embeddings, num_classes)
        E = self.calculate_lf_agreement(weak_labels, num_classes)

        scores = []
        for ent, emb_imp in zip(E, inf_e):
            scores.append(ent*emb_imp)
        scores = torch.tensor(scores)
        return scores


    def forward(self, weak_labels:np.ndarray, embeddings:torch.Tensor, preds:torch.Tensor, \
                mv_labels:torch.Tensor, community_pos_options:np.ndarray, train_mask:np.ndarray, \
                weak_label_pos_options:np.ndarray, weak_label_neg_options:np.ndarray, \
                negative_embeddings:torch.Tensor, iter_n:int, num_classes:int, \
                positive_score, negative_score) -> torch.float:
        '''
        This function calculates the three part loss function for WSGCL
        Inputs:
            weak_labels - (n x m) matrix containing the output of the LFs on the data
            embeddings - (n x d') embeddings learned by the WSGCLNet model
            preds - (n x c) output of the softmax activation on the WSGCLNet embeddings
            mv_labels - (n x 1) Majority Vote of the weak labels
            community_pos_options - positive samples based on community membership
            train_mask - training indices
            weak_label_pos_options - positive samples based on weak label distribution
            weak_label_neg_options - negative samples based on weak label distribution
            negative_embeddings - the corrupted embeddings obtained from WSGCLNet
            iter_n - the current epoch number
            num_classes - number of classes in the data (c)
  
        Returns:
            weighted_loss - the weighted sum of the network structure loss, weak label loss and NLL
        '''
        community_based_loss = -torch.mean(torch.log(positive_score + 1e-8) + torch.log(1 - negative_score + 1e-8))
        weak_label_loss = self.weak_label_loss(embeddings[train_mask], weak_label_pos_options, \
                                               weak_label_neg_options, negative_embeddings).mean()
        labels_loss = F.nll_loss(preds, torch.LongTensor(mv_labels), reduction='none')[train_mask]
        node_weights = self.calculate_final_weights(embeddings, weak_labels, num_classes)[train_mask]

        labels_loss = labels_loss * node_weights
        labels_loss = labels_loss.mean()

        weighted_loss = community_based_loss + labels_loss + weak_label_loss
        return weighted_loss
