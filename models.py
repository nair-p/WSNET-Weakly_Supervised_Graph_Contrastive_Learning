import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch_geometric.nn import GCNConv, SGConv
from utils import get_weak_label_pos_and_neg
from sklearn.decomposition import PCA
from torch_scatter import scatter_mean

from contrastive_loss import ContrastiveLoss


class WSGCLNet(torch.nn.Module):
    '''
    Class definition for the proposed WSGCL method
    '''
    def __init__(self, input_dim:int, n_classes:int, tau:float=0.5, r:int=10, embedding_dim:int=16) -> None:
        super(WSGCLNet, self).__init__()
        '''
        Constructor to initialize all the attributes of the WSGCLNet object
        '''
        self.input_dim = input_dim
        self.output_dim = n_classes
 
        self.layer1 = SGConv(input_dim, embedding_dim, K=2, cached=True)
        self.layer2 = GCNConv(embedding_dim, embedding_dim, \
                improved=True, cached=False, add_self_loops=True, normalize=True)
        self.classifier_layer = torch.nn.Linear(embedding_dim, self.output_dim, bias=False)
        self.readout = torch.nn.Linear(embedding_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)
        self.criterion = ContrastiveLoss(tau=tau, num_negative_samples_network=r) # the proposed 3 part contrastive loss
        self.sigmoid = torch.nn.Sigmoid()
    
    def corrupt(self, x):
        return x[torch.randperm(x.size(0))]

    def get_negative_embeddings(self, data):
        neg_node_feats = self.corrupt(data.x)
        edge_index = data.edge_index.type(torch.LongTensor)
        # intermediate_embedding = self.aggregate_layer(neg_node_feats, data.edge_index)
        intermediate_embedding = self.layer1(neg_node_feats, edge_index)
        intermediate_embedding = self.layer2(intermediate_embedding, edge_index)
        # output_embedding = self.classifier_layer(intermediate_embedding)
        self.negative_embedding = F.relu(intermediate_embedding)
    

    def forward(self, data):
        edge_index = data.edge_index.type(torch.LongTensor)

        # intermediate_embedding = self.aggregate_layer(data.x, edge_index)
        intermediate_embedding = self.layer1(data.x, edge_index)
        intermediate_embedding = self.layer2(intermediate_embedding, edge_index)
        output_embedding = self.classifier_layer(intermediate_embedding)
        self.embedding = F.relu(intermediate_embedding)

        # Compute average embeddings per community
        g = scatter_mean(self.embedding, data.community_labels, dim=0)
        # Negative pair: Corrupted graph
        h_corrupted = self.layer1(self.corrupt(data.x), data.edge_index.type(torch.LongTensor))
        h_corrupted = self.layer2(h_corrupted, data.edge_index.type(torch.LongTensor))
        # Similarity scores
        positive_score = self.sigmoid(self.readout(g[:, None, :] * self.embedding[None, :, :]).mean(dim=0))
        negative_score = self.sigmoid(self.readout(g[:, None, :] * h_corrupted[None, :, :]).mean(dim=0))

        return output_embedding, positive_score, negative_score


    def train_model(self, data, train_mask, iter_n):
        self.train()
        self.optimizer.zero_grad()
        
        logits, positive_score, negative_score = self.forward(data)
        self.get_negative_embeddings(data)
        
        preds = F.log_softmax(logits, dim=1)
        node_id = np.arange(len(data.x))[train_mask]

        pca = PCA(n_components=2)
        weak_label_embs = pca.fit_transform(data.weak_labels)
        (pos_options, neg_options) = get_weak_label_pos_and_neg(data, node_id, data.hlm_labels[train_mask], \
                                                                weak_label_embs[train_mask])

        loss = self.criterion(data.weak_labels, self.embedding, preds, \
                                data.hlm_labels, data.community_pos_options, train_mask, \
                                pos_options, neg_options, \
                                self.negative_embedding, iter_n, data.num_classes, positive_score, negative_score)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        pred_labels = preds.argmax(1).detach().cpu().numpy()[train_mask]

        return loss, pred_labels


    def test_model(self, data, test_mask):
        self.eval()

        with torch.no_grad():
            logits, _, _ = self.forward(data)
            softmax_preds = F.log_softmax(logits, dim=1)

            pred = softmax_preds.argmax(1).cpu().numpy()[test_mask]
            labels = data.y[test_mask]

            results = classification_report(labels, pred, output_dict=True)

        return results