import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances


class Clusterer():
	'''
	This class implements the clusterer which performs k-means clustering on the given data.
	'''
	def __init__(self, name):
		if name.lower() == 'kmeans':
			self.method = 'kmeans'
		else:
			raise NotImplementedError("Unrecognized clustering method. Use 'KMeans'")
		
	def _return_dists_from_centroid(self, data: np.ndarray) -> list:
		'''
		This function reuturns the distance from each cluster centroid. It should be called only after running cluster_kmeans.
		Input - (n x d') embeddings in numpy array format
		
		Returns:
		dist_to_closest_centroid - list of size n contains floats representing the distance of each data point from its nearest
		        cluster centroid
		'''
		centroids = self.kmeans.cluster_centers_
		cluster_labels = self.kmeans.labels_
		dist_to_closest_centroid = []

		for i, node in enumerate(data):
			curr_clus = cluster_labels[i]
			curr_clus_center = centroids[curr_clus]
			dd = cosine_distances(node.reshape(1, -1), curr_clus_center.reshape(1, -1)).item()
			dist_to_closest_centroid.append(dd)
		
		return dist_to_closest_centroid

	def cluster_kmeans(self, data:np.ndarray, num_clusters:int) -> None:
		'''
		This function performs KMeans clustering on the given data.
		Inputs:
		data - (n x d') embeddings learned by the WSGCLNet model in numpy array format
		num_clusters - the number of clusters to be found (k)
		'''
		self.kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(data)

	def _return_labels(self):
		'''
		This function returns the cluster assignment labels as a torch tensor. This function should be called only after 
		running cluster_kmeans function.
		'''
		return torch.Tensor(self.kmeans.labels_)