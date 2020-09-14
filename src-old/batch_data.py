
from numpy.random import shuffle
from math import ceil
import torch
import random
# def batch(data, batch_size):
# 	feats, labs = data
# 	indices = torch.randperm(len(feats))
# 	batches = []
# 	for batch_n in range(len(feats)//batch_size):
# 		features, labels = [], []
# 		batch_inds = indices[batch_n*batch_size:(batch_n+1)*batch_size]
# 		for i in batch_inds:
# 			feature, label = feats[i], labs[i]
# 			features.append(feature)
# 			labels.append(label)
# 		features, labels = torch.stack(features), torch.stack(labels)
# 		batches.append((features, labels))
# 	return batches

def batch(data, batch_size):
	pos_feats, neg_feats = data
	if batch_size%2:
		batch_size += 1
	batches = []
	batch_n = min(len(pos_feats)//(batch_size//2), len(neg_feats) // (batch_size//2))
	for batch_i in range(batch_n):
		batch_features, batch_labels = [None]*batch_size, [None]*batch_size
		batch_indices = torch.randperm(batch_size).tolist()
		while batch_indices:
			# which pos/neg features to pop
			pi = random.randint(0, len(pos_feats)-1)
			ni = random.randint(0, len(neg_feats)-1)
			p = pos_feats[pi]
			n = neg_feats[ni]
			# where to put them
			bpi = random.randint(0, len(batch_indices)-1)
			pi = batch_indices.pop(bpi)
			bni = random.randint(0, len(batch_indices)-1)
			ni = batch_indices.pop(bni)
			# execute
			batch_features[pi] = p
			batch_labels[pi] = torch.tensor([1], dtype=torch.float)
			batch_features[ni] = n
			batch_labels[ni] = torch.tensor([0], dtype=torch.float)
		batches.append((torch.stack(batch_features), torch.stack(batch_labels)))
	return batches



# def batch_split(data, batch_size):
# 	feats, labs = data
# 	indices = torch.randperm(len(feats))
# 	feature_batches, label_batches = [], []
# 	for batch_n in range(len(feats)//batch_size):
# 		features, labels = [], []
# 		batch_inds = indices[batch_n*batch_size:(batch_n+1)*batch_size]
# 		for i in batch_inds:
# 			feature, label = feats[i], labs[i]
# 			features.append(feature)
# 			labels.append(label)
# 		feature_batches.append(features)
# 		label_batches.append(labels)
# 	return feature_batches, label_batches
