import numpy as np 
import pickle as pkl 
import networkx as nx 
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh #to get eigen_values
import sys

def sample_mask(idx, l) :
    "vector of dim l with 1 at idx"
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    .x = training feature vecs
    .tx = test feature vecs
    .allx = superset
    .y = training labels
    .ty = test labels
    .ally = all labels
    .graph = dictionary of neighbour nodes for a particular node
    .text.index
    """
    names = ['x', 'y','tx', 'ty','allx', 'ally', 'graph']
    objects =[]
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x,y,tx,ty,allx,ally,graph = tuple(objects)
    index = []
    for line in open("data/ind.{}.test.index".format(dataset_str)) :
        index.append(int(line.strip()))
    test_idx_range = np.sort(index)

    features = sp.vstack((allx, tx)).tolil()
    features[index, : ] = features[test_idx_range, :]

    labels = np.vstack((ally,ty)) 
    labels[index, :] = labels[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

