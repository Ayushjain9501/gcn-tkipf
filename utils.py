import numpy as np 
import pickle as pkl 
import networkx as nx 
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh #to get eigen_values
import sys

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