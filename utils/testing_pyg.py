import os
import sys
import torch_geometric
from torch_geometric.datasets import Planetoid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.gcn import GCN
from models.gat import GAT
from models.sage import GraphSAGE
from models.gin import GIN

from utils.train_eval import train_and_test


data = Planetoid(root="tutorial1",name="Cora")

print('Running PyG Test...')

print("Training GCN Model")
train_and_test(data, GCN)

print("Training GAT Model")
train_and_test(data, GAT)

print("Training GraphSAGE Model")
train_and_test(data, GraphSAGE)

print("Training GINConv Model")
train_and_test(data, GIN)
