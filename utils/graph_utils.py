import json
import pandas as pd
import networkx as nx
import re
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler


def create_graph(postcode_to_index):
    G = nx.Graph()

    postcodes = postcode_to_index.keys()
    G.add_nodes_from(postcodes)
    
    """
        This neighbour_links is exhaustive. 
        The data is done from the perspective of the first postcode in the tuple
        So ('E1', 'E2') means E1 is connected to E2
        This is only for neighbouring postcodes
        Each edge is only added once, so ('E1', 'E2') is in the graph but ('E2', 'E1') is not
        Postcodes like E20 and W1P, W1C, N1C, etc have been conjoined with their surroundings for simplicity
        E20 is tiny and surrounded by E15, so it's been conjoined with E15
        W1P and W1C are both tiny and surrounded by W1, so they've been conjoined with W1
        The same for N1C, it's been conjoined with N1
        The same for E1W, it's been conjoined with E1
    """
    neighbour_links = [
        # 'E' postcodes
        ('E1', 'E2'), ('E1', 'E3'), ('E1', 'E14'), ('E1', 'EC2'), ('E1', 'EC3'),
        ('E2', 'E3'), ('E2', 'E8'), ('E2', 'E9'), ('E2', 'N1'),
        ('E3', 'E9'), ('E3', 'E14'), ('E3', 'E15'), ('E3', 'E16'),
        ('E4', 'E17'), ('E4', 'N9'), ('E4', 'N18'),
        ('E5', 'E8'), ('E5', 'E9'), ('E5', 'E10'), ('E5', 'E17'), ('E5', 'N15'), ('E5', 'N16'),
        ('E6', 'E7'), ('E6', 'E12'), ('E6', 'E13'), ('E6', 'E16'), ('E6', 'SE28'),
        ('E7', 'E11'), ('E7', 'E12'), ('E7', 'E13'), ('E7', 'E15'), 
        ('E8', 'E9'), ('E8', 'N1'), ('E8', 'N16'),
        ('E9', 'E10'), ('E9', 'E15'),
        ('E10', 'E11'), ('E10', 'E15'), ('E10', 'E17'), 
        ('E11', 'E12'), ('E11', 'E15'), ('E11', 'E17'), ('E11', 'E18'),
        # E12
        ('E13', 'E15'), ('E13', 'E16'),
        ('E14', 'E16'), ('E14', 'SE8'), ('E14', 'SE10'), ('E14', 'SE16'),
        ('E15', 'E16'),
        ('E16', 'SE7'), ('E16', 'SE10'), ('E16', 'SE18'), ('E16', 'SE28'),
        ('E17', 'E18'), ('E17', 'N15'), ('E17', 'N17'), ('E17', 'N18'), 
        # E18
        
        # 'N' postcodes
        ('N1', 'N5'), ('N1', 'N7'), ('N1', 'N16'), ('N1', 'WC1'), ('N1', 'EC1'), ('N1', 'EC2'), # N1 to E1 and to NW5 are corner cases and have NOT been included
        ('N2', 'N3'), ('N2', 'N6'), ('N2', 'N10'), ('N2', 'N12'), ('N2', 'NW11'),
        ('N3', 'N12'), ('N3', 'NW4'), ('N3', 'NW7'), ('N3', 'NW11'),
        ('N4', 'N5'), ('N4', 'N7'), ('N4', 'N8'), ('N4', 'N15'), ('N4', 'N16'), ('N4', 'N19'), 
        ('N5', 'N7'), ('N5', 'N16'),
        ('N6', 'N8'), ('N6', 'N10'), ('N6', 'N19'), ('N6', 'NW3'), ('N6', 'NW5'), ('N6', 'NW11'), 
        ('N7', 'N19'), ('N7', 'NW5'), # N7 to NW1 is a corner case and has NOT been included
        ('N8', 'N10'), ('N8', 'N15'), ('N8', 'N19'), ('N8', 'N22'),
        ('N9', 'N18'), ('N9', 'N21'), # N9 to N13 is a corner case and has NOT been included
        ('N10', 'N11'), ('N10', 'N12'), ('N10', 'N22'),
        ('N11', 'N12'), ('N11', 'N13'), ('N11', 'N14'), ('N11', 'N20'),
        ('N12', 'N20'), ('N12', 'NW7'), 
        ('N13', 'N14'), ('N13', 'N18'), ('N13', 'N21'), ('N13', 'N22'), # N13 to N17 is a corner case and has NOT been included
        ('N14', 'N21'), # N14 to N20 is a corner case and has NOT been included
        ('N15', 'N16'), ('N15', 'N17'), ('N15', 'N22'), 
        # N16
        ('N17', 'N22'),
        # N18 with N21 and N22 are both corner cases and have NOT been included
        ('N19', 'NW5'),
        ('N20', 'NW7'), 
        # N21
        # N22
        
        # 'NW' postcodes
        ('NW1', 'NW3'), ('NW1', 'NW5'), ('NW1', 'NW8'), ('NW1', 'WC1'), ('NW1', 'W1'),
        ('NW2', 'NW3'), ('NW2', 'NW4'), ('NW2', 'NW6'), ('NW2', 'NW9'), ('NW2', 'NW10'), ('NW2', 'NW11'),
        ('NW3', 'NW5'), ('NW3', 'NW6'), ('NW3', 'NW8'), ('NW3', 'NW11'),
        ('NW4', 'NW7'), ('NW4', 'NW9'), ('NW4', 'NW11'),
        # NW5
        ('NW6', 'NW8'), ('NW6', 'NW10'), ('NW6', 'W9'), ('NW6', 'W10'), 
        ('NW7', 'NW9'), 
        ('NW8', 'W2'), ('NW8', 'W9'), 
        ('NW9', 'NW10'),
        ('NW10', 'W3'), ('NW10', 'W5'), ('NW10', 'W10'), ('NW10', 'W12'),
        # NW11
        
        # 'W' postcodes
        ('W1', 'W2'), ('W1', 'SW1'), ('W1', 'WC2'), ('W1', 'WC1'),
        ('W2', 'W8'), ('W2', 'W9'), ('W2', 'W11'), ('W2', 'SW1'), ('W2', 'SW7'),
        ('W3', 'W4'), ('W3', 'W5'), ('W3', 'W12'),
        ('W4', 'W6'), ('W4', 'SW13'), ('W4', 'SW14'),
        ('W5', 'W13'), # W5 to W7 only neighbour each other at Northfields station, so not included
        ('W6', 'W12'), ('W6', 'W14'), ('W6', 'SW6'), ('W6', 'SW13'),
        ('W7', 'W13'),
        ('W8', 'W11'), ('W8', 'W14'), ('W8', 'SW5'), ('W8', 'SW7'),
        ('W9', 'W10'), # W9 to W11 only neighbour each other at Westbourne Park station, so not included
        ('W10', 'W11'), ('W10', 'W12'),
        ('W11', 'W12'), ('W11', 'W14'),
        ('W12', 'W14'),
        # W13
        ('W14', 'SW5'), ('W14', 'SW6'),
        
        # 'SW' postcodes
        ('SW1', 'SW3'), ('SW1', 'SW7'), ('SW1', 'SW8'), ('SW1', 'WC2'), ('SW1', 'SE1'), ('SW1', 'SE11'),
        ('SW2', 'SW4'), ('SW2', 'SW9'), ('SW2', 'SW12'), ('SW2', 'SW16'), ('SW2', 'SE24'), ('SW2', 'SE27'), #SW2 and SE21 is another corner case, Tulse Hill station
        ('SW3', 'SW7'), ('SW3', 'SW8'), ('SW3', 'SW10'), ('SW3', 'SW11'),
        ('SW4', 'SW8'), ('SW4', 'SW9'), ('SW4', 'SW11'), ('SW4', 'SW12'), 
        ('SW5', 'SW6'), ('SW5', 'SW7'), ('SW5', 'SW10'), 
        ('SW6', 'SW10'), ('SW6', 'SW11'), ('SW6', 'SW13'), ('SW6', 'SW15'), ('SW6', 'SW18'),
        ('SW7', 'SW10'),
        ('SW8', 'SW9'), ('SW8', 'SE11'), # SW8 and SE5 is another corner case, Oval station
        ('SW9', 'SE5'), ('SW9', 'SE24'), # SW9 and SE11 is another corner case, Oval station
        ('SW10', 'SW11'),
        ('SW11', 'SW12'), ('SW11', 'SW18'), # SW11 and SW17 is another corner case, Wandsworth Common station
        ('SW12', 'SW16'), ('SW12', 'SW17'), # SW12 and SW18 is another corner case, Wandsworth Common station
        ('SW13', 'SW14'), ('SW13', 'SW15'),
        ('SW14', 'SW15'), 
        ('SW15', 'SW18'), ('SW15', 'SW19'), 
        ('SW16', 'SW17'), ('SW16', 'SE19'), ('SW16', 'SE27'), 
        ('SW17', 'SW19'), 
        ('SW18', 'SW19'),
        ('SW19', 'SW20'),
        # SW20
        
        # 'SE' postcodes
        ('SE1', 'SE5'), ('SE1', 'SE11'), ('SE1', 'SE15'), ('SE1', 'SE16'), ('SE1', 'SE17'), ('SE1', 'WC2'), ('SE1', 'EC3'), ('SE1', 'EC4'),
        ('SE2', 'SE18'), ('SE2', 'SE28'),
        ('SE3', 'SE7'), ('SE3', 'SE9'), ('SE3', 'SE10'), ('SE3', 'SE12'), ('SE3', 'SE13'), 
        ('SE4', 'SE6'), ('SE4', 'SE8'), ('SE4', 'SE13'), ('SE4', 'SE14'), ('SE4', 'SE15'), ('SE4', 'SE23'),
        ('SE5', 'SE11'), ('SE5', 'SE15'), ('SE5', 'SE17'), ('SE5', 'SE22'), ('SE5', 'SE24'),
        ('SE6', 'SE12'), ('SE6', 'SE13'), ('SE6', 'SE23'), ('SE6', 'SE26'),
        ('SE7', 'SE3'), ('SE7', 'SE10'), ('SE7', 'SE18'), 
        ('SE8', 'SE10'), ('SE8', 'SE13'), ('SE8', 'SE14'), ('SE8', 'SE16'),
        ('SE9', 'SE12'), ('SE9', 'SE18'), 
        ('SE10', 'SE13'),
        ('SE11', 'SE17'),
        ('SE12', 'SE3'),
        # SE13
        ('SE14', 'SE15'), ('SE14', 'SE16'), 
        ('SE15', 'SE16'), ('SE15', 'SE22'), 
        # SE16
        # SE17
        ('SE18', 'SE28'),
        ('SE19', 'SE20'), ('SE19', 'SE21'), ('SE19', 'SE25'), ('SE19', 'SE26'), ('SE19', 'SE27'),
        ('SE20', 'SE25'), ('SE20', 'SE26'), 
        ('SE21', 'SE22'), ('SE21', 'SE23'), ('SE21', 'SE24'), ('SE21', 'SE26'), ('SE21', 'SE27'),
        ('SE22', 'SE23'), ('SE22', 'SE24'), ('SE22', 'SE15'), ('SE22', 'SE17'), ('SE22', 'SE19'), ('SE22', 'SE23'), ('SE22', 'SE26'),
        ('SE23', 'SE26'), 
        ('SE24', 'SE27'), 
        # SE25
        # SE26
        # SE27
        # SE28
        
        # 'EC' postcodes
        ('EC1', 'WC1'), ('EC1', 'EC2'), ('EC1', 'EC4'),
        ('EC2', 'EC3'), ('EC2', 'EC4'),
        ('EC3', 'EC4'), 
        # EC4
        
        # 'WC' postcodes
        ('WC1', 'EC2'), ('WC1', 'WC2')
        # WC2
        
        # Extra small postcodes
        # E1W
        # E20
        # E22
        # N1C (St Pancras International)
        # E98, NW26, N1P and N81 are non-geographic postcode, typically used for PO boxes, large organisations, or special services
        # 
        ]
    G.add_edges_from(neighbour_links)
    return G
    
def create_pyg_data(G, feature_matrix, postcode_to_index, future_targets):
    def g_to_pyg(G):
        data = from_networkx(G)
        features = [G.nodes[node]['features'] for node in G.nodes()]
        data.x = torch.stack(features)
        
        # Add future targets
        y = [G.nodes[node]['future_target'] for node in G.nodes()]
        data.y = torch.stack(y)
        
        return data

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        if node in postcode_to_index:
            node_features = feature_matrix[postcode_to_index[node]]
            G.nodes[node]['features'] = torch.tensor(node_features, dtype=torch.float)
            G.nodes[node]['future_target'] = future_targets[postcode_to_index[node]]
        else:
            G.nodes[node]['features'] = torch.zeros((feature_matrix.shape[1],), dtype=torch.float)
            G.nodes[node]['future_target'] = torch.zeros((14,), dtype=torch.float)
    
    data = g_to_pyg(G)
    
    return data

def gen_edge_weights(G, feature_matrix):
    num_nodes = len(G.nodes)
    edge_weights = {}

    # Normalize feature matrix
    feature_matrix_normalized = StandardScaler().fit_transform(feature_matrix)

    # Iterate through edges of the graph
    for u, v in G.edges():
        index_u = list(G.nodes()).index(u)
        index_v = list(G.nodes()).index(v)

        # Get feature vectors for nodes u and v
        features_u = feature_matrix_normalized[index_u]
        features_v = feature_matrix_normalized[index_v]

        # Calculate cosine similarity
        similarity = np.dot(features_u, features_v) / (np.linalg.norm(features_u) * np.linalg.norm(features_v))

        # Transform similarity to edge weight (mapping [0, 1] to [low weight, high weight])
        edge_weight = (similarity + 1) / 2  # Maps similarity from [-1, 1] to [0, 1]

        # Assign edge weight to dictionary
        edge_weights[(u, v)] = edge_weight
        edge_weights[(v, u)] = edge_weight  # Assuming undirected graph

    # Assign edge weights to the graph
    nx.set_edge_attributes(G, edge_weights, 'weight')

    return G

def data_prep(G, data_dict):
    num_nodes = len(G.nodes)
    num_crime_types = len(data_dict[next(iter(data_dict))])
    num_months = len(data_dict[next(iter(data_dict))][0])

    feature_matrix = np.zeros((num_nodes, num_crime_types, num_months))

    postcode_to_index = {postcode: idx for idx, postcode in enumerate(data_dict.keys())}

    for postcode, idx in postcode_to_index.items():
        features = data_dict[postcode]
        features = np.array(features)
        feature_matrix[idx] = features

    x = feature_matrix.reshape(num_nodes, -1)

    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    edge_attr = []

    for edge in G.edges(data=True):
        i = postcode_to_index[edge[0]]
        j = postcode_to_index[edge[1]]
        weight = edge[2].get('weight', 1.0)
        
        edge_index.append([i, j])
        edge_index.append([j, i])
        
        edge_attr.append(weight)
        edge_attr.append(weight)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data
