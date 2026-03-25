from utils.data_loader import *
from utils.graph_utils import *
from utils.train_eval import *
from utils.prints import *
from utils.logs import *
from utils.results import *

from models.gcn import GCN
from models.gat import GAT
from models.sage import GraphSAGE
from models.gin import GIN

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import DataLoader
from datetime import datetime

    # with open('output.txt', 'w') as file:
    #     # file.write("\nPostcode to index mapping:")
    #     # file.write(postcode_to_index)

    #     file.write("\nFormatted data dictionary:")
    #     print_formatted_dict(data_dict, file)

    #     file.write("\nEdge weights:")
    #     print_edge_weights(G, file)

    #     file.write("\nGraph information:")
    #     print_graph_info(G, file)

if __name__ == '__main__':    
    """
        # For each crime, generate a whole graph.
        # This means there should be 14 graphs in total.
        # Each of them will have all nodes, whose features are the crime counts for each month for that crime.
        # The edges will be the same as the original graph.
        
        # This function will create a PyG Data object from the graph.
        # Preparing the data to be used in PyG models: GCN, GAT, GraphSAGE, GINConv"""
    data_matrix, postcode_to_index, data_dict = load_data('./data/csv/crime_count_2.csv')
    num_postcodes, num_features = data_matrix.shape

    G = create_graph(postcode_to_index)
    G = gen_edge_weights(G, data_matrix)
    data = data_prep(G, data_dict)
    
    # print(data)
    
    
    # # Define hyperparameter ranges
    # prediction_types = ['all_crimes_one_month', 'one_crime_all_months', 'specific_node']
    # learning_rates = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # epochs_list = [50, 100, 250, 500, 1000]
    # weight_decays = [5e-2, 5e-3, 5e-4, 5e-5]
    # hidden_channels_list = [32, 64, 128, 256, 512]

    # # for prediction_type in prediction_types:
    # #     log_file = f'./experiment_results__{prediction_type}.csv'
    # #     initialize_log(log_file)
        
    # for prediction_type in prediction_types:        
    #     features, labels, out_channels = prepare_features_labels(data, data_dict, prediction_type)
    #     prepared_data = Data(x=features, edge_index=data.edge_index, edge_attr=data.edge_attr, y=labels)
    #     log_file = f'./experiment_results__{prediction_type}.csv'
    #     for lr in learning_rates:
    #         for epochs in epochs_list:
    #             for weight_decay in weight_decays:
    #                 for hidden_channels in hidden_channels_list:
    #                     for model_class in [GCN, GAT, GraphSAGE, GIN]:
    #                         print(f"{datetime.now().strftime('[%H:%M:%S] ')}\t"
    #                             f"Model: {model_class.__name__:12s} | "
    #                             f"Prediction: {prediction_type:20s} | "
    #                             f"LR: {lr:<7.4f} | "
    #                             f"Epochs: {epochs:<5d} | "
    #                             f"Weight Decay: {weight_decay:<8.5f} | "
    #                             f"Hidden Channels: {hidden_channels:<4d}")
                            
    #                         if prediction_type == 'specific_node':
    #                             # TODO - Currently, only one node is being tested.
    #                             # This can be changed to test all nodes.
    #                             node_index = 0 
    #                             train_and_test_specific_node(prepared_data, model_class, out_channels,
    #                                                             node_index=node_index, epochs=epochs, lr=lr, 
    #                                                             weight_decay=weight_decay, 
    #                                                             hidden_channels=hidden_channels, 
    #                                                             log_file=log_file) 
    #                         else:
    #                             train_and_test(prepared_data, model_class, out_channels, 
    #                                             epochs=epochs, lr=lr, 
    #                                             weight_decay=weight_decay, 
    #                                             hidden_channels=hidden_channels, 
    #                                             log_file=log_file)
                                
                        
  
    # print("\nHeatamp of adjacency matrix")
    # plot_adjacency_heatmap(G, n=20)
    # plot_adjacency_heatmap(G)

    # print("\nVisualizing graph")
    # vis_graph(G)
