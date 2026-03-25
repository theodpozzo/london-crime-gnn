import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def print_data_info(data):
    file.write("\nData object information:")
    file.write(f"\n {data}")
    file.write("\nDetailed information:")
    # print(f"Number of graphs: {data.num_graphs}")
    file.write(f"\nNumber of nodes: {data.num_nodes}")
    file.write(f"\nNumber of edges: {data.num_edges}")
    file.write(f"\nNumber of node features: {data.num_node_features}")
    file.write(f"\nNumber of edge features: {data.num_edge_features if data.edge_attr is not None else 'No edge features'}")
    file.write(f"\nNumber of classes: {data.y.unique().size(0) if data.y is not None else 'No labels'}")

def print_formatted_dict(data_dict, file):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    crime_types = [
        "Anti-social behaviour",
        "Bicycle theft",
        "Burglary",
        "Criminal damage and arson",
        "Drugs",
        "Other crime",
        "Other theft",
        "Possession of weapons",
        "Public order",
        "Robbery",
        "Shoplifting",
        "Theft from the person",
        "Vehicle crime",
        "Violence and sexual offences"
    ]

    for key, value in data_dict.items():
        file.write(f"\nPostcode: {key}\n")
        # Print the header row
        file.write(" " * 30 + " ".join([f"{month:>6}" for month in months]))
        
        # Print each row with the crime type and corresponding data
        for crime_type, crimes in zip(crime_types, value):
            formatted_crimes = [f"{int(crime):6}" for crime in crimes]
            file.write(f"\n{crime_type:<30}" + " ".join(formatted_crimes))
        
        file.write("\n")  # Add a newline for better readability

def print_adjacency_matrix(G):
    # Get all nodes in the graph and sort them
    nodes = sorted(G.nodes)
    
    # Create an empty adjacency matrix with zeros
    adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    
    # Fill the matrix with edge weights
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0)  # Default to 1.0 if no weight is specified
        adjacency_matrix.loc[u, v] = weight
        adjacency_matrix.loc[v, u] = weight  # If the graph is undirected
    
    # Convert the matrix to a string with formatting
    file.write("\t" + "\t".join(nodes))
    for node in nodes:
        file.write(node + "\t" + "\t".join(f"{adjacency_matrix.loc[node, n]:.4f}" if adjacency_matrix.loc[node, n] != 0 else "0" for n in nodes))

def print_graph_info(G, file):
    file.write(f"\nNumber of nodes: {G.number_of_nodes()}")
    file.write(f"\nNumber of edges: {G.number_of_edges()}")
    file.write(f"\nGraph is connected: {nx.is_connected(G)}")
    file.write(f"\nGraph density: {nx.density(G):.4f}")
    file.write(f"\nGraph diameter: {nx.diameter(G)}")
    file.write(f"\nGraph average clustering coefficient: {nx.average_clustering(G):.4f}")
    file.write(f"\nGraph average shortest path length: {nx.average_shortest_path_length(G):.4f}")
    file.write(f"\nGraph degree assortativity coefficient: {nx.degree_assortativity_coefficient(G):.4f}")
    file.write(f"\nGraph average degree: {np.mean(list(dict(G.degree()).values())):.4f}")
        
def print_edge_weights(G, file):
    for u, v, data in G.edges(data=True):
        file.write(f"\nEdge ({u}, {v}) has weight: {data['weight']:.4f}")
