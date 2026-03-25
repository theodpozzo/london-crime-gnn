import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions_on_graph(G, predictions):
    """
    Visualize model predictions on the graph.
    
    Args:
    - G: NetworkX graph object
    - predictions: Dictionary mapping node IDs to predicted values
    """
    node_colors = [predictions[node] for node in G.nodes]
    pos = nx.spring_layout(G)  # Layout for visualizing the graph
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes with color mapped to predicted values
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Blues, node_size=500)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Dummy plot for creating colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(predictions.values()), vmax=max(predictions.values())))
    sm.set_array([])
    plt.colorbar(sm, label='Predicted Value')
    
    plt.title('Model Predictions on Graph', fontsize=15)
    plt.axis('off')  # Turn off axis
    plt.show()

def main():
    # Example: Assuming you have a NetworkX graph G and predictions dictionary
    G = nx.Graph()
    G.add_nodes_from(['E1', 'E2', 'E3'])
    G.add_edges_from([('E1', 'E2'), ('E1', 'E3'), ('E2', 'E3')])
    
    predictions = {
        'E1': 0.8,
        'E2': 0.5,
        'E3': 0.3
    }
    
    visualize_predictions_on_graph(G, predictions)

if __name__ == '__main__':
    main()
