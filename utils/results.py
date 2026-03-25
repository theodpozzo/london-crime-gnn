import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

def vis_graph(G):
    color_map = {'N': '#04FE99', 
             'NW': '#FECF9B', 
             'E': '#CF9BFF', 
             'EC': '#FCCE68',
             'W': '#9BFF00',
             'SW': '#23D6FF',
             'SE': '#F26B5F',
             'WC': '#5CFFD3'}

    # Extract postcode area using regex
    def extract_area(postcode):
        match = re.match(r'([A-Za-z]+)', postcode)
        if match:
            return match.group(1)
        return None
    
    # Making sure that N is in the North, E in the East, etc.
    def get_initial_positions(G):
        pos = {}
        for node in G.nodes():
            area = extract_area(node)
            if area:
                if 'N' in area:
                    y = 1
                elif 'S' in area:
                    y = -1
                else:
                    y = 0
                if 'E' in area:
                    x = 1
                elif 'W' in area:
                    x = -1
                else:
                    x = 0

                # Random small offset to avoid exact overlaps in the initial placement
                pos[node] = np.array([x, y]) + 0.1 * np.random.rand(2)
            else:
                pos[node] = np.random.rand(2) * 2 - 1
        return pos
    
    # Improved adjust_positions function using Fruchterman-Reingold layout
    def adjust_positions(G, pos, min_dist=0.2, iterations=50, damping=0.05):
        # Fruchterman-Reingold force-directed algorithm
        pos = nx.spring_layout(G, pos=pos, iterations=iterations)
        for _ in range(iterations):
            displacement = {n: np.zeros(2) for n in pos}
            for n1 in pos:
                for n2 in pos:
                    if n1 != n2:
                        delta = pos[n1] - pos[n2]
                        dist = np.linalg.norm(delta)
                        if dist < min_dist:
                            repulsion = (delta / dist) * (min_dist - dist)
                            displacement[n1] += repulsion
                            displacement[n2] -= repulsion
            for n in pos:
                pos[n] += displacement[n] * damping
        return pos
    
    colors = [color_map.get(extract_area(postcode), 'black') for postcode in G.nodes()]

    plt.figure(figsize=(20, 20))
    
    # Using the Kamada-Kawai layout because it looks the cleanest
    pos = nx.kamada_kawai_layout(G)
    initial_pos = get_initial_positions(G)
    pos = adjust_positions(G, initial_pos)
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=750, node_color=colors, edgecolors='black')
    edges = nx.draw_networkx_edges(G, pos, width=1.0, edge_color='black', style='solid', alpha=0.5)
    
    labels = nx.draw_networkx_labels(G, pos, font_size=9)
    
    # Draw edge weights
    # edge_labels = {(u, v): round(data['weight'], 2) for u, v, data in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=6)
    
    plt.suptitle('London Postcode Network (LPN)', fontsize=15)
    plt.title('Using the Kamada-Kawai layout and adjusted with the Fruchterman-Reginold layout', fontsize=10, y=0.95)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_adjacency_heatmap(G, n=None):
    if n is None:
        n = G.number_of_nodes()
        
    # Select the first n nodes
    nodes = sorted(G.nodes)[:n]
    adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    for u, v, data in G.edges(data=True):
        if u in nodes and v in nodes:
            weight = data.get('weight', 1.0)
            adjacency_matrix.loc[u, v] = weight
            adjacency_matrix.loc[v, u] = weight

    if n is not None and n <= 20:
        xtick_fontsize = 10
        ytick_fontsize = 10
    else:
        xtick_fontsize = 4
        ytick_fontsize = 4
        
    plt.figure(figsize=(20, 16))
    sns.heatmap(adjacency_matrix, cbar=True, annot=False, xticklabels=True, yticklabels=True)
    plt.xticks(rotation=90, fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    # plt.title('Adjacency Matrix Heatmap of Edge Weights', fontsize=15, loc='center')
    plt.show()
    
def filter_results(data, filter_param, condition, filter_value):
    # Filter the data based on the filter parameter condition and value provided
    match condition:
        case 'greater':
            return data[data[filter_param] > filter_value]
        case 'less':
            return data[data[filter_param] < filter_value]
        case 'equal':
            return data[data[filter_param] == filter_value]
        case 'not equal':
            return data[data[filter_param] != filter_value]
    
    raise ValueError("Condition not recognized. Use 'less', 'greater' or 'equal'.")

# Function to plot subplots for different plot types
def plot_subplots(data, plot_type, x_params, y_param, category_param=None, nrows=2, ncols=2, bins=30):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    axes = axes.flatten()

    for i, x_param in enumerate(x_params):
        plt.sca(axes[i])
        match plot_type:
            case 'line':
                plot_results_line(data, x_param, y_param, category_param)
            case 'scatter':
                plot_results_scatter(data, x_param, y_param, category_param)
            case 'bar':
                plot_results_bar(data, x_param, y_param, category_param)
            case 'box':
                plot_results_box(data, x_param, y_param, category_param)
            case 'histogram':
                plot_results_histogram(data, y_param, bins)
            case 'heatmap':
                plot_results_heatmap(data, x_param, y_param, category_param)
            case 'violin':
                plot_results_violin(data, x_param, y_param, category_param)
            case 'pie':
                plot_results_pie(data, category_param, y_param)
            case 'pair':
                plt.sca(axes[0])
                plot_results_pair(data)
                break
        
        if plot_type not in ['pie', 'heatmap', 'pair']:
            axes[i].legend()

    plt.tight_layout()
    plt.show()

# Actual plotting functions
def plot_results_line(data, x_param, y_param, category_param):
    categories = data[category_param].unique()
    for category in categories:
        category_data = data[data[category_param] == category]
        plt.plot(category_data[x_param], category_data[y_param], label=category)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} vs {x_param}')

def plot_results_scatter(data, x_param, y_param, category_param):
    categories = data[category_param].unique()
    for category in categories:
        category_data = data[data[category_param] == category]
        plt.scatter(category_data[x_param], category_data[y_param], label=category)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} vs {x_param}')

def plot_results_bar(data, x_param, y_param, category_param):
    sns.barplot(x=x_param, y=y_param, hue=category_param, data=data, palette='Set2')
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} vs {x_param}')
    plt.legend()

def plot_results_pie(data, category_param, value_param):
    pie_data = data.groupby(category_param)[value_param].sum()
    pie_data.plot.pie(autopct='%1.1f%%')
    plt.ylabel('')
    plt.title(f'{value_param} distribution by {category_param}')

def plot_results_box(data, x_param, y_param, category_param):
    sns.boxplot(x=x_param, y=y_param, hue=category_param, data=data)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} distribution across {x_param}')
    plt.legend()

def plot_results_histogram(data, y_param, bins=30):
    plt.hist(data[y_param], bins=bins, edgecolor='k')
    plt.xlabel(y_param)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {y_param}')

def plot_results_heatmap(data, x_param, y_param, value_param):
    pivot_table = data.pivot_table(index=x_param, columns=y_param, values=value_param, aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(f'Heatmap of {value_param} by {x_param} and {y_param}')
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    
def plot_results_pair(data):
    sns.pairplot(data)
    plt.title('Pair Plot')

def plot_results_violin(data, x_param, y_param, category_param):
    if len(data[category_param].unique()) > 2:
        sns.violinplot(x=x_param, y=y_param, hue=category_param, data=data)
    else:
        sns.violinplot(x=x_param, y=y_param, hue=category_param, data=data, split=True)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{y_param} distribution across {x_param}')
    plt.legend()
    
def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()
    
def plot_predicted_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(range(len(y_true)), y_true, alpha=0.5, label='Actual Values', color='b', marker='o')
    plt.scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Predicted Values', color='r', marker='x')
    
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(f'{model_name} - Predicted vs Actual')
    
    plt.legend()
    
    plt.show(block=False)

    
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show(block=False)
    
def plot_correlation_heatmap(data):
    corr = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.show()
    
def display_metrics(test_results, model_name):
    y_true = test_results['y_true']
    y_pred = test_results['y_pred']
    
    print(f'Test MSE: {test_results["mse"]:.4f}')
    print(f'Test MAE: {test_results["mae"]:.4f}')
    print(f'Test R2 Score: {test_results["r2"]:.4f}')
    print(f'Test RMSE: {test_results["rmse"]:.4f}')
    print(f'Test Explained Variance Score: {test_results["ev_score"]:.4f}')

    
    plot_predicted_vs_actual(y_true, y_pred, model_name)
    # plot_residuals(y_true, y_pred)
    




# Plots that I will actually use in the paper
# 3 Box plots displaying MSE: One for each prediction type (Each csv file)

# Load data
data = pd.read_csv('results/all_crimes_one_month_cleaned.csv')
data['Prediction Type'] = 'All Crimes, One Month'
data_2 = pd.read_csv('results/one_crime_all_months_cleaned.csv')
data_2['Prediction Type'] = 'One Crime, All Months'
data_3 = pd.read_csv('results/specific_node_cleaned.csv')
data_3['Prediction Type'] = 'Specific Node'

# Combine data into one DataFrame
all_data = pd.concat([data, data_2, data_3])

# List of metrics to plot
metrics = ['MSE', 'MAE', 'R2', 'RMSE', 'EV Score']

# Function to remove outliers based on IQR
def remove_outliers(df, metric):
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[metric] >= (Q1 - 1.5 * IQR)) & (df[metric] <= (Q3 + 1.5 * IQR))]

# Remove outliers for each metric
for metric in metrics:
    all_data = remove_outliers(all_data, metric)

# Create box plots for each metric

for i, metric in enumerate(metrics, 1):
    plt.figure(figsize=(18, 20))
    plt.plot(i)
    sns.boxplot(data=all_data, x='Prediction Type', y=metric, hue='Model')
    plt.title(f'{metric} distribution across Prediction Types and Models')
    plt.ylabel(metric)
    plt.xlabel('')
    plt.show(block=False)
plt.show()
