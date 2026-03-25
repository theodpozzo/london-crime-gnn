import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.results import *

# File paths to your CSVs
file_paths = ['results/all_crimes_one_month_cleaned.csv', 'results/one_crime_all_months_cleaned.csv', 'results/specific_node_cleaned.csv']

# List of metrics we want to plot
metrics = ['MSE', 'MAE', 'R2', 'RMSE', 'EV Score']

# Process each CSV file
for file_path in file_paths:
    # Load the data from CSV
    df = pd.read_csv(file_path)
    print(f'Processing file: {file_path}')
    
    # Filter results to remove outliers of MSE over 10000
    df = filter_results(df, 'MSE', 'less', 10000)

    # Create a box plot for each metric to compare models
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Model', y=metric, data=df)
        plt.title(f'{metric} Comparison Across Models ({file_path})')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show(block=False)

    # Create a line plot for R2 vs Learning Rate for each model
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Learning Rate', y='R2', hue='Model', data=df, marker='o')
    plt.title(f'R2 vs Learning Rate ({file_path})')
    plt.xlabel('Learning Rate')
    plt.ylabel('R2')
    plt.grid(True)
    plt.show(block=False)

    # Additional line plots for each metric against Learning Rate for each model
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        sns.lineplot(x='Learning Rate', y=metric, hue='Model', data=df, marker='o')
        plt.title(f'{metric} vs Learning Rate ({file_path})')
        plt.xlabel('Learning Rate')
        plt.ylabel(metric)
        plt.grid(True)
        plt.show(block=False)
    plt.show()  # Show all plots for this file
