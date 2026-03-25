import pandas as pd
import numpy as np
import torch

def load_data(filepath):
    crime_data = pd.read_csv(filepath)
    crime_data['crimetype'] = crime_data['crimetype'].str.strip()
    
    pivot_table = crime_data.pivot_table(
        index='postcode',
        columns=['crimetype', 'month'],
        values='crime_count',
        fill_value=0
    )

    # Reorder the columns to ensure all months (1-12) are present for each crime type
    crime_types = pivot_table.columns.get_level_values('crimetype').unique()
    months = range(1, 13)
    new_columns = pd.MultiIndex.from_product([crime_types, months], names=['crimetype', 'month'])
    pivot_table = pivot_table.reindex(columns=new_columns, fill_value=0)
    
    # Flatten the column multi-index
    pivot_table.columns = [f'{crime}_{month}' for crime, month in pivot_table.columns]

    # Convert the pivot table to a feature matrix
    feature_matrix = pivot_table.values
    data_dict = reshape_data(pivot_table)

    # Mapping postcodes to feature matrix rows
    postcode_to_index = {postcode: idx for idx, postcode in enumerate(pivot_table.index)}
            
    return feature_matrix, postcode_to_index, data_dict
   
# Function to reshape the data for making a dictionary of postcodes and their features
def reshape_data(df):
    postcodes = df.index
    feature_dict = {}
    
    for postcode in postcodes:
        features = df.loc[postcode].values.reshape(14, 12)  # Reshape into 14 rows (crime types) and 12 columns (months)
        feature_dict[postcode] = features.tolist()
    
    return feature_dict

def prepare_features_labels(data, data_dict, prediction_type='one_crime_all_months', node_index=0):
    num_nodes = data.x.shape[0]
    num_crime_types = len(data_dict[next(iter(data_dict))])
    num_months = len(data_dict[next(iter(data_dict))][0])

    if prediction_type == 'one_crime_all_months':
        # All nodes, all months, one crime
        features = data.x[:, :-num_months]  # All nodes, first 13 crimes, all months
        labels = data.x[:, -num_months:]  # All nodes, last crime, all months
        out_channels = num_months
    elif prediction_type == 'all_crimes_one_month':
        # All nodes, one month, all crimes
        features = data.x[:, :-num_crime_types]  # All nodes, all crimes, first 11 months
        labels = data.x[:, -num_crime_types:]  # All nodes, all crimes, last month
        out_channels = num_crime_types
    elif prediction_type == 'specific_node':
        # All crime data for a specific node
        features = data.x.clone()  # Keep all nodes in features
        labels = data.x[node_index]  # Just the target node's data as label
        out_channels = labels.size(0)  # Number of features for the target node

    else:
        raise ValueError("Invalid prediction type")
    
    return features, labels, out_channels

