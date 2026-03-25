import re
import pandas as pd
import ast

# Load the data
file_paths = ['results/all_crimes_one_month.csv', 'results/one_crime_all_months.csv', 'results/specific_node.csv']
for file_path in file_paths:
    df = pd.read_csv(file_path)
    test_results = df['Test Results']

    # Define the regex pattern to match the part to be removed
    if file_path == 'results/specific_node.csv':
        pattern = r",\s*'y_true':\s*array\(\[.*?\],\s*dtype=float32\),\s*'y_pred':\s*array\(\[.*?\],\s*dtype=float32\)\s*}"
    else:
        pattern = r", 'y_true': array\(\[.*?\], dtype=float32\), 'y_pred': array\(\[.*?\], dtype=float32\)}"


    # Function to clean each string in the Series
    def clean_string(s):
        return re.sub(pattern, "}", s, flags=re.DOTALL)

    # Apply the function to each element in the 'Test Results' Series
    df['Cleaned Results'] = test_results.apply(clean_string)

    df['MSE'] = df['Cleaned Results'].apply(lambda x: ast.literal_eval(x)['mse'])
    df['Torch MSE'] = df['Cleaned Results'].apply(lambda x: ast.literal_eval(x)['torch_mse'])
    df['MAE'] = df['Cleaned Results'].apply(lambda x: ast.literal_eval(x)['mae'])
    df['R2'] = df['Cleaned Results'].apply(lambda x: ast.literal_eval(x)['r2'])
    df['RMSE'] = df['Cleaned Results'].apply(lambda x: ast.literal_eval(x)['rmse'])
    df['EV Score'] = df['Cleaned Results'].apply(lambda x: ast.literal_eval(x)['ev_score'])

    # Drop the 'Test Results' and 'Cleaned Results' columns if not needed
    df = df.drop(columns=['Test Results', 'Cleaned Results'])

    print(df)
    
    # Save the cleaned DataFrame to a new CSV file
    new_file_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(new_file_path, index=False)
    
