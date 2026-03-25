import pandas as pd
import ast
from utils.results import *

csv_file_1 = 'experiment_results__all_crimes_one_month.csv'
csv_file_2 = 'experiment_results__one_crime_all_months.csv'
csv_file_3 = 'experiment_results__specific_node.csv'
data_1 = pd.read_csv(csv_file_1)
data_2 = pd.read_csv(csv_file_2)
data_3 = pd.read_csv(csv_file_3)

# Function to safely evaluate a string to a dictionary
def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {}

# Extract the 'Test Results' dictionary into separate columns
for df in [data_1, data_2, data_3]:
    test_results_df = pd.DataFrame(df['Test Results'].tolist())
    df = pd.concat([df, test_results_df], axis=1)

    # df['Torch MSE'] = test_results.apply(lambda x: x.get('torch_mse', None))
    # df['MAE'] = test_results.apply(lambda x: x.get('mae', None))
    # df['R2'] = test_results.apply(lambda x: x.get('r2', None))
    # df['RMSE'] = test_results.apply(lambda x: x.get('rmse', None))
    # df['EV Score'] = test_results.apply(lambda x: x.get('ev_score', None))
    # df['Y True'] = test_results.apply(lambda x: x.get('y_true', None))
    # df['Y Pred'] = test_results.apply(lambda x: x.get('y_pred', None))

# # Drop the original 'Test Results' column if not needed
# data_1 = data_1.drop(columns=['Test Results'])
# data_2 = data_2.drop(columns=['Test Results'])
# data_3 = data_3.drop(columns=['Test Results'])


# data_1 = filter_results(data_1, 'MSE', 'less', 10000)
# data_2 = filter_results(data_2, 'MSE', 'less', 10000)
# data_3 = filter_results(data_3, 'MSE', 'less', 10000)

print(data_1)
print(data_2)
print(data_3)




# x_params = ['Epochs', 'Hidden Channels', 'Learning Rate', 'Weight Decay']
# y_param = 'Test Results'
# category_param = 'Model'

# plot_subplots(data, 'line', x_params, y_param, category_param)
# plot_subplots(data, 'scatter', x_params, y_param, category_param)
# plot_subplots(data, 'bar', x_params, y_param, category_param)
# plot_subplots(data, 'pie', x_params, y_param, category_param)
# plot_subplots(data, 'box', x_params, y_param, category_param)
# plot_subplots(data, 'histogram', x_params, y_param, category_param, bins=30)
# plot_subplots(data, 'pair', x_params, y_param, category_param)
# plot_subplots(data, 'violin', x_params, y_param, category_param)


# x_params_heatmap = ['Epochs', 'Hidden Channels', 'Learning Rate', 'Weight Decay']
# y_param_heatmap = 'Models'
# value_param_heatmap = 'Test MSE'

# plot_subplots(data, 'heatmap', x_params_heatmap, y_param_heatmap, value_param_heatmap)
